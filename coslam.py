import os
#os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'


# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json
import cv2

from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
import config
from model.scene_rep import JointEncoding
from model.layer_timing import DeferredCudaTimer
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion

# Pytorch profiler imports
import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity
import os

#nvtx 
import nvtx

from contextlib import contextmanager

@contextmanager
def nvtx_range(name: str):
    """NVTX range using Start/End (cross-thread friendly for profilers)."""
    if torch.cuda.is_available():
        rid = torch.cuda.nvtx.range_start(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_end(rid)
    else:
        # Fallback for CPU-only execution
        yield

import math

# Helpers for patch sampling
def _grid_jitter_toplefts_fast(H, W, ph, pw, sh, sw, seed=None):
    """
    Vectorized generator of non-overlapping top-left coordinates with bounded jitter.
    Assumes sh>=ph and sw>=pw (non-overlap). Returns an array [N,2] of (top,left),
    in a random order (rows are permuted once).
    """
    assert sh >= ph and sw >= pw, "Non-overlap required (sh>=ph and sw>=pw)."
    rng = np.random.default_rng(seed)

    vmax = H - ph
    umax = W - pw
    if vmax < 0 or umax < 0:
        return np.empty((0,2), dtype=np.int32)  # no capacity

    v0 = np.arange(0, vmax + 1, sh, dtype=np.int32)   # shape [Nv]
    u0 = np.arange(0, umax + 1, sw, dtype=np.int32)   # shape [Nu]

    # per-cell jitter bounds (vectorized)
    dv_max_v = np.minimum(sh - ph, vmax - v0)         # [Nv]
    du_max_u = np.minimum(sw - pw, umax - u0)         # [Nu]
    dv_max = dv_max_v[:, None]                        # [Nv,1]
    du_max = du_max_u[None, :]                        # [1,Nu]

    # sample jitter per cell
    dv = rng.integers(0, dv_max + 1, size=(v0.size, u0.size), dtype=np.int32) if np.any(dv_max) else np.zeros((v0.size,u0.size), dtype=np.int32)
    du = rng.integers(0, du_max + 1, size=(v0.size, u0.size), dtype=np.int32) if np.any(du_max) else np.zeros((v0.size,u0.size), dtype=np.int32)

    tops  = (v0[:, None] + dv).astype(np.int32)       # [Nv,Nu]
    lefts = (u0[None, :] + du).astype(np.int32)       # [Nv,Nu]

    # flatten to [N,2] and permute rows once for random patch order
    coords = np.stack([tops.ravel(), lefts.ravel()], axis=1)  # [N,2]
    if coords.shape[0] > 1:
        perm = rng.permutation(coords.shape[0])
        coords = coords[perm]
    return coords

def _patch_indices_colmajor_grouped_fast(H, top, left, ph, pw):
    """
    Vectorized indices for a ph×pw patch at (top,left) in COL-MAJOR order.
    For col-major flatten idx = w*H + h:
      bases = (left .. left+pw-1) * H  -> shape [pw]
      rows  = (top .. top+ph-1)        -> shape [ph]
      indices = bases[:,None] + rows[None,:] -> shape [pw,ph], then ravel('C')
    """
    bases = (np.arange(left, left + pw, dtype=np.int64) * H).reshape(-1, 1)  # [pw,1]
    rows  = np.arange(top,  top  + ph, dtype=np.int64).reshape(1, -1)        # [1,ph]
    return (bases + rows).ravel(order='C')  # length = ph*pw, columns-first


# Config function for the pytorch profiler
def create_profiler_config(
    output_dir="./profiling/PytorchProfiler/office0",
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    experimental_config=None
):
    """Create optimized profiler configuration for SLAM systems"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Advanced configuration for SLAM profiling
    profiler_config = {
        'activities': activities,
        'record_shapes': record_shapes,  # Important for tensor shape analysis
        'with_stack': with_stack,       # Enables stack trace capture
        'with_flops': True,             # Track FLOPS for neural operations
        'profile_memory': True,         # Essential for memory analysis
        'with_modules': True,           # Track PyTorch module execution
    }
    
    return profiler_config

class CoSLAM():
    def __init__(self, config):
        self.config = config
        self.validate_tracking_bucket_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        
        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device)

        timing_cfg = config.get('timing', {})
        mode = timing_cfg.get('mode', 'none')
        self.max_timing_frames = timing_cfg.get('max_frames', None)
        self.disable_timing_eval = bool(timing_cfg.get('disable_eval', True))
        self.coslam_timing = DeferredCudaTimer(
            enabled=(mode == 'coslam'),
            output_csv=timing_cfg.get('coslam_output_csv', './profiling/pass1/office0/coslam_layer_timing.csv'),
            warmup_frames=timing_cfg.get('warmup_frames', 0),
        )

        # Profiling configuration
        self.enable_profiling = False #config.get('profiling', {}).get('enabled', False)
        self.profiler_config = create_profiler_config(
            output_dir=config.get('profiling', {}).get('output_dir', './profiling/PytorchProfiler/office0')
        )
        self.profiler = None
        self.profiling_step = 0
        print("3D Morton sample sort is :", self.config['grid'].get('morton_sort', False))

    def validate_tracking_bucket_config(self):
        bucket_enabled = self.config.get('tracking_bucket', {}).get('enabled', False)
        ba_bucket_enabled = self.config.get('ba_bucket', {}).get('enabled', False)
        naive_enabled = self.tracking_naive_pruning_enabled()
        if bucket_enabled and naive_enabled:
            raise ValueError(
                "Config error: tracking_bucket.enabled and "
                "tracking_naive_pruning.enabled cannot both be true. "
                "Disable naive tracking pruning when using the bucketed sampler."
            )
        if bucket_enabled and self.config.get('training', {}).get('n_samples_d', 0) <= 0:
            raise ValueError(
                "Config error: tracking_bucket.enabled requires "
                "training.n_samples_d > 0."
            )
        if ba_bucket_enabled and self.config.get('training', {}).get('n_samples_d', 0) <= 0:
            raise ValueError(
                "Config error: ba_bucket.enabled requires "
                "training.n_samples_d > 0."
            )
        ba_pool_factor = self.config.get('ba_bucket', {}).get('pool_factor', self.config['mapping']['iters'])
        if ba_bucket_enabled and ba_pool_factor < self.config['mapping']['iters']:
            raise ValueError(
                "Config error: ba_bucket.pool_factor must be >= mapping.iters "
                "so enough buckets exist for bundle adjustment."
            )

    def tracking_naive_pruning_enabled(self):
        naive_cfg = self.config.get('tracking_naive_pruning', {})
        legacy_enabled = self.config.get('training', {}).get('uniform_samples_until_depth', False)
        return bool(naive_cfg.get('enabled', False) or legacy_enabled)

    def tracking_naive_pruning_morton_sort_enabled(self):
        naive_cfg = self.config.get('tracking_naive_pruning', {})
        return bool(naive_cfg.get('morton_sort', self.config['grid'].get('morton_sort', True)))

    def prepare_naive_tracking_pruning(self, target_d, frame_id):
        n_samples_d = self.config['training']['n_samples_d']
        if n_samples_d <= 0:
            return None

        near = self.config['cam']['near']
        far = self.config['cam']['far']
        depth = target_d.squeeze(-1)
        valid_depth = (depth > 0)
        depth_end = depth.clamp(min=near, max=far)
        base_uniform = torch.linspace(near, far, n_samples_d, device=target_d.device)
        # Ablation baseline: counts = (base_uniform[None, :] <= depth_end[:, None]).sum(dim=1).long()
        counts = torch.searchsorted(base_uniform, depth_end.contiguous(), right=True).long()
        counts = torch.where(valid_depth, counts, torch.zeros_like(counts))

        if valid_depth.any():
            n_uniform = max(1, min(int(counts[valid_depth].max().item()), n_samples_d))
        else:
            n_uniform = n_samples_d

        if getattr(self.model, 'collect_uniform_sampling_stats', False):
            self.model.uniform_sampling_stats.record_batch(
                ray_counts_before_depth=counts,
                valid_depth=valid_depth,
                selected_uniform_samples=n_uniform,
                requested_uniform_samples=n_samples_d,
            )

        naive_cfg = self.config.get('tracking_naive_pruning', {})
        if naive_cfg.get('debug_log', False):
            print(
                "[tracking-naive-pruning] "
                f"frame={frame_id} sample={target_d.shape[0]} "
                f"uniform_count={n_uniform} "
                f"morton3d_samples={self.tracking_naive_pruning_morton_sort_enabled()}"
            )

        return n_uniform
    
    def compute_uniform_until_depth_counts(self, target_d):
        near = self.config['cam']['near']
        far = self.config['cam']['far']
        n_samples_d = self.config['training']['n_samples_d']

        depth = target_d.squeeze(-1)
        valid_depth = torch.isfinite(depth) & (depth > 0)
        dense_count = torch.full_like(depth, n_samples_d, dtype=torch.long)

        depth_end = torch.where(valid_depth, depth, torch.full_like(depth, far))
        depth_end = depth_end.clamp(min=near, max=far)
        base_uniform = torch.linspace(near, far, n_samples_d, device=target_d.device)
        # Ablation baseline: counts = (base_uniform[None, :] <= depth_end[:, None]).sum(dim=1).long()
        counts = torch.searchsorted(base_uniform, depth_end.contiguous(), right=True).long()
        counts = counts.clamp(min=1, max=n_samples_d)
        return torch.where(valid_depth, counts, dense_count)

    def sample_unique_indices_randint_optimized(self, population_size, sample_size):
        """
        Rejection sampling for unique indices without allocating a full permutation.
        Good when sample_size is much smaller than population_size, as on Jetson Orin.
        """
        # First draw with a small margin for the expected duplicates.
        draw_size = sample_size + int(sample_size * 0.05)
        candidates = torch.randint(0, population_size, (draw_size,), device=self.device)

        selected = torch.unique(candidates, sorted=False)

        while selected.numel() < sample_size:
            need = sample_size - selected.numel()
            draw_size = need + int(need * 0.5)
            extra = torch.randint(0, population_size, (draw_size,), device=self.device)
            selected = torch.unique(torch.cat((selected, extra)), sorted=False)

        return selected[:sample_size]

    def prepare_tracking_buckets(self, batch, iH, iW, frame_id):
        bucket_cfg = self.config.get('tracking_bucket', {})
        tracking_iters = self.config['tracking']['iter']
        tracking_sample = self.config['tracking']['sample']
        pool_factor = int(bucket_cfg.get('pool_factor', tracking_iters))
        if pool_factor < tracking_iters:
            raise ValueError(
                "Config error: tracking_bucket.pool_factor must be >= tracking.iter "
                "so enough rays exist to form one bucket per tracking iteration."
            )

        H = self.dataset.H - iH * 2
        W = self.dataset.W - iW * 2
        pool_size = pool_factor * tracking_sample
        if pool_size > H * W:
            raise ValueError(
                f"Config error: tracking_bucket pool size {pool_size} exceeds "
                f"available tracking pixels {H * W}."
            )

        pool_indices_cpu = torch.tensor(random.sample(range(H * W), pool_size), dtype=torch.long)
        # Ablation GPU full permutation: pool_indices = torch.randperm(H * W, device=self.device)[:pool_size]
        # Ablation GPU rejection sampling: pool_indices = self.sample_unique_indices_randint_optimized(H * W, pool_size)
        indice_h = pool_indices_cpu % H
        indice_w = pool_indices_cpu // H

        direction = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :]
        rgb = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :]
        depth = batch['depth'].squeeze(0)[iH:-iH, iW:-iW]

        rays_d_cam_pool = direction[indice_h, indice_w, :].to(self.device)
        target_s_pool = rgb[indice_h, indice_w, :].to(self.device)
        target_d_pool = depth[indice_h, indice_w].unsqueeze(-1).to(self.device)
        target_d_pool = torch.where(torch.isfinite(target_d_pool), target_d_pool, torch.zeros_like(target_d_pool))
        counts = self.compute_uniform_until_depth_counts(target_d_pool)

        count_order = torch.argsort(counts)
        usable = pool_factor * tracking_sample
        pool_positions = torch.arange(pool_size, device=self.device)
        bucket_indices = pool_positions[count_order[:usable]].reshape(pool_factor, tracking_sample)
        bucket_counts = counts[count_order[:usable]].reshape(pool_factor, tracking_sample)

        if bucket_cfg.get('shuffle_bucket_order', True):
            bucket_order = torch.randperm(pool_factor, device=self.device)
            bucket_indices = bucket_indices[bucket_order]
            bucket_counts = bucket_counts[bucket_order]

        bucket_indices = bucket_indices[:tracking_iters]
        bucket_counts = bucket_counts[:tracking_iters]

        flat_indices = bucket_indices.reshape(-1)
        bucket_shape = (tracking_iters, tracking_sample)
        uniform_counts = bucket_counts.max(dim=1).values.detach().cpu().tolist()
        target_d_flat = target_d_pool[flat_indices]
        target_d_flat = torch.where(torch.isfinite(target_d_flat), target_d_flat, torch.zeros_like(target_d_flat))

        if bucket_cfg.get('debug_log', False):
            print(
                "[tracking-bucket] "
                f"frame={frame_id} pool={pool_size} buckets={tracking_iters} "
                f"sample={tracking_sample} "
                f"morton3d_samples={bucket_cfg.get('morton_inside_bucket', True)} "
                f"uniform_counts={uniform_counts}"
            )

        return {
            'rays_d_cam': rays_d_cam_pool[flat_indices].reshape(*bucket_shape, 3),
            'target_s': target_s_pool[flat_indices].reshape(*bucket_shape, 3),
            'target_d': target_d_flat.reshape(*bucket_shape, 1),
            'uniform_counts': uniform_counts,
        }

    def bucket_ray_pool_by_uniform_count(self, rays_cpu, ids_cpu, bucket_count, bucket_sample):
        pool_size = bucket_count * bucket_sample
        if pool_size > rays_cpu.shape[0]:
            raise ValueError(
                f"Config error: requested BA bucket pool {pool_size} exceeds "
                f"available sampled rays {rays_cpu.shape[0]}."
            )

        rays_cpu = rays_cpu[:pool_size]
        ids_cpu = ids_cpu[:pool_size]
        rays_d_cam_pool = rays_cpu[..., :3].to(self.device)
        target_s_pool = rays_cpu[..., 3:6].to(self.device)
        target_d_pool = rays_cpu[..., 6:7].to(self.device)
        ids_pool = ids_cpu.to(device=self.device, dtype=torch.int64)
        target_d_pool = torch.where(torch.isfinite(target_d_pool), target_d_pool, torch.zeros_like(target_d_pool))
        counts = self.compute_uniform_until_depth_counts(target_d_pool)

        count_order = torch.argsort(counts)
        bucket_indices = count_order[:pool_size].reshape(bucket_count, bucket_sample)
        bucket_counts = counts[count_order[:pool_size]].reshape(bucket_count, bucket_sample)
        bucket_shape = (bucket_count, bucket_sample)
        flat_indices = bucket_indices.reshape(-1)

        return {
            'rays_d_cam': rays_d_cam_pool[flat_indices].reshape(*bucket_shape, 3),
            'target_s': target_s_pool[flat_indices].reshape(*bucket_shape, 3),
            'target_d': target_d_pool[flat_indices].reshape(*bucket_shape, 1),
            'ids_all': ids_pool[flat_indices].reshape(*bucket_shape),
            'bucket_counts': bucket_counts,
        }

    def prepare_ba_buckets(self, current_rays, cur_frame_id):
        ba_cfg = self.config.get('ba_bucket', {})
        mapping_iters = self.config['mapping']['iters']
        mapping_sample = self.config['mapping']['sample']
        pool_factor = int(ba_cfg.get('pool_factor', mapping_iters))
        if pool_factor < mapping_iters:
            raise ValueError(
                "Config error: ba_bucket.pool_factor must be >= mapping.iters "
                "so enough buckets exist for bundle adjustment."
            )

        num_keyframes = len(self.keyframeDatabase.frame_ids)
        cur_sample = max(
            mapping_sample // num_keyframes,
            self.config['mapping']['min_pixels_cur'],
        )
        kf_pool_size = pool_factor * mapping_sample
        cur_pool_size = pool_factor * cur_sample
        keyframe_population = num_keyframes * self.keyframeDatabase.num_rays_to_save
        current_population = self.dataset.H * self.dataset.W
        if kf_pool_size > keyframe_population:
            raise ValueError(
                f"Config error: ba_bucket keyframe pool size {kf_pool_size} exceeds "
                f"available keyframe rays {keyframe_population}."
            )
        if cur_pool_size > current_population:
            raise ValueError(
                f"Config error: ba_bucket current-frame pool size {cur_pool_size} exceeds "
                f"available current pixels {current_population}."
            )

        kf_rays, kf_ids = self.keyframeDatabase.sample_global_rays(kf_pool_size)
        kf_ids_all = (kf_ids // self.config['mapping']['keyframe_every']).to(torch.int64)

        idx_cur = torch.tensor(random.sample(range(0, current_population), cur_pool_size), dtype=torch.long)
        cur_rays = current_rays[idx_cur, :]
        cur_ids_all = torch.full((cur_pool_size,), -1, dtype=torch.int64)

        preserve_ratio = ba_cfg.get('preserve_current_ratio', True)
        if preserve_ratio:
            kf_buckets = self.bucket_ray_pool_by_uniform_count(
                kf_rays,
                kf_ids_all,
                pool_factor,
                mapping_sample,
            )
            cur_buckets = self.bucket_ray_pool_by_uniform_count(
                cur_rays,
                cur_ids_all,
                pool_factor,
                cur_sample,
            )

            if ba_cfg.get('shuffle_bucket_order', True):
                bucket_order = torch.randperm(pool_factor, device=self.device)
                for buckets in (kf_buckets, cur_buckets):
                    for key in ('rays_d_cam', 'target_s', 'target_d', 'ids_all', 'bucket_counts'):
                        buckets[key] = buckets[key][bucket_order]

            for buckets in (kf_buckets, cur_buckets):
                for key in ('rays_d_cam', 'target_s', 'target_d', 'ids_all', 'bucket_counts'):
                    buckets[key] = buckets[key][:mapping_iters]

            uniform_counts_tensor = torch.maximum(
                kf_buckets['bucket_counts'].max(dim=1).values,
                cur_buckets['bucket_counts'].max(dim=1).values,
            )
            rays_d_cam = torch.cat([kf_buckets['rays_d_cam'], cur_buckets['rays_d_cam']], dim=1)
            target_s = torch.cat([kf_buckets['target_s'], cur_buckets['target_s']], dim=1)
            target_d = torch.cat([kf_buckets['target_d'], cur_buckets['target_d']], dim=1)
            ids_all = torch.cat([kf_buckets['ids_all'], cur_buckets['ids_all']], dim=1)
        else:
            combined_rays = torch.cat([kf_rays, cur_rays], dim=0)
            combined_ids = torch.cat([kf_ids_all, cur_ids_all], dim=0)
            combined_buckets = self.bucket_ray_pool_by_uniform_count(
                combined_rays,
                combined_ids,
                pool_factor,
                mapping_sample + cur_sample,
            )

            if ba_cfg.get('shuffle_bucket_order', True):
                bucket_order = torch.randperm(pool_factor, device=self.device)
                for key in ('rays_d_cam', 'target_s', 'target_d', 'ids_all', 'bucket_counts'):
                    combined_buckets[key] = combined_buckets[key][bucket_order]

            for key in ('rays_d_cam', 'target_s', 'target_d', 'ids_all', 'bucket_counts'):
                combined_buckets[key] = combined_buckets[key][:mapping_iters]

            uniform_counts_tensor = combined_buckets['bucket_counts'].max(dim=1).values
            rays_d_cam = combined_buckets['rays_d_cam']
            target_s = combined_buckets['target_s']
            target_d = combined_buckets['target_d']
            ids_all = combined_buckets['ids_all']

        uniform_counts = uniform_counts_tensor.detach().cpu().tolist()
        if ba_cfg.get('debug_log', False):
            print(
                "[ba-bucket] "
                f"frame={cur_frame_id} pool_kf={kf_pool_size} pool_cur={cur_pool_size} "
                f"buckets={mapping_iters} sample_kf={mapping_sample} sample_cur={cur_sample} "
                f"preserve_current_ratio={preserve_ratio} "
                f"morton3d_samples={ba_cfg.get('morton_inside_bucket', self.config['grid'].get('morton_sort', True))} "
                f"uniform_counts={uniform_counts}"
            )

        return {
            'rays_d_cam': rays_d_cam,
            'target_s': target_s,
            'target_d': target_d,
            'ids_all': ids_all,
            'uniform_counts': uniform_counts,
        }
    
    def start_profiling(self, wait=1, warmup=0, active=10, repeat=1):
        """Start profiling with optimized schedule for SLAM"""
        if not self.enable_profiling:
            return
            
        schedule = torch.profiler.schedule(
            wait=wait,      # Steps to skip at beginning
            warmup=warmup,  # Warmup steps
            active=active,  # Steps to profile
            repeat=repeat   # How many cycles to repeat
        )
        self.profiler = profile(
            **self.profiler_config,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiling/PytorchProfiler/office0/tensorboard/FFM_TR_BA')
        )
        self.profiler.start()
        print("🔍 PyTorch Profiler started for Co-SLAM")

    def step_profiler(self):
        """Step the profiler - call this at each main loop iteration"""
        if self.profiler:
            self.profiler.step()
            self.profiling_step += 1

    def stop_profiling(self):
        """Stop profiling and save results"""
        if self.profiler:
            self.profiler.stop()
            print(f"🔍 PyTorch Profiler stopped after {self.profiling_step} steps")
    
    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('Using axis-angle as rotation representation, identity init would cause inf')
        
        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError
        
    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        self.load_gt_pose() 
    
    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(torch.float32).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(torch.float32).to(self.device)

    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config, 
                                self.dataset.H, 
                                self.dataset.W, 
                                num_kf, 
                                self.dataset.num_rays_to_save, 
                                self.device)
    
    def load_gt_pose(self):
        '''
        Load the ground truth pose
        '''
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[i] = pose
 
    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
    
    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')

    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']

    def select_samples(self, H, W, samples):
        """
        Return [samples] COL-MAJOR indices in [0, H*W) for the given H×W window.

        - Default (mode!='patch'): original RANDOM behavior (python random.sample).
        - mode=='patch': non-overlapping patch/stripe sampling (stripes = ph=1 or pw=1),
        with randomized patch order (once) and preserved intra-patch order.
        Preconditions: sh>=ph and sw>=pw (hard requirement).
        """
        cfg  = self.config.get('sampling_tracking', {})
        mode = cfg.get('mode', 'random')

        # --- keep original random sampling exactly as-is ---
        if mode != 'patch':
            idx = random.sample(range(H * W), int(samples))
            return torch.tensor(idx)  # int64 inferred

        # --- fast PATCH mode ---
        ph = max(1, int(cfg.get('ph', 16)))
        pw = max(1, int(cfg.get('pw', 16)))
        sh = int(cfg.get('sh', ph))  # default tiling
        sw = int(cfg.get('sw', pw))
        seed = cfg.get('seed', None)

        # clamp to window
        ph = min(ph, H); pw = min(pw, W)
        if ph <= 0 or pw <= 0:
            raise ValueError(f"[select_samples] Invalid patch size ph={ph}, pw={pw}")
        if sh < ph or sw < pw:
            raise ValueError(f"[select_samples] Non-overlap required: sh={sh}>=ph={ph} and sw={sw}>=pw={pw}")

        # total capacity check (upper bound)
        Nv = 1 + (H - ph) // sh if H >= ph else 0
        Nu = 1 + (W - pw) // sw if W >= pw else 0
        capacity = Nv * Nu * (ph * pw)
        if capacity < samples:
            raise RuntimeError(
                f"[select_samples] Not enough non-overlapping patch capacity "
                f"({capacity} < {samples}). Increase coverage (reduce sh/sw or increase ph/pw)."
            )

        # generate randomized patch order
        coords = _grid_jitter_toplefts_fast(H, W, ph, pw, sh, sw, seed)
        if coords.shape[0] == 0:
            raise RuntimeError("[select_samples] No valid patch positions for given H,W,ph,pw")

        # preallocate output and fill in one patch at a time (no Python per-element work)
        out = np.empty(samples, dtype=np.int64)
        pos = 0
        per_patch = ph * pw
        # how many patches we need at most
        max_patches = math.ceil(samples / per_patch)

        for (top, left) in coords[:max_patches]:
            patch = _patch_indices_colmajor_grouped_fast(H, int(top), int(left), ph, pw)  # length = per_patch
            need = samples - pos
            if need <= 0:
                break
            take = patch if need >= per_patch else patch[:need]
            n = take.size
            out[pos:pos+n] = take
            pos += n
            if pos >= samples:
                break

        # safety (should be full due to capacity check)
        if pos < samples:
            raise RuntimeError("[select_samples] Internal underfill; please report.")

        return torch.from_numpy(out)



    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]
        
        if smooth and self.config['training']['smooth_weight']>0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'], 
                                                                                  self.config['training']['smooth_vox'], 
                                                                                  margin=self.config['training']['smooth_margin'])
        
        return loss             
    
    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        self.start_profiling(wait=0, warmup=0, active=1, repeat=1)
        self.model.set_timing_context(0, 'First_frame_mapping')
        self.coslam_timing.set_context(0, 'First_frame_mapping')
        with record_function("FF_mapping"):
            print('First frame mapping...')
            with nvtx_range("cpu_to_gpu_c2w_FFM"):
                c2w = batch['c2w'][0].to(self.device)

            self.est_c2w_data[0] = c2w
            self.est_c2w_data_rel[0] = c2w

            with nvtx_range("model_train?_FFM"):
                self.model.train()

            # Training
            for i in range(n_iters):

                with nvtx_range("initialize_map_optimizer_FFM"):
                    self.map_optimizer.zero_grad()

                with nvtx_range("select_samples_FFM"):    
                    """ 
                    To do patch sampling, you should edit the config file of the dataset, and provide patch dimensions
                     """
                    indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])
                
                with nvtx_range("ray_sampling&coords_transform_FFM"):
                    indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
                    rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
                    target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
                    target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

                    rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
                    rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

                # Forward
                with self.coslam_timing.stage('forward_FFM'):
                    with nvtx_range("forward_FFM"):
                        ret = self.model.forward(rays_o, rays_d, target_s, target_d)

                with self.coslam_timing.stage('Loss_calculation_FFM'):
                    with nvtx_range("Loss_calculation_FFM"):
                        loss = self.get_loss_from_ret(ret)

                with self.coslam_timing.stage('backward_FFM'):
                    with nvtx_range("backward_FFM"):
                        loss.backward()

                with self.coslam_timing.stage('map_optim_FFM'):
                    with nvtx_range("map_optim_FFM"):
                        self.map_optimizer.step()

            with nvtx_range("add_keyframe_FFM"):
                # First frame will always be a keyframe
                self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
                if self.config['mapping']['first_mesh']:
                    self.save_mesh(0)
            
            print('First frame mapping done')

        self.model.flush_scene_timing()
        self.coslam_timing.flush()
        self.model.flush_scene_timing()
        self.coslam_timing.flush()
        self.step_profiler()
        return ret, loss

    def current_frame_mapping(self, batch, cur_frame_id):
        '''
        Current frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        if self.config['mapping']['cur_frame_iters'] <= 0:
            return
        print('Current frame mapping...')
        self.model.set_timing_context(cur_frame_id, 'Current_frame_mapping')
        self.coslam_timing.set_context(cur_frame_id, 'Current_frame_mapping')
        
        c2w = self.est_c2w_data[cur_frame_id].to(self.device)

        self.model.train()

        # Training
        for i in range(self.config['mapping']['cur_frame_iters']):
            self.cur_map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])
            
            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            with self.coslam_timing.stage('Forward_CFM'):
                ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            with self.coslam_timing.stage('Loss_calculation_CFM'):
                loss = self.get_loss_from_ret(ret)
            with self.coslam_timing.stage('Backward_CFM'):
                loss.backward()
            with self.coslam_timing.stage('Optimize_map_CFM'):
                self.cur_map_optimizer.step()
        
        self.model.flush_scene_timing()
        self.coslam_timing.flush()
        
        return ret, loss

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss
    
    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot']},
                                               {"params": cur_trans, "lr": self.config[task]['lr_trans']}])
        
        return cur_rot, cur_trans, pose_optimizer
    
    
    def global_BA(self, batch, cur_frame_id):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        self.start_profiling(wait=0, warmup=0, active=1, repeat=1)
        self.model.set_timing_context(cur_frame_id, 'Bundle_Adjustment')
        self.coslam_timing.set_context(cur_frame_id, 'Bundle_Adjustment')
        with record_function("Bundle_Adjustment"):

            pose_optimizer = None
        
            # all the KF poses: 0, 5, 10, ...
            with nvtx_range("stack_KF_poses_BA"):
                poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
            
            # frame ids for all KFs, used for update poses after optimization
            with nvtx_range("which_poses_to_opt_BA"):
                frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))
                if len(self.keyframeDatabase.frame_ids) < 2:
                    poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
                    current_pose = self.est_c2w_data[cur_frame_id][None,...]
                    poses_all = torch.cat([poses_fixed, current_pose], dim=0)
                
                else:
                    poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)   # for BA, the pose of the first frame should stay fixed
                    current_pose = self.est_c2w_data[cur_frame_id][None,...]

                    if self.config['mapping']['optim_cur']:
                        cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                        pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                        poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

                    else:
                        cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
                        pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                        poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
            
            # Set up optimizer
            with nvtx_range("setup_optimizer_BA"):
                self.map_optimizer.zero_grad()
                if pose_optimizer is not None:
                    pose_optimizer.zero_grad()

            with nvtx_range("cat&reshape_BA"):
                current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
                current_rays = current_rays.reshape(-1, current_rays.shape[-1])

            ba_bucket_cfg = self.config.get('ba_bucket', {})
            bucketed_ba = ba_bucket_cfg.get('enabled', False)
            ba_morton_sort = ba_bucket_cfg.get(
                'morton_inside_bucket',
                self.config['grid'].get('morton_sort', True),
            )
            ba_buckets = None
            if bucketed_ba and self.config['mapping']['iters'] > 0:
                with nvtx_range("prepare_ba_buckets"):
                    ba_buckets = self.prepare_ba_buckets(current_rays, cur_frame_id)

            for i in range(self.config['mapping']['iters']):
                with self.coslam_timing.stage('BA_ITER_PROFILE_TOTAL'):
                    with nvtx_range(f"BA_ITER_PROFILE_TOTAL"):
                        # Sample rays with real frame ids
                        # rays [bs, 7]
                        # frame_ids [bs]
                        with self.coslam_timing.stage('sample_global_rays_BA'):
                            with nvtx_range("sample_global_rays_BA"):
                                if bucketed_ba:
                                    rays_d_cam = ba_buckets['rays_d_cam'][i]
                                    target_s = ba_buckets['target_s'][i]
                                    target_d = ba_buckets['target_d'][i]
                                    ids_all = ba_buckets['ids_all'][i]
                                    uniform_count = ba_buckets['uniform_counts'][i]
                                else:
                                    rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

                                    #TODO: Checkpoint...
                                    idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W),max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']))
                                    current_rays_batch = current_rays[idx_cur, :]

                                    rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
                                    ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)
                                    uniform_count = None
                        if not bucketed_ba:
                            with self.coslam_timing.stage('cpu_to_gpu_BA'):
                                with nvtx_range("cpu_to_gpu_gt_BA"):
                                    rays_d_cam = rays[..., :3].to(self.device)
                                    target_s = rays[..., 3:6].to(self.device)
                                    target_d = rays[..., 6:7].to(self.device)

                        # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
                        with self.coslam_timing.stage('resize_tensors_BA'):
                            with nvtx_range("resize_tensors_BA"):
                                rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
                                rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
                                rays_d = rays_d.reshape(-1, 3)

                        with self.coslam_timing.stage('Forward_BA'):
                            with nvtx_range("Forward_BA"):
                                ret = self.model.forward(
                                    rays_o,
                                    rays_d,
                                    target_s,
                                    target_d,
                                    uniform_samples_until_depth=bool(bucketed_ba),
                                    uniform_sample_count=uniform_count,
                                    morton_sort_override=bool(ba_morton_sort) if bucketed_ba else None,
                                )

                        with self.coslam_timing.stage('Loss_calculation_BA'):
                            with nvtx_range("Loss_calculation_BA"):
                                loss = self.get_loss_from_ret(ret, smooth=True)

                        with self.coslam_timing.stage('Backward_BA'):
                            with nvtx_range("Backward_BA"):
                                loss.backward(retain_graph=True)

                        with self.coslam_timing.stage('Optimize_map_pose_BA'):
                            with nvtx_range("Optimize_map_pose_BA"):
                                if (i + 1) % cfg["mapping"]["map_accum_step"] == 0:
                                    if (i + 1) > cfg["mapping"]["map_wait_step"]:
                                        self.map_optimizer.step()
                                    else:
                                        print('Wait update')
                                    self.map_optimizer.zero_grad()

                                if pose_optimizer is not None and (i + 1) % cfg["mapping"]["pose_accum_step"] == 0:
                                    pose_optimizer.step()
                                    # get SE3 poses to do forward pass
                                    pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                                    pose_optim = pose_optim.to(self.device)
                                    # So current pose is always unchanged
                                    if self.config['mapping']['optim_cur']:
                                        poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                                    else:
                                        current_pose = self.est_c2w_data[cur_frame_id][None,...]
                                        # SE3 poses
                                        poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)

                                    # zero_grad here
                                    pose_optimizer.zero_grad()
                        # CRITICAL: Sync here so NCU captures the full duration of kernels 
                        # launched within this NVTX range. but only for ncu not other types of profiling
                        if self.config.get('sync_for_ncu', False):
                            torch.cuda.synchronize()

            with nvtx_range("update_pose_BA"):
                if pose_optimizer is not None and len(frame_ids_all) > 1:
                    for i in range(len(frame_ids_all[1:])):
                        self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
                
                    if self.config['mapping']['optim_cur']:
                        print('Update current pose')
                        self.est_c2w_data[cur_frame_id] = self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
        self.model.flush_scene_timing()
        self.coslam_timing.flush()
        self.step_profiler()
 
    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose using camera motion model
        '''
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev
            
        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id-2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            delta = c2w_est_prev@c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta@c2w_est_prev
        
        return self.est_c2w_data[frame_id]

    def tracking_pc(self, batch, frame_id):
        '''
        Tracking camera pose of current frame using point cloud loss
        (Not used in the paper, but might be useful for some cases)
        '''

        c2w_gt = batch['c2w'][0].to(self.device)

        cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        cur_trans = torch.nn.parameter.Parameter(cur_c2w[..., :3, 3].unsqueeze(0))
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(cur_c2w[..., :3, :3]).unsqueeze(0))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config['tracking']['lr_rot']},
                                               {"params": cur_trans, "lr": self.config['tracking']['lr_trans']}])
        best_sdf_loss = None

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        thresh=0

        if self.config['tracking']['iter_point'] > 0:
            indice_pc = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, self.config['tracking']['pc_samples'])
            rays_d_cam = batch['direction'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_s = batch['rgb'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_d = batch['depth'][:, iH:-iH, iW:-iW].reshape(-1, 1)[indice_pc].to(self.device)

            valid_depth_mask = ((target_d > 0.) * (target_d < 5.))[:,0]

            rays_d_cam = rays_d_cam[valid_depth_mask]
            target_s = target_s[valid_depth_mask]
            target_d = target_d[valid_depth_mask]

            for i in range(self.config['tracking']['iter_point']):
                pose_optimizer.zero_grad()
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)


                rays_o = c2w_est[...,:3, -1].repeat(len(rays_d_cam), 1)
                rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)
                pts = rays_o + target_d * rays_d

                pts_flat = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

                out = self.model.query_color_sdf(pts_flat)

                sdf = out[:, -1]
                rgb = torch.sigmoid(out[:,:3])

                # TODO: Change this
                loss = 5 * torch.mean(torch.square(rgb-target_s)) + 1000 * torch.mean(torch.square(sdf))

                if best_sdf_loss is None:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()

                with torch.no_grad():
                    c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                    if loss.cpu().item() < best_sdf_loss:
                        best_sdf_loss = loss.cpu().item()
                        best_c2w_est = c2w_est.detach()
                        thresh = 0
                    else:
                        thresh +=1
                if thresh >self.config['tracking']['wait_iters']:
                    break

                loss.backward()
                pose_optimizer.step()
        

        if self.config['tracking']['best']:
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]


        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            # Not a keyframe, need relative pose
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta
        print('Best loss: {}, Camera loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))
    
    def tracking_render(self, batch, frame_id):
        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''
        self.start_profiling(wait=0, warmup=0, active=1, repeat=1)
        self.model.set_timing_context(frame_id, 'Tracking')
        self.coslam_timing.set_context(frame_id, 'Tracking')
        with record_function("tracking_render"):
            with nvtx_range("cpu_to_gpu_pose_tr"):
                c2w_gt = batch['c2w'][0].to(self.device)

            # Initialize current pose
            with nvtx_range("initialize_pose_tr"):
                if self.config['tracking']['iter_point'] > 0:
                    cur_c2w = self.est_c2w_data[frame_id]
                else:
                    cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

            indice = None
            best_sdf_loss = None
            thresh=0

            iW = self.config['tracking']['ignore_edge_W']
            iH = self.config['tracking']['ignore_edge_H']
            tracking_bucket_cfg = self.config.get('tracking_bucket', {})
            bucketed_tracking = tracking_bucket_cfg.get('enabled', False)
            naive_tracking_pruning = self.tracking_naive_pruning_enabled()
            bucket_morton_sort = tracking_bucket_cfg.get(
                'morton_inside_bucket',
                self.config['grid'].get('morton_sort', True),
            )
            naive_uniform_count = None
            naive_morton_sort = (
                self.tracking_naive_pruning_morton_sort_enabled()
                if naive_tracking_pruning
                else None
            )
            tracking_buckets = None
            if bucketed_tracking:
                with nvtx_range("prepare_tracking_buckets"):
                    tracking_buckets = self.prepare_tracking_buckets(batch, iH, iW, frame_id)

            cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None,...], mapping=False)

            # Start tracking
            for i in range(self.config['tracking']['iter']):
                with self.coslam_timing.stage('TR_ITER_PROFILE_TOTAL'):
                    with nvtx_range("TR_ITER_PROFILE_TOTAL"):
                        with nvtx_range("initialize_pose_optimizer_tr"):
                            pose_optimizer.zero_grad()
                        with nvtx_range("pos_matrix_from_tensor_tr"):    
                            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                        # Note here we fix the sampled points for optimisation
                        with nvtx_range("select_samples_tr"):
                            if bucketed_tracking:
                                rays_d_cam = tracking_buckets['rays_d_cam'][i]
                                target_s = tracking_buckets['target_s'][i]
                                target_d = tracking_buckets['target_d'][i]
                                uniform_count = tracking_buckets['uniform_counts'][i]
                            elif indice is None:
                                indice = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, self.config['tracking']['sample'])
                            
                                # Slicing
                                indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)
                                rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
                                target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
                                target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
                                if naive_tracking_pruning:
                                    with nvtx_range("prepare_naive_tracking_pruning"):
                                        naive_uniform_count = self.prepare_naive_tracking_pruning(target_d, frame_id)
                        with nvtx_range("Tensor_ops_s&d_tr"):
                            if not bucketed_tracking:
                                uniform_count = naive_uniform_count

                            n_tracking_rays = rays_d_cam.shape[0]
                            rays_o = c2w_est[...,:3, -1].repeat(n_tracking_rays, 1)
                            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

                        with self.coslam_timing.stage('forward_tr'):
                            with nvtx_range("forward_tr"):
                                ret = self.model.forward(
                                    rays_o,
                                    rays_d,
                                    target_s,
                                    target_d,
                                    uniform_samples_until_depth=bool(bucketed_tracking or naive_tracking_pruning),
                                    uniform_sample_count=uniform_count,
                                    morton_sort_override=(
                                        bool(bucket_morton_sort)
                                        if bucketed_tracking
                                        else bool(naive_morton_sort)
                                        if naive_tracking_pruning
                                        else None
                                    ),
                                )
                        
                        with self.coslam_timing.stage('loss_calculation_tr'):
                            with nvtx_range("loss_calculation_tr"):
                                loss = self.get_loss_from_ret(ret)
                        
                        with nvtx_range("sdf_loss_tr"):
                            if best_sdf_loss is None:
                                best_sdf_loss = loss.cpu().item()
                                best_c2w_est = c2w_est.detach()

                        with nvtx_range("c2w_from_matrix_tr"):
                            with torch.no_grad():
                                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                                if loss.cpu().item() < best_sdf_loss:
                                    best_sdf_loss = loss.cpu().item()
                                    best_c2w_est = c2w_est.detach()
                                    thresh = 0
                                else:
                                    thresh +=1
                            
                            if thresh >self.config['tracking']['wait_iters']:
                                break

                        with self.coslam_timing.stage('backward_tr'):
                            with nvtx_range("backward_tr"):
                                loss.backward()

                        with self.coslam_timing.stage('pose_optimizer_tr'):
                            with nvtx_range("pose_optimizer_tr"):
                                pose_optimizer.step()
                        # Only synchronize for NCU to capture the full duration of kernels launched within this NVTX range. but only for ncu not other types of profiling       
                        if self.config.get('sync_for_ncu', False):
                                torch.cuda.synchronize()

            if self.config['tracking']['best']:
                # Use the pose with smallest loss
                self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
            else:
                # Use the pose after the last iteration
                self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

            # Save relative pose of non-keyframes
            with nvtx_range("save_pose_frame_tr"):
                if frame_id % self.config['mapping']['keyframe_every'] != 0:
                    kf_id = frame_id // self.config['mapping']['keyframe_every']
                    kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                    c2w_key = self.est_c2w_data[kf_frame_id]
                    delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
                    self.est_c2w_data_rel[frame_id] = delta
            
            print('Best loss: {}, Last loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))
        self.model.flush_scene_timing()
        self.coslam_timing.flush()
        self.step_profiler()

    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i] 
                poses[i] = delta @ c2w_key
        
        return poses

    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        # Optimizer for BA
        trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
    
        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
        
        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))
        
        # Optimizer for current frame mapping
        if self.config['mapping']['cur_frame_iters'] > 0:
            params_cur_mapping = [{'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
            if not self.config['grid']['oneGrid']:
                params_cur_mapping.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
                 
            self.cur_map_optimizer = optim.Adam(params_cur_mapping, betas=(0.9, 0.99))
        
    
    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'mesh_track{}.ply'.format(i))
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color
        extract_mesh(self.model.query_sdf, 
                        self.config, 
                        self.bounding_box, 
                        color_func=color_func, 
                        marching_cube_bound=self.marching_cube_bound, 
                        voxel_size=voxel_size, 
                        mesh_savepath=mesh_savepath)      
        
    @nvtx.annotate()    
    def run(self):

        self.create_optimizer() 
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])

        # Start Co-SLAM!
        for i, batch in tqdm(enumerate(data_loader)):
            if self.max_timing_frames is not None and i >= int(self.max_timing_frames):
                break
            # Visualisation
            if self.config['mesh']['visualisation']:
                rgb = cv2.cvtColor(batch["rgb"].squeeze().cpu().numpy(), cv2.COLOR_BGR2RGB)
                raw_depth = batch["depth"]
                mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)
                depth_colormap = colormap_image(batch["depth"])
                depth_colormap[:, mask] = 255.
                depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
                image = np.hstack((rgb, depth_colormap))
                cv2.namedWindow('RGB-D'.format(i), cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RGB-D'.format(i), image)
                key = cv2.waitKey(1)

            # First frame mapping
            if i == 0:
                with nvtx_range("First_frame_mapping"):
                    self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            
            # Tracking + Mapping
            else:
                if self.config['tracking']['iter_point'] > 0:
                    self.tracking_pc(batch, i)
                with nvtx_range("Tracking"):
                    self.tracking_render(batch, i)
    
                if i%self.config['mapping']['map_every']==0:
                    self.current_frame_mapping(batch, i)
                    with nvtx_range("Bundle_Adjustment"):
                        self.global_BA(batch, i)

                    
                # Add keyframe
                if i % self.config['mapping']['keyframe_every'] == 0:
                    with nvtx_range("add_Keyframe"):
                        self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
                    print('add keyframe:',i)
            

                if (not self.disable_timing_eval) and i % self.config['mesh']['vis']==0:
                    self.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval'])
                    pose_relative = self.convert_relative_pose()
                    pose_evaluation(self.pose_gt, self.est_c2w_data, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i)
                    pose_evaluation(self.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r', name='output_relative.txt')

                    if cfg['mesh']['visualisation']:
                        cv2.namedWindow('Traj:'.format(i), cv2.WINDOW_AUTOSIZE)
                        traj_image = cv2.imread(os.path.join(self.config['data']['output'], self.config['data']['exp_name'], "pose_r_{}.png".format(i)))
                        # best_traj_image = cv2.imread(os.path.join(best_logdir_scene, "pose_r_{}.png".format(i)))
                        # image_show = np.hstack((traj_image, best_traj_image))
                        image_show = traj_image
                        cv2.imshow('Traj:'.format(i), image_show)
                        key = cv2.waitKey(1)


        self.model.flush_scene_timing()
        self.coslam_timing.flush()
        model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'checkpoint{}.pt'.format(i)) 
        
        self.save_ckpt(model_savepath)
        if not self.disable_timing_eval:
            self.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])
            
            pose_relative = self.convert_relative_pose()
            pose_evaluation(self.pose_gt, self.est_c2w_data, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i)
            pose_evaluation(self.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r', name='output_relative.txt')

        #TODO: Evaluation of reconstruction


if __name__ == '__main__':
            
    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy("coslam.py", os.path.join(save_path, 'coslam.py'))

    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    slam = CoSLAM(cfg)

    slam.run()
