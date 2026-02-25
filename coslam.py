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

#CUDA extension for Morton code
from model.morton_2D import morton2d_keys_cuda

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        
        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device)

        # Profiling configuration
        self.enable_profiling = False #config.get('profiling', {}).get('enabled', False)
        self.profiler_config = create_profiler_config(
            output_dir=config.get('profiling', {}).get('output_dir', './profiling/PytorchProfiler/office0')
        )
        self.profiler = None
        self.profiling_step = 0
        print("Morton2D is :", self.config['grid'].get('morton2D', False))
    
    def sort_rays_by_morton(self, indices=None, rays_d=None, frame_ids=None, H=None, W=None):
        """
        Sorts rays based on 2D Morton codes to improve cache locality.
        Conditioned by config['grid']['morton2D'].
        """
        # 1. Ablation Switch
        if not self.config['grid'].get('morton2D', False):
            return None 

        with nvtx_range("morton_sort_overhead"):
            R = 128 
            
            # --- CASE A: Single Frame (using flattened indices) ---
            if indices is not None:
                # Recover 2D coordinates from flattened indices
                # u = idx % W, v = idx // W
                y = (indices // W).float() / H
                x = (indices % W).float() / W
                uv = torch.stack([x, y], dim=-1).to(self.device) # [N, 2]
                
                # Generate keys (frame_id is None, so it's just pure spatial sort)
                keys = morton2d_keys_cuda(uv, None, R)
                
            # --- CASE B: Multi-Frame / Bundle Adjustment ---
            elif rays_d is not None:
                # We use ray directions as a proxy for screen coordinates.
                # Project or simpler: normalize direction X/Y to [0,1].
                # (We just need relative order, not exact pixels)
                uv = rays_d[:, :2] * 0.5 + 0.5 
                uv = torch.clamp(uv, 0.0, 1.0)
                
                # frame_ids MUST be Int64 for the C++ kernel
                fids = frame_ids.to(torch.int64) if frame_ids is not None else None
                
                # Generate packed keys: [Frame_ID | Morton_Code]
                keys = morton2d_keys_cuda(uv, fids, R)
            
            else:
                return None

            # 2. Sort the keys to get permutation
            perm = torch.argsort(keys)
            
        return perm
    
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
                    # --- START FIX ---
                    # 1. Move to GPU for sorting
                    indice = indice.to(self.device)
                    
                    # 2. Sort
                    perm = self.sort_rays_by_morton(indices=indice, H=self.dataset.H, W=self.dataset.W)
                    if perm is not None:
                        indice = indice[perm]
                    
                    # 3. CRITICAL: Move back to CPU to index the CPU batch tensors
                    indice = indice.cpu()
                    # --- END FIX ---
                
                with nvtx_range("ray_sampling&coords_transform_FFM"):
                    indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
                    rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
                    target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
                    target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

                    rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
                    rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

                # Forward
                with nvtx_range("forward_FFM"):
                    ret = self.model.forward(rays_o, rays_d, target_s, target_d)

                with nvtx_range("Loss_calculation_FFM"):
                    loss = self.get_loss_from_ret(ret)

                with nvtx_range("backward_FFM"):
                    loss.backward()

                with nvtx_range("map_optim_FFM"):
                    self.map_optimizer.step()

            with nvtx_range("add_keyframe_FFM"):
                # First frame will always be a keyframe
                self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
                if self.config['mapping']['first_mesh']:
                    self.save_mesh(0)
            
            print('First frame mapping done')

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
        
        c2w = self.est_c2w_data[cur_frame_id].to(self.device)

        self.model.train()

        # Training
        for i in range(self.config['mapping']['cur_frame_iters']):
            self.cur_map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])
            # --- START FIX ---
            indice = indice.to(self.device)
            perm = self.sort_rays_by_morton(indices=indice, H=self.dataset.H, W=self.dataset.W)
            if perm is not None:
                indice = indice[perm]
            indice = indice.cpu() # <--- Move back to CPU
            # --- END FIX ---
            
            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.cur_map_optimizer.step()
        
        
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
            

            for i in range(self.config['mapping']['iters']):
                with nvtx_range(f"BA_ITER_PROFILE_TOTAL_BA"):
                    # Sample rays with real frame ids
                    # rays [bs, 7]
                    # frame_ids [bs]
                    with nvtx_range("sample_global_rays_BA"):
                        rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

                    with nvtx_range("sampling&cat_BA"):
                        #TODO: Checkpoint...
                        idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W),max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']))
                        current_rays_batch = current_rays[idx_cur, :]

                        rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
                        ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)
                        # --- NEW: Sort Multi-Frame Rays ---
                        perm = self.sort_rays_by_morton(
                            rays_d=rays[..., :3].to(self.device), 
                            frame_ids=ids_all.to(self.device)
                        )
                        
                        if perm is not None:
                            # Fix: move perm to cpu() because 'rays' is likely on CPU
                            perm = perm.cpu() 
                            rays = rays[perm]
                            ids_all = ids_all[perm]
                        # ----------------------------------

                    with nvtx_range("cpu_to_gpu_gt_BA"):
                        rays_d_cam = rays[..., :3].to(self.device)
                        target_s = rays[..., 3:6].to(self.device)
                        target_d = rays[..., 6:7].to(self.device)

                    # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
                    with nvtx_range("resize_tensors_BA"):
                        rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
                        rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
                        rays_d = rays_d.reshape(-1, 3)

                    with nvtx_range("Forward_BA"):
                        ret = self.model.forward(rays_o, rays_d, target_s, target_d)

                    with nvtx_range("Loss_calculation_BA"):
                        loss = self.get_loss_from_ret(ret, smooth=True)

                    with nvtx_range("Backward_BA"):
                        loss.backward(retain_graph=True)

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
                    # launched within this NVTX range.
                    torch.cuda.synchronize()

            with nvtx_range("update_pose_BA"):
                if pose_optimizer is not None and len(frame_ids_all) > 1:
                    for i in range(len(frame_ids_all[1:])):
                        self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
                
                    if self.config['mapping']['optim_cur']:
                        print('Update current pose')
                        self.est_c2w_data[cur_frame_id] = self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
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

            cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None,...], mapping=False)

            # Start tracking
            for i in range(self.config['tracking']['iter']):
                with nvtx_range("TR_ITER_PROFILE_TOTAL_tr"):
                    with nvtx_range("initialize_pose_optimizer_tr"):
                        pose_optimizer.zero_grad()
                    with nvtx_range("pos_matrix_from_tensor_tr"):    
                        c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                    # Note here we fix the sampled points for optimisation
                    with nvtx_range("select_samples_tr"):
                        if indice is None:
                            indice = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, self.config['tracking']['sample'])
                            # --- START FIX ---
                            indice = indice.to(self.device)
                            perm = self.sort_rays_by_morton(indices=indice, H=self.dataset.H, W=self.dataset.W)
                            if perm is not None:
                                indice = indice[perm]
                            indice = indice.cpu() # <--- Move back to CPU
                            # --- END FIX ---
                        
                            # Slicing
                            indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)
                            rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
                    with nvtx_range("Tensor_ops_s&d_tr"):
                        target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
                        target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)

                        rays_o = c2w_est[...,:3, -1].repeat(self.config['tracking']['sample'], 1)
                        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

                    with nvtx_range("forward_tr"):
                        ret = self.model.forward(rays_o, rays_d, target_s, target_d)
                    
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

                    with nvtx_range("backward_tr"):
                        loss.backward()

                    with nvtx_range("pose_optimizer_tr"):
                        pose_optimizer.step()
            
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
            if i>20:
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
            

                if i % self.config['mesh']['vis']==0:
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


        model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'checkpoint{}.pt'.format(i)) 
        
        self.save_ckpt(model_savepath)
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
