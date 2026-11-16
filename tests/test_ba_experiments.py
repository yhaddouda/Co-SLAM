"""Dense objective/gradient equivalence and rendering fallback for BA packing.

Read the actual renderer methods without importing CUDA-only encoders, allowing
these mathematical checks to run in the coslam environment on CPU as well.
"""
import ast
import unittest
import random
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch
import yaml
import config

from model.utils import compute_loss, get_sdf_loss, masked_mse_loss, mse2psnr
from optimization.ba_experiments import (
    forward_ba_lattice, render_ba_lattice, prepare_mixed_ba_batches, commit_ba_poses_fast,
    ba_bucket_mode,
)
from model.keyframe import KeyFrameDatabase

ROOT = Path(__file__).resolve().parents[1]


class Timer:
    def stage(self, *args):
        return nullcontext()


class AnalyticField:
    def __init__(self, surface=2.03, constant=False):
        self.config = yaml.safe_load((ROOT / 'configs/Replica/replica.yaml').read_text())
        self.config['ba_experiment']['collect_samples'] = True
        self.config['training']['perturb'] = 0
        self.surface = torch.tensor(surface, requires_grad=True)
        self.constant = constant
        self.layer_timing = Timer()
        self.collect_uniform_sampling_stats = False
        self.debug_naive_pruning = False
        self.training = True

    def run_network(self, points, **kwargs):
        sdf = (self.surface - points[..., 2]) / .1
        if self.constant:
            sdf = torch.ones_like(sdf) * self.surface
        rgb = points * .2 + self.surface * .1
        return torch.cat([rgb, sdf[..., None]], -1)


source = ast.parse((ROOT / 'model/scene_rep.py').read_text())
scene_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'JointEncoding')
for name in ('sdf2weights', 'raw2outputs', 'render_rays', 'forward'):
    method = next(n for n in scene_class.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(torch=torch, compute_loss=compute_loss, get_sdf_loss=get_sdf_loss,
                     masked_mse_loss=masked_mse_loss, mse2psnr=mse2psnr)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / 'model/scene_rep.py'), 'exec'), namespace)
    setattr(AnalyticField, name, namespace[name])


class BAExperimentTests(unittest.TestCase):
    def test_production_bucket_policy_is_explicit_and_validated(self):
        cfg = dict(training=dict(n_importance=0), ba_bucket=dict(enabled=True))
        self.assertEqual(ba_bucket_mode(cfg), 'legacy')
        cfg['ba_bucket']['mode'] = 'guarded'
        self.assertEqual(ba_bucket_mode(cfg), 'guarded')
        for margin in (-1, float('nan'), float('inf')):
            cfg['ba_bucket']['depth_margin'] = margin
            with self.assertRaises(ValueError):
                ba_bucket_mode(cfg)
        cfg['ba_bucket']['depth_margin'] = .1
        cfg['training']['n_importance'] = 16
        with self.assertRaises(ValueError):
            ba_bucket_mode(cfg)
        cfg['training']['n_importance'] = 0
        cfg['ba_experiment'] = dict(variant='bucket_dense')
        with self.assertRaises(ValueError):
            ba_bucket_mode(cfg)
        cfg['ba_experiment']['variant'] = 'baseline'
        cfg['ba_bucket']['mode'] = 'typo'
        with self.assertRaises(ValueError):
            ba_bucket_mode(cfg)
        cfg['ba_bucket']['enabled'] = False
        self.assertEqual(ba_bucket_mode(cfg), 'off')

    def setUp(self):
        self.nvtx = patch('torch.cuda.nvtx.range', lambda *args, **kwargs: nullcontext())
        self.nvtx.start()
        self.addCleanup(self.nvtx.stop)

    def compare_dense(self, surface=2.03, depths=(2., 2.5), perturb=0, constant=False,
                      scene_config=None):
        model = AnalyticField(surface, constant)
        if scene_config is not None:
            model.config = config.load_config(str(ROOT / scene_config))
            model.config['ba_experiment'] = dict(collect_samples=True)
        model.config['training']['perturb'] = perturb
        origins = torch.zeros(len(depths), 3, requires_grad=True)
        directions = torch.tensor([[.1, .2, 1.]]).repeat(len(depths), 1)
        target = torch.tensor(depths)[:, None]
        rgb = torch.full((len(depths), 3), .5)
        torch.manual_seed(52)
        dense = model.forward(origins, directions, rgb, target)
        torch.manual_seed(52)
        packed = forward_ba_lattice(model, origins, directions, rgb, target, guarded=True)
        for key in ('rgb', 'depth', 'rgb_loss', 'depth_loss', 'sdf_loss', 'fs_loss'):
            torch.testing.assert_close(packed[key], dense[key])
        weights = dict(rgb_loss=5., depth_loss=.1, sdf_loss=1000., fs_loss=10.)
        dense_loss = sum(weights[key] * dense[key] for key in weights)
        packed_loss = sum(weights[key] * packed[key] for key in weights)
        for actual, expected in zip(torch.autograd.grad(packed_loss, (model.surface, origins)),
                                    torch.autograd.grad(dense_loss, (model.surface, origins))):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
        return model.ba_experiment_sample_counts[-1]

    def test_guarded_matches_dense_losses_and_map_pose_gradients(self):
        count = self.compare_dense()
        self.assertLess(count['queried_samples'], count['dense_samples'])

    def test_baseline_perturbations_and_gradients_are_preserved(self):
        self.compare_dense(perturb=1)

    def test_surface_beyond_measured_band_triggers_fallback(self):
        count = self.compare_dense(surface=2.4, depths=(2., 2.))
        self.assertGreater(count['fallback_samples'], 0)
        self.assertEqual(count['queried_samples'], count['dense_samples'])

    def test_partial_render_support_triggers_fallback(self):
        count = self.compare_dense(surface=2.17, depths=(2.08, 2.08))
        self.assertGreater(count['fallback_samples'], 0)

    def test_no_crossing_is_checked_densely(self):
        count = self.compare_dense(constant=True)
        self.assertEqual(count['queried_samples'], count['dense_samples'])

    def test_missing_depth_retains_dense_sampling(self):
        count = self.compare_dense(depths=(0., 0.))
        self.assertEqual(count['queried_samples'], count['dense_samples'])

    def test_fractional_missing_rgb_weight_matches_configuration(self):
        model = AnalyticField()
        model.config['training']['rgb_missing'] = .05
        origin = torch.zeros(2, 3)
        direction = torch.tensor([[0., 0., 1.]]).repeat(2, 1)
        target = torch.tensor([[0.], [2.]])
        color = torch.ones(2, 3)
        ret = model.forward(origin, direction, color, target)
        weights = torch.tensor([[.05], [1.]])
        expected = compute_loss(ret['rgb'] * weights, color * weights)
        torch.testing.assert_close(ret['rgb_loss'], expected)

    def test_guarded_objective_on_all_four_dataset_sampling_configs(self):
        for path in ('Tum/fr1_desk.yaml', 'ScanNet/scene0000.yaml',
                     'Synthetic/tg.yaml', 'Replica/office0.yaml'):
            for depths in ((0., 1.9, 2., 2.5), (0., 0.)):
                with self.subTest(config=path, depths=depths):
                    self.compare_dense(depths=depths, perturb=1,
                                       scene_config='configs/' + path)

    def test_single_ray_and_depth_beyond_far(self):
        self.compare_dense(depths=(2.,))
        count = self.compare_dense(depths=(6.,))
        self.assertEqual(count['queried_samples'], count['dense_samples'])

    def test_unconditional_lattice_is_separate_approximate_variant(self):
        model = AnalyticField(surface=2.4)
        origin = torch.zeros(2, 3)
        direction = torch.tensor([[0., 0., 1.]]).repeat(2, 1)
        target = torch.full((2, 1), 2.)
        ret = render_ba_lattice(model, origin, direction, target, guarded=False)
        self.assertTrue(torch.isfinite(ret['depth']).all())
        count = model.ba_experiment_sample_counts[-1]
        self.assertEqual(count['fallback_samples'], 0)
        self.assertLess(count['queried_samples'], count['dense_samples'])

    def test_preload_preserves_baseline_ray_draws_ids_and_rng_state(self):
        cfg = dict(mapping=dict(sample=32, min_pixels_cur=8, iters=10, keyframe_every=5))
        database = KeyFrameDatabase(cfg, 20, 20, 3, 100, 'cpu')
        database.frame_ids = torch.tensor([0, 5, 10])
        database.rays = torch.arange(3 * 100 * 7).float().reshape(3, 100, 7)
        current = torch.arange(20 * 20 * 7).float().reshape(400, 7) * -1
        slam = SimpleNamespace(config=cfg, keyframeDatabase=database,
                               dataset=SimpleNamespace(H=20, W=20), device='cpu')
        random.seed(71)
        expected_rays, expected_ids = [], []
        for _ in range(10):
            rays, ids = database.sample_global_rays(32)
            indices = random.sample(range(400), max(32 // 3, 8))
            expected_rays.append(torch.cat([rays, current[indices]], 0))
            expected_ids.append(torch.cat([ids // 5, -torch.ones(len(indices))]).long())
        expected_state = random.getstate()
        random.seed(71)
        torch_state = torch.get_rng_state()
        actual = prepare_mixed_ba_batches(slam, current)
        self.assertEqual(random.getstate(), expected_state)
        torch.testing.assert_close(torch.get_rng_state(), torch_state)
        torch.testing.assert_close(torch.cat([actual['rays_d_cam'], actual['target_s'], actual['target_d']], -1),
                                   torch.stack(expected_rays))
        torch.testing.assert_close(actual['ids_all'], torch.stack(expected_ids))

    def test_batched_pose_commit_matches_original_and_detaches_storage(self):
        pose_source = ast.parse((ROOT / 'optimization/utils.py').read_text())
        namespace = dict(torch=torch)
        for name in ('axis_angle_to_matrix', 'at_to_transform_matrix'):
            node = next(n for n in pose_source.body if isinstance(n, ast.FunctionDef) and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / 'optimization/utils.py'), 'exec'), namespace)
        transform = namespace['at_to_transform_matrix']
        for optimize_current in (False, True):
            count = 3 if optimize_current else 2
            rotations = (torch.arange(count * 3).reshape(count, 3).float() + 1) * .1
            rotations.requires_grad_(True)
            translations = torch.arange(count * 3).reshape(count, 3).float().requires_grad_(True)
            batch = transform(rotations, translations)
            controller = SimpleNamespace(config=dict(mapping=dict(optim_cur=optimize_current)),
                                         est_c2w_data={0: torch.eye(4), 15: torch.eye(4)})
            commit_ba_poses_fast(controller, batch, torch.tensor([0, 5, 10]), 15)
            for index, frame in enumerate((5, 10)):
                expected = transform(rotations[index:index+1], translations[index:index+1]).detach()[0]
                torch.testing.assert_close(controller.est_c2w_data[frame], expected)
                self.assertFalse(controller.est_c2w_data[frame].requires_grad)
            if optimize_current:
                torch.testing.assert_close(controller.est_c2w_data[15], transform(rotations[-1:], translations[-1:]).detach()[0])
            else:
                torch.testing.assert_close(controller.est_c2w_data[15], torch.eye(4))
            torch.testing.assert_close(controller.est_c2w_data[0], torch.eye(4))
            before = controller.est_c2w_data[5].clone()
            with torch.no_grad():
                batch.add_(1)
            torch.testing.assert_close(controller.est_c2w_data[5], before)


if __name__ == '__main__':
    unittest.main()
