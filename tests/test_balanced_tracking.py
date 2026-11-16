"""Spatial tracking must retain distant and missing-depth image regions."""
import ast
from contextlib import nullcontext
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'coslam.py').read_text())
scene = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CoSLAM')
node = next(node for node in scene.body if isinstance(node, ast.FunctionDef)
            and node.name == 'prepare_tracking_depth_spatial_batch')
namespace = dict(torch=torch, np=np, math=math, nvtx_range=lambda *args: nullcontext())
exec(compile(ast.Module(body=[node], type_ignores=[]), 'coslam.py', 'exec'), namespace)
prepare = namespace['prepare_tracking_depth_spatial_batch']


class BalancedTrackingTests(unittest.TestCase):
    def prepare(self, mode, missing=False):
        height, width = 64, 64
        rows, columns = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        depth = torch.where((rows + columns) % 2 == 0, 1., 4.)
        if missing:
            depth[:, :8] = 0.
            depth[:8, :8] = float('nan')
        batch = dict(depth=depth[None], rgb=torch.ones(1, height, width, 3),
                     direction=torch.stack([rows, columns, torch.ones_like(rows)], -1).float()[None])
        config = dict(tracking=dict(sample=128), training=dict(n_samples_d=64),
                      tracking_depth_spatial=dict(mode=mode, pool_factor=4, grid_h=8, grid_w=8, min_coverage=.75))
        generator = np.random.default_rng(19)
        slam = SimpleNamespace(config=config, dataset=SimpleNamespace(H=height, W=width),
                               device='cpu', tracking_depth_spatial_rng=lambda: generator)
        def counts(target, valid_depth=None):
            values = torch.searchsorted(torch.linspace(0., 5., 64), target.squeeze(-1).contiguous(), right=True)
            return torch.where(valid_depth, values.clamp(1, 64), 64)
        slam.compute_uniform_until_depth_counts = counts
        torch.manual_seed(17)
        return prepare(slam, batch, 0, 0, 1)

    def test_spatial_batch_preserves_far_constraints_and_dense_sample_count(self):
        balanced, pruned = self.prepare('balanced_dense'), self.prepare('pruned')
        self.assertEqual(balanced['uniform_count'], 64)
        far_fraction = float((balanced['target_d'] == 4.).float().mean())
        self.assertGreater(far_fraction, .35)
        self.assertLess(far_fraction, .65)
        self.assertEqual(int((pruned['target_d'] == 4.).sum()), 0)
        coordinates = balanced['rays_d_cam'][:, :2].long()
        self.assertEqual(torch.unique(coordinates, dim=0).shape[0], 128)
        cells = (coordinates[:, 0] // 8) * 8 + coordinates[:, 1] // 8
        self.assertGreaterEqual(torch.unique(cells).numel(), 48)

    def test_missing_depth_regions_are_retained_and_nonfinite_depth_is_sanitized(self):
        batch = self.prepare('balanced_dense', missing=True)
        self.assertTrue(torch.isfinite(batch['target_d']).all())
        self.assertGreater(int((batch['target_d'] == 0).sum()), 0)
        self.assertEqual(batch['uniform_count'], 64)


if __name__ == '__main__':
    unittest.main()
