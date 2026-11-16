"""Semantic checks for execution changes that can affect tracking trajectories."""
import ast
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from optimization.tracking_experiments import (
    DeferredBestPose, DevicePoseTransform, DenseDepthLattice, tracking_execution_flags,
)

ROOT = Path(__file__).resolve().parents[1]


def original_axis_angle_transform():
    tree = ast.parse((ROOT / 'optimization/utils.py').read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in {'axis_angle_to_matrix', 'at_to_transform_matrix'}]
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<original poses>', 'exec'), namespace)
    return namespace['at_to_transform_matrix']


class TrackingExecutionTests(unittest.TestCase):
    def test_cached_dense_renderer_preserves_samples_losses_gradients_and_rng(self):
        from test_ba_experiments import AnalyticField
        results, rng_states = [], []
        with patch('torch.cuda.nvtx.range', lambda *args, **kwargs: nullcontext()):
            for use_cache in (False, True):
                field = AnalyticField()
                field.config['training']['perturb'] = 1.
                cache = DenseDepthLattice(field.config) if use_cache else None
                target = torch.tensor([[0.], [0.7], [2.1]])
                rgb = torch.full((3, 3), .5)
                torch.manual_seed(73)
                iterations = []
                for index in range(3):
                    origin = torch.full((3, 3), .01 * index, requires_grad=True)
                    direction = torch.tensor([[.1, .2, 1.]]).repeat(3, 1).requires_grad_(True)
                    ret = field.forward(origin, direction, rgb, target, depth_sampling_cache=cache)
                    loss = sum(ret[name] for name in ('rgb_loss','depth_loss','sdf_loss','fs_loss'))
                    gradients = torch.autograd.grad(loss, (origin, direction, field.surface))
                    iterations.append((ret, gradients))
                results.append(iterations)
                rng_states.append(torch.get_rng_state())
        for baseline, cached in zip(*results):
            for name in baseline[0]:
                torch.testing.assert_close(baseline[0][name], cached[0][name], rtol=0, atol=0)
            for expected, actual in zip(baseline[1], cached[1]):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        torch.testing.assert_close(rng_states[0], rng_states[1], rtol=0, atol=0)

    def test_depth_cache_requires_one_fixed_dense_batch(self):
        from test_ba_experiments import AnalyticField
        config = AnalyticField().config
        rays, target = torch.zeros(3, 3), torch.ones(3, 1)
        cache = DenseDepthLattice(config)
        cache.prepare(rays, target)
        with self.assertRaises(ValueError):
            cache.prepare(rays, target.clone())
        with self.assertRaises(ValueError):
            DenseDepthLattice(config).prepare(rays, target.requires_grad_())
        config['tracking_experiment'] = dict(variant='cached_fast')
        config['tracking_depth_spatial']['enabled'] = False
        config['tracking_bucket']['enabled'] = False
        config['tracking_naive_pruning']['enabled'] = False
        config['training']['uniform_samples_until_depth'] = False
        self.assertTrue(tracking_execution_flags(config)['cache_depth'])
        config['training']['uniform_samples_until_depth'] = True
        self.assertFalse(tracking_execution_flags(config)['cache_depth'])

    def test_pose_values_and_rotation_translation_gradients(self):
        torch.manual_seed(17)
        rotation = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        translation = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(5, 4, 4, dtype=torch.float64)
        original = original_axis_angle_transform()(rotation, translation)
        fast = DevicePoseTransform(rotation)(rotation, translation)
        torch.testing.assert_close(original, fast, rtol=0, atol=0)
        reference_gradient = torch.autograd.grad((original * weight).sum(), (rotation, translation))
        fast_gradient = torch.autograd.grad((fast * weight).sum(), (rotation, translation))
        for expected, actual in zip(reference_gradient, fast_gradient):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_best_selection_matches_strict_python_comparisons(self):
        for values in ([3., 1., 1., 2.], [float('nan'), 1., 0.],
                       [3., float('nan'), 2.], [float('inf'), float('inf')],
                       [3., float('-inf'), float('-inf')]):
            with self.subTest(values=values):
                reference = torch.zeros(1, 4, 4, requires_grad=True)
                selection = DeferredBestPose(len(values), reference)
                best_index, minimum = 0, values[0]
                for index, value in enumerate(values):
                    pose = torch.full_like(reference, float(index), requires_grad=True)
                    selection.add(torch.tensor(value), pose)
                    # Changing source storage must not mutate an earlier candidate.
                    with torch.no_grad():
                        pose.add_(100)
                    if value < minimum:
                        best_index, minimum = index, value
                chosen = selection.best()
                torch.testing.assert_close(chosen, torch.full_like(chosen, float(best_index)))
                self.assertFalse(chosen.requires_grad)

    def test_early_stopping_uses_existing_host_decision_path(self):
        config = {'tracking': {'iter': 10, 'wait_iters': 9},
                  'tracking_experiment': {'variant': 'baseline_fast'}}
        self.assertFalse(tracking_execution_flags(config)['defer_best'])
        config['tracking']['wait_iters'] = 10
        self.assertTrue(tracking_execution_flags(config)['defer_best'])
        config['tracking_experiment']['variant'] = 'baseline'
        self.assertFalse(any(tracking_execution_flags(config).values()))

    def test_ba_body_preserves_tested_variant_defaults(self):
        path = ROOT / 'results/ba_supervisor_20261008/suite/wall/office1/baseline_fast_seed0_repeat0/source/coslam.py'
        if not path.exists():
            self.skipTest('Saved BA source snapshot unavailable')
        def body(source):
            tree = ast.parse(source)
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CoSLAM')
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'global_BA')
            # BA now exposes independent switches. Substitute their historical
            # variant defaults, then compare every remaining operation/schedule.
            class VariantDefaults(ast.NodeTransformer):
                def visit_Assign(self, statement):
                    if any(isinstance(target, ast.Name) and target.id == 'ba_execution'
                           for target in statement.targets):
                        return None
                    return self.generic_visit(statement)

                def visit_Subscript(self, expression):
                    if isinstance(expression.value, ast.Name) and expression.value.id == 'ba_execution':
                        variants = {
                            'preload_rays': "ba_variant in {'lattice_guarded_preload', 'baseline_fast', 'lattice_guarded_fast'}",
                            'fast_pose_commit': "ba_variant in {'baseline_fast', 'lattice_guarded_fast'}",
                        }
                        return ast.parse(variants[expression.slice.value], mode='eval').body
                    return self.generic_visit(expression)
            node = VariantDefaults().visit(node)
            return ast.dump(node, include_attributes=False)
        self.assertEqual(body(path.read_text()), body((ROOT / 'coslam.py').read_text()))


if __name__ == '__main__':
    unittest.main()
