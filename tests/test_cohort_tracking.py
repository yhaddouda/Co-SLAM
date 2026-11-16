"""All-ray support tracking attached to the pose-cohort BA policy."""
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
import pruning_cpu_fixtures as ray_tests
from optimization.sample_pruning import pruning_policy, validate_pruning


class CohortTrackingTests(unittest.TestCase):
    def setUp(self):
        nvtx = patch('torch.cuda.nvtx.range', lambda *a, **k: nullcontext())
        nvtx.start()
        self.addCleanup(nvtx.stop)
        self.helper = ray_tests.RayFieldFixture()

    def model(self, mode='oscillating'):
        model = self.helper.model(mode=mode)
        model.config['sample_pruning']['policy'] = 'ray_cohort'
        model.config['tracking']['iter'] = 10
        for section in ('ba_bucket', 'tracking_bucket', 'tracking_naive_pruning',
                        'tracking_depth_spatial'):
            model.config.setdefault(section, {})['enabled'] = False
        model.config.setdefault('ba_experiment', {})['variant'] = 'baseline'
        return model

    def test_default_and_periodic_dense_anchors_are_tracking_only(self):
        model = self.model()
        for every, expected in (
                (None, [None] + ['ray_cohort'] * 8 + [None]),
                (3, [None, 'ray_cohort', 'ray_cohort', None, 'ray_cohort',
                     'ray_cohort', None, 'ray_cohort', 'ray_cohort', None])):
            if every is not None:
                model.config['sample_pruning']['tracking_anchor_every'] = every
            actual = []
            for iteration in range(10):
                model._pruning_iteration = iteration
                actual.append(pruning_policy(model))
                self.assertEqual(model._pruning_iteration,
                                 iteration + int(actual[-1] is None))
            self.assertEqual(actual, expected)
        model._pruning_stage = 'Bundle_Adjustment'
        for iteration in range(model.config['mapping']['iters']):
            model._pruning_iteration = iteration
            self.assertEqual(pruning_policy(model), 'ray_cohort')
            self.assertEqual(model._pruning_iteration, iteration)
        model._pruning_stage = 'Tracking'
        model.config['sample_pruning'].update(cohort_tracking='ray_cycle', tracking_anchor_every=0)
        for iteration in range(10):
            model._pruning_iteration = iteration
            self.assertEqual(pruning_policy(model), 'ray_cohort')
            self.assertEqual(model._pruning_iteration, iteration)

    def test_support_forward_keeps_geometry_gradients_jitter_and_every_ray(self):
        for mode in ('oscillating', 'late_surface', 'no_crossing'):
            with self.subTest(mode=mode):
                model = self.model(mode)
                args = self.helper.inputs((1.2, 2.3, 0., 3.2))
                dense = self.helper.dense(model, args)
                dense_points = model.queried_points[-1].reshape(4, -1, 3)
                native_rng = torch.get_rng_state()
                model._pruning_iteration = 1
                torch.manual_seed(197)
                sparse = model.forward(*args)
                torch.testing.assert_close(torch.get_rng_state(), native_rng)
                self.assertEqual(model._pruning_iteration, 2)
                keep, indices = model._support_selection[1:]
                self.assertTrue(keep[2].all())  # Missing depth stays dense.
                self.assertTrue(keep.any(dim=1).all())
                torch.testing.assert_close(model.queried_points[-1],
                    dense_points.reshape(-1, 3)[indices], rtol=0, atol=0)
                self.assertLess(indices.numel(), keep.numel())
                parameters = (model.coefficients, args[0], args[1])
                for key in ('fs_loss', 'sdf_loss'):
                    torch.testing.assert_close(sparse[key], dense[key],
                                               rtol=3e-5, atol=2e-6)
                    actual = torch.autograd.grad(sparse[key], parameters,
                                                 retain_graph=True)
                    expected = torch.autograd.grad(dense[key], parameters,
                                                   retain_graph=True)
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, rtol=5e-5, atol=2e-5)
                row = model.pruning_diagnostics[-1]
                self.assertEqual(set(row), {'stage', 'frame', 'iteration',
                    'dense_samples', 'queried_samples', 'recovery_samples',
                    'recovered_rays', 'rays', 'selected_rays', 'weight_sum',
                    'cohort_old_rays', 'anchor_rays',
                    'cohort_transition_rays', 'cohort_mature_rays'})
                self.assertEqual(row['iteration'], 1)
                self.assertEqual(row['queried_samples'], indices.numel())
                for key in ('rays', 'selected_rays', 'weight_sum', 'anchor_rays'):
                    self.assertEqual(row[key], 4)
                self.assertEqual(row['cohort_old_rays'], 0)

    def test_dense_anchor_restores_native_values_gradients_and_rng(self):
        for iteration in (0, 9):
            model = self.model('late_surface')
            args = self.helper.inputs()
            dense = self.helper.dense(model, args)
            state = torch.get_rng_state()
            model._pruning_iteration = iteration
            torch.manual_seed(197)
            actual = model.forward(*args)
            torch.testing.assert_close(torch.get_rng_state(), state)
            self.assertEqual(model._pruning_iteration, iteration + 1)
            for key in (*ray_tests.LOSSES, 'rgb', 'depth'):
                torch.testing.assert_close(actual[key], dense[key], rtol=0, atol=0)
            parameters = (model.coefficients, args[0], args[1])
            a = torch.autograd.grad(sum(actual[k] for k in ray_tests.LOSSES), parameters)
            b = torch.autograd.grad(sum(dense[k] for k in ray_tests.LOSSES), parameters)
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertFalse(getattr(model, 'pruning_diagnostics', []))

    def test_configuration_guards(self):
        for name, value in (('cohort_tracking', 'unknown'),
                            ('tracking_anchor_every', -1),
                            ('tracking_anchor_every', True),
                            ('tracking_anchor_every', 1.5)):
            model = self.model()
            model.config['sample_pruning'][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_pruning(model.config)
        validate_pruning(self.model().config)


if __name__ == '__main__':
    unittest.main()
