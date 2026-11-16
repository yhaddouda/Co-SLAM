"""Full-observation pose selection and exact complementary-half budgets."""
import copy
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch
import pruning_cpu_fixtures as pose_tests
import pruning_cpu_fixtures as ray_tests
from optimization.sample_pruning import validate_pruning


class DenseAnchorTrackingTests(unittest.TestCase):
    def test_actual_best_uses_only_dense_checkpoints_without_skipping_steps(self):
        for best in (False, True):
            for representation in ('axis_angle', 'quat'):
                with self.subTest(best=best, representation=representation):
                    native = pose_tests.TrackingSlam(representation, best)
                    anchors = pose_tests.TrackingSlam(representation, best)
                    # With six iterations and stride three, full checkpoints
                    # are 0, 3, 5. Low half-batch scores at 1,2,4 must not win.
                    native.losses = anchors.losses = [4., -5., -10., 3., -20., 5.]
                    anchors.config['sample_pruning'].update(
                        tracking_pose_selection='dense_anchors', tracking_anchor_every=3)
                    batch = pose_tests.tracking_batch()
                    native.tracking_render(batch, 7)
                    anchors.tracking_render(batch, 7)
                    self.assertEqual(anchors.optimizer.steps, native.optimizer.steps)
                    self.assertEqual(anchors.optimizer.steps, 6)
                    self.assertEqual(anchors.model.ray_counts, [4] * 6)
                    torch.testing.assert_close(torch.stack(anchors.model.evaluated),
                                               torch.stack(native.model.evaluated), rtol=0, atol=0)
                    expected = anchors.model.evaluated[3 if best else 5]
                    torch.testing.assert_close(anchors.est_c2w_data[7], expected, rtol=0, atol=0)
                    if not best:
                        torch.testing.assert_close(anchors.est_c2w_data[7], native.est_c2w_data[7], rtol=0, atol=0)
                    torch.testing.assert_close(anchors.est_c2w_data_rel[7],
                        expected @ anchors.est_c2w_data[5].inverse(), rtol=0, atol=0)

    def test_only_eligible_checkpoints_advance_existing_early_stop_counter(self):
        slam = pose_tests.TrackingSlam()
        slam.losses = [1., 10., 20., 2., 30., 40.]
        slam.config['tracking']['wait_iters'] = 1
        slam.config['sample_pruning'].update(
            tracking_pose_selection='dense_anchors', tracking_anchor_every=3)
        slam.tracking_render(pose_tests.tracking_batch(), 7)
        self.assertEqual(len(slam.model.evaluated), 4)
        self.assertEqual(slam.optimizer.steps, 3)
        torch.testing.assert_close(slam.est_c2w_data[7], slam.model.evaluated[0])

    def test_q2_anchor3_actual_forward_queries_every_ray_seven_times(self):
        with patch('torch.cuda.nvtx.range', lambda *a, **k: nullcontext()):
            for population in (8, 9):
                helper = ray_tests.RayFieldFixture()
                model = helper.model()
                model.config['tracking']['iter'] = 10
                model.config['sample_pruning'].update(policy='ray_cohort',
                    cohort_tracking='ray_cycle', tracking_cycle_groups=2,
                    tracking_anchor_every=3, tracking_pose_selection='dense_anchors')
                args = helper.inputs(tuple(torch.linspace(.6, 3.4, population).tolist()))
                model._ray_sampling_metadata = torch.cat((args[1].detach(), args[2], args[3]), dim=-1)
                appearances = torch.zeros(population, dtype=torch.int64)
                dense_iterations = []
                total_queries = 0
                for iteration in range(10):
                    self.assertEqual(model._pruning_iteration, iteration)
                    before = len(getattr(model, 'pruning_diagnostics', []))
                    model.forward(*args)
                    queried = len(model.queried_points[-1])
                    total_queries += queried
                    if len(getattr(model, 'pruning_diagnostics', [])) == before:
                        dense_iterations.append(iteration)
                        selected = torch.arange(population)
                    else:
                        selected = model._ray_cycle_selection[1][iteration % 2]
                    appearances[selected] += 1
                    points_per_ray = model.config['training']['n_samples_d'] + model.config['training']['n_range_d']
                    self.assertEqual(queried, len(selected) * points_per_ray)
                self.assertEqual(dense_iterations, [0, 3, 6, 9])
                torch.testing.assert_close(appearances, torch.full_like(appearances, 7))
                self.assertEqual(total_queries, population * points_per_ray * 7)
                self.assertEqual(model._pruning_iteration, 10)

    def test_configuration_guards_and_native_final_pose_compatibility(self):
        base = dict(training={}, tracking=dict(iter=10, best=True),
                    sample_pruning=dict(enabled=False, policy='dense',
                        tracking_pose_selection='dense_anchors', tracking_anchor_every=3))
        validate_pruning(base)
        last_pose = copy.deepcopy(base)
        last_pose['tracking']['best'] = False
        validate_pruning(last_pose)
        for name, value in (('tracking_pose_selection', 'unknown'),
                            ('tracking_pose_average', 4),
                            ('tracking_anchor_every', 0),
                            ('tracking_anchor_every', -1),
                            ('tracking_anchor_every', True),
                            ('tracking_anchor_every', 1.5),
                            ('policy', 'ray_subset')):
            bad = copy.deepcopy(base)
            bad['sample_pruning'][name] = value
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_pruning(bad)
        cycle = copy.deepcopy(base)
        cycle['sample_pruning'].update(enabled=True, policy='ray_cohort', cohort_tracking='ray_cycle')
        validate_pruning(cycle)
        cycle['sample_pruning']['cohort_tracking'] = 'support'
        with self.assertRaises(ValueError):
            validate_pruning(cycle)


if __name__ == '__main__':
    unittest.main()
