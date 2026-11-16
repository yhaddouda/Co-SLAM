"""Attribute empty pose-block work while executing the actual CPU BA method.

The existing differentiable pose stand-in isolates selection, gradients, fresh
Adam state and commits. It deliberately makes no bitwise SE(3)-roundtrip claim.
"""
import random
import unittest

import numpy as np
import torch

import test_cohort_selection as cohort_tests


class AuditedModel(cohort_tests.CPUModel):
    def __init__(self, slam):
        super().__init__(slam)
        self.population_inputs = []
        self.selection_inputs = []
        self.loss_values = []

    def forward(self, origins, directions, color, depth, **kwargs):
        self.population_inputs.append(tuple(value.detach().clone() for value in
            (origins, directions, color, depth)))
        self.selection_inputs.append((self._ray_group_ids.clone(),
            self._cohort_active_ids.copy(), self._cohort_probabilities.copy()))
        result = super().forward(origins, directions, color, depth, **kwargs)
        self.loss_values.append(result['loss'].detach().clone())
        return result


class AuditedSlam(cohort_tests.CPUSlam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = AuditedModel(self)
        self.map_optimizer = torch.optim.Adam([self.model.parameter], lr=.01)
        self.pose_gradients = []

    def get_pose_param_optim(self, poses):
        rotation, translation, optimizer = super().get_pose_param_optim(poses)
        native_step = optimizer.step

        def audited_step(*args, **kwargs):
            self.pose_gradients.append((rotation.grad.detach().clone(),
                                        translation.grad.detach().clone()))
            return native_step(*args, **kwargs)

        optimizer.step = audited_step
        return rotation, translation, optimizer


class CohortPoseAttributionTests(unittest.TestCase):
    @staticmethod
    def batch():
        return dict(direction=torch.tensor([[[[.1, .2, 1.]]]]),
                    rgb=torch.full((1, 1, 1, 3), .5), depth=torch.ones(1, 1, 1))

    def paired_runs(self, n, count=8, optim_cur=True, accumulation=1):
        reduced = AuditedSlam(n=n, count=count, optim_cur=optim_cur)
        control = AuditedSlam(n=n, count=count, optim_cur=optim_cur)
        control.config['sample_pruning']['cohort_keep_empty_pose_blocks'] = True
        for slam in (reduced, control):
            slam.config['mapping'].update(iters=4, pose_accum_step=accumulation)
        before = {frame: pose.clone() for frame, pose in reduced.est_c2w_data.items()}
        for slam in (reduced, control):
            random.seed(29)
            slam.global_BA(self.batch(), n * 5)
        return reduced, control, before

    def assert_same_observation_problem(self, reduced, control):
        self.assertEqual(reduced.sample_calls, control.sample_calls)
        self.assertEqual(reduced.model.full_batch_sizes, control.model.full_batch_sizes)
        self.assertEqual(len(reduced.model.population_inputs),
                         len(control.model.population_inputs))
        for left, right in zip(reduced.model.population_inputs,
                               control.model.population_inputs):
            for a, b in zip(left, right):
                torch.testing.assert_close(a, b, rtol=0., atol=0.)
        for left, right in zip(reduced.model.selection_inputs,
                               control.model.selection_inputs):
            torch.testing.assert_close(left[0], right[0], rtol=0., atol=0.)
            for a, b in zip(left[1:], right[1:]):
                np.testing.assert_array_equal(a, b)
        for a, b in zip(reduced.model.loss_values, control.model.loss_values):
            torch.testing.assert_close(a, b, rtol=0., atol=0.)

    def test_empty_blocks_have_zero_gradients_and_active_adam_updates_agree(self):
        for n in (10, 23):
            for optim_cur in (False, True):
                for accumulation in (1, 2):
                    with self.subTest(history=n, optim_cur=optim_cur,
                                      accumulation=accumulation):
                        reduced, control, before = self.paired_runs(
                            n, optim_cur=optim_cur, accumulation=accumulation)
                        self.assert_same_observation_problem(reduced, control)
                        short = reduced._cohort_pose_diagnostics[-1]
                        full = control._cohort_pose_diagnostics[-1]
                        self.assertFalse(short['cohort_keep_empty_pose_blocks'])
                        self.assertTrue(full['cohort_keep_empty_pose_blocks'])
                        self.assertEqual(short['active_history_indices'],
                                         full['active_history_indices'])
                        self.assertEqual(full['optimized_history_indices'], list(range(1, n)))
                        self.assertLess(short['optimized_pose_count'], full['optimized_pose_count'])
                        active = short['optimized_history_indices']
                        inactive = sorted(set(range(1, n)) - set(active))
                        self.assertTrue(inactive)
                        full_rows = [index - 1 for index in active]
                        if optim_cur:
                            full_rows.append(n - 1)
                        inactive_rows = [index - 1 for index in inactive]
                        self.assertEqual(len(reduced.pose_gradients), 4 // accumulation)
                        self.assertEqual(len(reduced.pose_gradients), len(control.pose_gradients))
                        for short_grad, full_grad in zip(reduced.pose_gradients,
                                                         control.pose_gradients):
                            for a, b in zip(short_grad, full_grad):
                                self.assertEqual(torch.count_nonzero(b[inactive_rows]).item(), 0)
                                torch.testing.assert_close(a, b[full_rows], rtol=0., atol=0.)
                        for frame in before:
                            torch.testing.assert_close(reduced.est_c2w_data[frame],
                                control.est_c2w_data[frame], rtol=0., atol=0.)
                        for index in [0, *inactive]:
                            torch.testing.assert_close(control.est_c2w_data[index * 5],
                                                       before[index * 5], rtol=0., atol=0.)
                        for index in active:
                            self.assertFalse(torch.equal(control.est_c2w_data[index * 5],
                                                         before[index * 5]))
                        if not optim_cur:
                            torch.testing.assert_close(control.est_c2w_data[n * 5],
                                                       before[n * 5], rtol=0., atol=0.)
                        torch.testing.assert_close(reduced.model.parameter,
                                                   control.model.parameter, rtol=0., atol=0.)

    def test_early_ba_rule_and_full_cohort_are_unchanged(self):
        for n, count in ((1, 8), (4, 1)):
            for optim_cur in (False, True):
                with self.subTest(history=n, cohorts=count, optim_cur=optim_cur):
                    reduced, control, before = self.paired_runs(n, count, optim_cur)
                    self.assert_same_observation_problem(reduced, control)
                    self.assertEqual(reduced.pose_parameter_counts, control.pose_parameter_counts)
                    for frame in before:
                        torch.testing.assert_close(reduced.est_c2w_data[frame],
                                                   control.est_c2w_data[frame], rtol=0., atol=0.)
                    if n == 1:
                        self.assertEqual(control.pose_parameter_counts, [])
                        self.assertFalse(control._cohort_pose_diagnostics[-1]['optim_cur'])
                        for frame in before:
                            torch.testing.assert_close(control.est_c2w_data[frame], before[frame],
                                                       rtol=0., atol=0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
