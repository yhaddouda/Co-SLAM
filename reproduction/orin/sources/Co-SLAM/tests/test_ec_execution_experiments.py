"""Execution invariants in the isolated EC-SLAM workspace."""
import importlib.util
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'experiments/ec_slam_execution/src/execution_experiments.py'
SPEC = importlib.util.spec_from_file_location('ec_execution_tests', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ECExecutionTests(unittest.TestCase):
    def test_deferred_selection_preserves_ties_ceiling_nan_and_last(self):
        for values, expected in (([3., 1., 1.], 1), ([float('nan'), 2., 1.], 2),
                                 ([1e11, 1e9, 2.], 2), ([1., float('nan'), 2.], 0)):
            with self.subTest(values=values):
                selection = MODULE.DeferredCameraSelection(len(values), torch.zeros(7), keep_matrix=True)
                for i, value in enumerate(values):
                    selection.add(torch.tensor(value), torch.full((7,), float(i)), torch.full((3, 4), float(i)))
                camera, matrix = selection.best()
                torch.testing.assert_close(camera, torch.full((7,), float(expected)))
                torch.testing.assert_close(matrix, torch.full((3, 4), float(expected)))
                camera, matrix = selection.best(last_best=True)
                torch.testing.assert_close(camera, torch.full((7,), float(len(values) - 1)))
        selection = MODULE.DeferredCameraSelection(1, torch.zeros(7))
        selection.add(torch.tensor(1e11), torch.zeros(7), torch.zeros(3, 4))
        with self.assertRaises(RuntimeError):
            selection.best()

    def test_mapping_cache_preserves_pose_values_and_gradients(self):
        torch.manual_seed(9)
        reference = torch.randn(20, 4, 4)
        ids = [3, 5, 12]
        optimized = torch.randn(3, 4, 4, requires_grad=True)
        baseline = reference.clone()
        baseline[ids] = optimized
        fast = reference.index_copy(0, torch.tensor(ids), optimized)
        weights = torch.randn_like(reference)
        torch.testing.assert_close(baseline, fast, rtol=0, atol=0)
        a, = torch.autograd.grad((baseline * weights).sum(), (optimized,))
        b, = torch.autograd.grad((fast * weights).sum(), (optimized,))
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_verbose_tracking_retains_existing_diagnostics(self):
        config = dict(tracking_experiment=dict(variant='baseline_fast'), verbose=True,
                      debug=dict(flag=False, showTrackerLoss=False))
        self.assertFalse(MODULE.tracking_flags(config)['defer_best'])
        config['verbose'] = False
        self.assertTrue(MODULE.tracking_flags(config)['defer_best'])


if __name__ == '__main__':
    unittest.main()
