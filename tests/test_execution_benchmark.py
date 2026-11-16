"""Validate ATE conventions against the repository's existing evaluator."""
import unittest
from pathlib import Path

import numpy as np

from scripts.Validation.execution_benchmark_common import rigid_ate
from tools.eval_ate import align

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def test_ate_matches_existing_horn_evaluator(self):
        path = ROOT / 'results/ba_experiments/full/office1/baseline_seed0/trajectories.npz'
        if path.exists():
            saved = np.load(path)
            truth, estimated = saved['ground_truth'], saved['estimated']
        else:
            rng = np.random.default_rng(4)
            truth = np.tile(np.eye(4), (30, 1, 1))
            truth[:, :3, 3] = rng.normal(size=(30, 3))
            estimated = truth.copy()
            estimated[:, :3, 3] += rng.normal(size=(30, 3)) * .01
        _, _, errors = align(np.matrix(estimated[:, :3, 3].astype(np.float64).T),
                             np.matrix(truth[:, :3, 3].astype(np.float64).T))
        expected = float(np.sqrt(np.mean(errors ** 2)))
        self.assertAlmostEqual(rigid_ate(truth, estimated), expected, places=15)


if __name__ == '__main__':
    unittest.main()
