"""Run in the activated coslam environment (loads CoSLAM's CUDA extensions)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from coslam import CoSLAM


class SmoothnessTests(unittest.TestCase):
    def test_half_features_accumulate_without_overflow(self):
        slam = CoSLAM.__new__(CoSLAM)
        slam.bounding_box = torch.tensor([[0., 1.], [0., 1.], [0., 1.]])
        slam.config = {'grid': {'tcnn_encoding': True}}
        features = ((torch.arange(7)[:, None, None, None] % 2) * 4.)
        features = features.expand(7, 7, 7, 128).half().clone().requires_grad_()
        slam.model = SimpleNamespace(query_sdf=lambda *a, **kw: features)
        overflowing = (features[1:] - features[:-1]).square().sum() / 8**3
        self.assertTrue(torch.isinf(overflowing))
        actual = slam.smoothness(sample_points=8)
        reference = (features.float()[1:] - features.float()[:-1]).square().sum() / 8**3
        torch.testing.assert_close(actual, reference)
        actual_grad = torch.autograd.grad(actual, features)[0]
        reference_grad = torch.autograd.grad(reference, features)[0]
        self.assertTrue(torch.isfinite(actual_grad).all())
        torch.testing.assert_close(actual_grad, reference_grad)


if __name__ == '__main__':
    unittest.main()
