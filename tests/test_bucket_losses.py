"""Loss regressions for depth-sorted BA batches; no CUDA extensions needed."""
import unittest

import torch
import torch.nn.functional as F

from model.utils import get_sdf_loss, masked_mse_loss


class BucketLossTests(unittest.TestCase):
    def test_depth_matches_valid_mean_and_gradient(self):
        for mask in (torch.tensor([True, False, True]), torch.ones(3, dtype=torch.bool)):
            pred = torch.tensor([0.4, 0.7, 1.2], requires_grad=True)
            target = torch.tensor([0.5, 0., 1.])
            actual = masked_mse_loss(pred, target, mask)
            expected = F.mse_loss(pred[mask], target[mask])
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(
                torch.autograd.grad(actual, pred)[0],
                torch.autograd.grad(expected, pred)[0],
            )

    def test_missing_and_truncated_depth_batches(self):
        for depth in (0., 5., 6.):
            pred = torch.randn(16, requires_grad=True)
            target = torch.full((16,), depth)
            loss = masked_mse_loss(pred, target, (target > 0) & (target < 5))
            self.assertEqual(loss.item(), 0.)
            loss.backward()
            torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_missing_depth_sdf_losses_and_gradients(self):
        z = torch.linspace(0., 5., 85).expand(16, -1)
        sdf = torch.randn_like(z, requires_grad=True)
        fs_loss, sdf_loss = get_sdf_loss(z, torch.zeros(16, 1), sdf, .05, 'l2')
        self.assertEqual(fs_loss.item(), 0.)
        self.assertEqual(sdf_loss.item(), 0.)
        (fs_loss + sdf_loss).backward()
        torch.testing.assert_close(sdf.grad, torch.zeros_like(sdf))

    def test_supervised_sdf_matches_original(self):
        z = torch.linspace(0., 2., 85).expand(3, -1)
        target = torch.tensor([[0.], [0.7], [1.2]])
        sdf = torch.randn_like(z, requires_grad=True)
        front = (z < target - .05).float()
        surface = ((z >= target - .05) & (z <= target + .05) & (target > 0)).float()
        nf, ns = front.count_nonzero(), surface.count_nonzero()
        expected = (
            F.mse_loss(sdf * front, front) * (1 - nf / (nf + ns)),
            F.mse_loss((z + sdf * .05) * surface, target * surface) * (1 - ns / (nf + ns)),
        )
        actual = get_sdf_loss(z, target, sdf, .05, 'l2')
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e)
        torch.testing.assert_close(
            torch.autograd.grad(sum(actual), sdf)[0],
            torch.autograd.grad(sum(expected), sdf)[0],
        )

    def test_missing_depth_eikonal(self):
        z = torch.linspace(0., 5., 85).expand(2, -1)
        sdf = torch.randn_like(z, requires_grad=True)
        grad = torch.randn(2, 85, 3, requires_grad=True)
        losses = get_sdf_loss(z, torch.zeros(2, 1), sdf, .05, 'l2', grad=grad)
        self.assertEqual(sum(losses).item(), 0.)
        sum(losses).backward()
        torch.testing.assert_close(grad.grad, torch.zeros_like(grad))


if __name__ == '__main__':
    unittest.main()
