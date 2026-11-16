"""Hash-grid Adam epsilon should damp tiny gradient noise during BA."""
import unittest

import torch

from optimization.ba_experiments import ba_bucket_mode, configure_ba_embed_eps


class BAOptimizerStabilityTests(unittest.TestCase):
    def test_small_gradient_noise_is_damped_without_changing_decoder_epsilon(self):
        def update(epsilon):
            decoder = torch.nn.Parameter(torch.ones(2))
            encoding = torch.nn.Parameter(torch.ones(2))
            optimizer = torch.optim.Adam([
                {'params': [decoder], 'eps': 1e-8},
                {'params': [encoding], 'eps': 1e-15},
            ], lr=.01, betas=(.9, .99))
            config = dict(training=dict(n_importance=0),
                          ba_bucket=dict(enabled=True, mode='guarded', embed_eps=epsilon))
            configure_ba_embed_eps(optimizer, config)
            self.assertEqual(optimizer.param_groups[0]['eps'], 1e-8)
            encoding.grad = torch.tensor([1e-12, 1e-4])
            optimizer.step()
            return (1. - encoding).detach()

        original = update(1e-15)
        damped = update(1e-8)
        self.assertLess(damped[0], original[0] * .001)
        torch.testing.assert_close(damped[1], original[1], rtol=2e-4, atol=1e-7)

    def test_disabled_bucket_and_unspecified_epsilon_preserve_optimizer(self):
        for bucket in (dict(enabled=False, embed_eps=1e-8), dict(enabled=True)):
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.Adam([{'params': [], 'eps': 1e-8},
                                          {'params': [parameter], 'eps': 1e-15}])
            configure_ba_embed_eps(optimizer, dict(ba_bucket=bucket))
            self.assertEqual(optimizer.param_groups[1]['eps'], 1e-15)

    def test_invalid_epsilon_is_rejected(self):
        for value in (0., -1., float('nan'), float('inf'), True):
            with self.subTest(epsilon=value), self.assertRaises(ValueError):
                ba_bucket_mode(dict(ba_bucket=dict(enabled=True, embed_eps=value)))


if __name__ == '__main__':
    unittest.main()
