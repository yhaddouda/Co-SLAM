"""Guarded BA must retain the dense objective on curved, multi-surface fields.

Use the actual dense renderer through the existing CPU helper. These cases
exercise more than a single planar crossing and compare gradients of the
field, ray origins, and ray directions without importing CUDA encoders.
"""
import importlib.util
import json
import random
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
helper_spec = importlib.util.spec_from_file_location(
    "_ba_dense_renderer_helper", ROOT / "tests/test_ba_experiments.py")
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)

from optimization.ba_experiments import forward_ba_lattice
from scripts.Validation.check_guarded_objective import check_guarded_objective


class CurvedField(helper.AnalyticField):
    def __init__(self, mode="oscillating"):
        super().__init__()
        self.mode = mode
        self.coefficients = torch.tensor([.35, .21, .17, 6.7], requires_grad=True)
        self.config['cam'].update(near=0., far=4., depth_trunc=5.)
        self.config['training'].update(n_samples_d=33, n_range_d=11,
                                       range_d=.125, trunc=.125)
        self.queried_points = []

    def run_network(self, points, **kwargs):
        self.queried_points.append(points.detach().clone())
        a, b, c, frequency = self.coefficients
        x, y, z = points.unbind(-1)
        if self.mode == "oscillating":
            # Many crossings; the surface positions vary with x and y.
            sdf = torch.sin(frequency * z + a * x + b * y.square() + c)
        elif self.mode == "late_surface":
            sdf = 2.8 + a * x.square() + b * y - z
        elif self.mode == "no_crossing":
            sdf = 1. + (a * x + b * y + c * z).square()
        elif self.mode == "boundary_surface":
            sdf = 2.0625 + a * x.square() + b * y - z
        else:
            raise ValueError(self.mode)
        rgb = torch.stack([torch.sin(x + a * z), y.square() * b + z * c,
                           torch.cos(y + c * z) + a * x], -1)
        return torch.cat([rgb, sdf[..., None]], -1)


class ModuleCurvedField(CurvedField, torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        CurvedField.__init__(self)
        self.coefficients = torch.nn.Parameter(self.coefficients)


class DiagnosticDataset:
    def __getitem__(self, frame):
        # Confirm that even any incidental dataset RNG use is restored.
        random.random()
        np.random.rand()
        torch.rand(1)
        return dict(direction=torch.tensor([.1, -.05, 1.]).repeat(8, 1),
                    rgb=torch.full((8, 3), .5), depth=torch.full((8,), 2.))


class GuardedFieldTests(unittest.TestCase):
    def setUp(self):
        self.nvtx = patch('torch.cuda.nvtx.range',
                          lambda *args, **kwargs: nullcontext())
        self.nvtx.start()
        self.addCleanup(self.nvtx.stop)

    def compare(self, mode="oscillating", depths=(0., 1.5, 2., 2.5, 6.),
                perturb=1, scale=1., trunc=.125, margin=0., boundary=False,
                depth_range=.125):
        field = CurvedField(mode)
        field.config['training'].update(perturb=perturb, trunc=trunc,
                                       range_d=depth_range)
        field.config['data']['sc_factor'] = scale
        count = len(depths)
        origins = (torch.arange(count * 3).reshape(count, 3).float() * .007)
        directions = torch.tensor([.13, -.07, 1.]).repeat(count, 1)
        if boundary:
            origins = torch.zeros(count, 3)
            directions = torch.tensor([0., 0., 1.]).repeat(count, 1)
        origins.requires_grad_(True)
        directions.requires_grad_(True)
        target = torch.tensor(depths)[:, None]
        rgb = torch.linspace(.15, .85, count * 3).reshape(count, 3)

        torch.manual_seed(197)
        dense = field.forward(origins, directions, rgb, target)
        dense_rng = torch.get_rng_state()
        dense_points = field.queried_points.pop()
        torch.manual_seed(197)
        guarded = forward_ba_lattice(field, origins, directions, rgb, target,
                                     guarded=True, margin=margin)
        torch.testing.assert_close(torch.get_rng_state(), dense_rng)
        for key in ('rgb', 'depth', 'rgb_loss', 'depth_loss', 'sdf_loss', 'fs_loss'):
            torch.testing.assert_close(guarded[key], dense[key], rtol=2e-5,
                                       atol=2e-6, msg=key)
        weights = dict(rgb_loss=5., depth_loss=.1, sdf_loss=1000., fs_loss=10.)
        inputs = (field.coefficients, origins, directions)
        dense_total = sum(weights[key] * dense[key] for key in weights)
        guarded_total = sum(weights[key] * guarded[key] for key in weights)
        for actual, expected in zip(torch.autograd.grad(guarded_total, inputs),
                                    torch.autograd.grad(dense_total, inputs)):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-5)
        return field, dense_points

    def test_nonplanar_multiple_crossings_preserve_losses_and_all_gradients(self):
        for perturb in (0, 1):
            with self.subTest(perturb=perturb):
                field, _ = self.compare(perturb=perturb)
                counts = field.ba_experiment_sample_counts[-1]
                self.assertLess(counts['queried_samples'], counts['dense_samples'])

    def test_scaled_truncation_and_depth_margin_preserve_dense_objective(self):
        for scale, trunc, margin in ((.4, .125, 0.), (2., .125, .075),
                                     (1., .05, .2), (1., .25, 0.)):
            with self.subTest(scale=scale, trunc=trunc, margin=margin):
                self.compare(scale=scale, trunc=trunc, margin=margin)

    def test_missing_depth_and_beyond_far_keep_dense_network_queries(self):
        for depths in ((0., 0.), (6., 6.), (-1., 0.)):
            with self.subTest(depths=depths):
                field, _ = self.compare(depths=depths)
                counts = field.ba_experiment_sample_counts[-1]
                self.assertEqual(counts['queried_samples'], counts['dense_samples'])

    def test_late_curved_surface_and_no_crossing_query_full_tail(self):
        for mode in ('late_surface', 'no_crossing'):
            with self.subTest(mode=mode):
                field, _ = self.compare(mode=mode, depths=(1.5, 1.75))
                counts = field.ba_experiment_sample_counts[-1]
                self.assertGreater(counts['fallback_samples'], 0)
                self.assertEqual(counts['queried_samples'], counts['dense_samples'])

    def test_exact_truncation_boundary_is_retained(self):
        # Binary-exact d + scaled truncation = 2.125 is both a uniform
        # sample and the end of the centered band. Geometry supervises it.
        field, dense_points = self.compare(depths=(2., 2.), perturb=0,
                                           scale=.5, trunc=.25, boundary=True)
        self.assertTrue((dense_points[:, 2] == 2.125).any())
        queried = torch.cat(field.queried_points)
        self.assertEqual(int((queried[:, 2] == 2.125).sum()),
                         int((dense_points[:, 2] == 2.125).sum()))
        counts = field.ba_experiment_sample_counts[-1]
        self.assertLess(counts['queried_samples'], counts['dense_samples'])

    def test_first_crossing_near_depth_edge_checks_render_support(self):
        field, _ = self.compare(mode='boundary_surface', depths=(2., 2.),
                                perturb=0, boundary=True, trunc=.25)
        counts = field.ba_experiment_sample_counts[-1]
        # Here the retained prefix already includes all rendering support;
        # the same curved field with a smaller cutoff needs fallback.
        self.assertLess(counts['queried_samples'], counts['dense_samples'])
        field, _ = self.compare(mode='boundary_surface', depths=(2., 2.),
                                perturb=0, boundary=True, trunc=.125,
                                depth_range=.25)
        self.assertGreater(field.ba_experiment_sample_counts[-1]['fallback_samples'], 0)

    def test_frozen_map_check_restores_rng_parameters_gradients_and_metadata(self):
        from types import SimpleNamespace
        field = ModuleCurvedField()
        field.train(False)
        original_experiment = dict(variant='baseline', collect_samples=False)
        field.config['ba_experiment'] = original_experiment
        original_history = [dict(marker='existing benchmark samples')]
        field.ba_experiment_sample_counts = original_history
        gradient = torch.arange(4).float()
        field.coefficients.grad = gradient
        parameters = field.coefficients.detach().clone()
        torch_state, python_state = torch.get_rng_state(), random.getstate()
        numpy_state = np.random.get_state()
        slam = SimpleNamespace(model=field, dataset=DiagnosticDataset(),
                               est_c2w_data={i: torch.eye(4) for i in (0, 2, 4)})
        report = check_guarded_objective(slam, samples=6)
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['results']), 4)
        json.dumps(report, allow_nan=False)
        self.assertFalse(field.training)
        self.assertIs(field.config['ba_experiment'], original_experiment)
        self.assertIs(field.ba_experiment_sample_counts, original_history)
        self.assertIs(field.coefficients.grad, gradient)
        torch.testing.assert_close(field.coefficients, parameters)
        torch.testing.assert_close(torch.get_rng_state(), torch_state)
        self.assertEqual(random.getstate(), python_state)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_state[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_state[1])
        self.assertEqual(after_numpy[2:], numpy_state[2:])


if __name__ == '__main__':
    unittest.main()
