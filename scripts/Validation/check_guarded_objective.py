"""Check guarded BA against the dense objective on a frozen trained map.

Call after recording ATE, timings, and benchmark sample counts. This performs
backward comparisons without optimizer updates and restores RNG/config/history.
It supports CPU fields for regression tests and the actual CUDA TCNN map.
"""
import math
import random

import numpy as np
import torch

from optimization.ba_experiments import forward_ba_lattice


def _gradient_error(dense, guarded):
    """Treat an unused gradient as zero and report its norm difference."""
    error_squared = reference_squared = 0.
    maximum = 0.
    for expected, actual in zip(dense, guarded):
        if expected is None and actual is None:
            continue
        if expected is None:
            expected = torch.zeros_like(actual)
        if actual is None:
            actual = torch.zeros_like(expected)
        expected, actual = expected.float(), actual.float()
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise AssertionError('Nonfinite dense or guarded gradient')
        difference = actual - expected
        error_squared += float(difference.square().sum())
        reference_squared += float(expected.square().sum())
        if difference.numel():
            maximum = max(maximum, float(difference.abs().max()))
    return dict(relative_norm_error=math.sqrt(error_squared / max(reference_squared, 1e-20)),
                maximum_absolute_error=maximum,
                dense_norm=math.sqrt(reference_squared))


def check_guarded_objective(slam, samples=256):
    """Compare dense/guarded losses and map/rotation/translation gradients.

    Frames use estimated poses, so unavailable ScanNet ground truth does not
    affect this check. The middle frame is also checked after a 0.2 m camera
    translation. Fixed CPU ray draws and paired Torch perturbations make the
    two forwards comparable. Existing parameter ``.grad`` values are untouched.
    """
    if isinstance(samples, bool) or int(samples) != samples or samples < 1:
        raise ValueError('samples must be a positive integer')
    model = slam.model
    config = model.config
    if config['training'].get('n_importance', 0):
        raise ValueError('The guarded objective check requires n_importance=0')
    frame_ids = sorted(slam.est_c2w_data)
    if not frame_ids:
        raise ValueError('No estimated camera poses to check')
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    device = next(model.parameters()).device
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    selected_frames = sorted({frame_ids[0], frame_ids[len(frame_ids) // 2], frame_ids[-1]})
    cases = [(frame, 0.) for frame in selected_frames]
    cases.append((frame_ids[len(frame_ids) // 2], .2))
    weights = tuple((loss, config['training'][weight]) for loss, weight in (
        ('rgb_loss', 'rgb_weight'), ('depth_loss', 'depth_weight'),
        ('sdf_loss', 'sdf_weight'), ('fs_loss', 'fs_weight')))
    original_experiment = config.get('ba_experiment')
    had_experiment = 'ba_experiment' in config
    had_counts = hasattr(model, 'ba_experiment_sample_counts')
    original_counts = getattr(model, 'ba_experiment_sample_counts', None)
    modes = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    results = []
    try:
        config['ba_experiment'] = dict(original_experiment or {}, collect_samples=True)
        model.ba_experiment_sample_counts = []
        model.train(True)
        with torch.random.fork_rng(devices=cuda_devices), torch.enable_grad():
            for frame, offset in cases:
                batch = slam.dataset[frame]
                directions = batch['direction'].reshape(-1, 3)
                count = min(int(samples), len(directions))
                if count < 1:
                    raise ValueError(f'Frame {frame} has no rays')
                # A private generator does not consume the experiment RNG.
                indices = torch.randperm(len(directions), generator=torch.Generator().manual_seed(17))[:count]
                directions = directions[indices].to(device)
                color = batch['rgb'].reshape(-1, 3)[indices].to(device)
                depth = batch['depth'].reshape(-1, 1)[indices].to(device)
                pose = slam.est_c2w_data[frame].detach().to(device)
                rotation = pose[:3, :3].clone().requires_grad_(True)
                translation = pose[:3, 3].clone()
                translation[0] += offset
                translation.requires_grad_(True)
                rays_d = directions @ rotation.T
                rays_o = translation[None, :].expand_as(rays_d)
                inputs = parameters + (rotation, translation)
                comparisons = []
                for guarded in (False, True):
                    # Seed only the devices fork_rng will restore.
                    torch.random.default_generator.manual_seed(73)
                    if cuda_devices:
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(73)
                    if guarded:
                        ret = forward_ba_lattice(model, rays_o, rays_d, color, depth,
                                                 guarded=True, margin=0.,
                                                 morton_sort_override=False)
                    else:
                        ret = model.forward(rays_o, rays_d, color, depth,
                                            uniform_samples_until_depth=False,
                                            morton_sort_override=False)
                    total = sum(weight * ret[key] for key, weight in weights)
                    # The two forwards share the ray transformation graph.
                    gradients = torch.autograd.grad(total, inputs, allow_unused=True,
                                                    retain_graph=not guarded)
                    comparisons.append((ret, gradients))
                dense, guarded = comparisons
                output_errors = {}
                context = f'frame={frame}, translation_offset_m={offset}'
                for key in ('rgb', 'depth', 'rgb_loss', 'depth_loss', 'sdf_loss', 'fs_loss'):
                    torch.testing.assert_close(dense[0][key], guarded[0][key],
                                               rtol=2e-4, atol=2e-6,
                                               msg=f'{context}: {key}')
                    output_errors[key] = float((dense[0][key] - guarded[0][key]).abs().max())
                gradient_errors = dict(
                    all=_gradient_error(dense[1], guarded[1]),
                    map=_gradient_error(dense[1][:-2], guarded[1][:-2]),
                    rotation=_gradient_error(dense[1][-2:-1], guarded[1][-2:-1]),
                    translation=_gradient_error(dense[1][-1:], guarded[1][-1:]))
                if gradient_errors['all']['relative_norm_error'] > .003:
                    raise AssertionError(f"{context}: gradient relative norm error "
                                         f"{gradient_errors['all']['relative_norm_error']} exceeds .003")
                for group in ('map', 'rotation', 'translation'):
                    error = gradient_errors[group]
                    if error['relative_norm_error'] > .003 and error['maximum_absolute_error'] > 2e-5:
                        raise AssertionError(f'{context}: {group} gradient mismatch: {error}')
                results.append(dict(frame=frame, translation_offset_m=offset, rays=count,
                    maximum_absolute_output_errors=output_errors,
                    losses_dense={key: float(dense[0][key]) for key, _ in weights},
                    losses_guarded={key: float(guarded[0][key]) for key, _ in weights},
                    gradient_errors=gradient_errors,
                    network_samples=dict(model.ba_experiment_sample_counts[-1])))
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        for module, training in modes:
            module.training = training
        if had_experiment:
            config['ba_experiment'] = original_experiment
        else:
            config.pop('ba_experiment', None)
        if had_counts:
            model.ba_experiment_sample_counts = original_counts
        else:
            del model.ba_experiment_sample_counts
    return dict(passed=True, requested_rays=int(samples), margin=0.,
                output_rtol=2e-4, output_atol=2e-6,
                gradient_relative_norm_tolerance=.003, results=results)
