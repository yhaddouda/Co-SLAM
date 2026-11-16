"""Optional BA sampling experiments; baseline and legacy buckets stay separate.

The lattice variants preserve the baseline ray batch, its 43-sample lattice,
perturbations, and geometry-loss denominator. Only network queries are packed.
The guarded variant queries omitted points when the renderer's first crossing
or support cannot be established from the retained prefix.
"""
import torch
from contextlib import nullcontext

from model.utils import compute_loss, get_sdf_loss, masked_mse_loss, mse2psnr

VARIANTS = {'baseline', 'bucket', 'bucket_dense', 'bucket_reference', 'lattice', 'lattice_guarded',
            'lattice_guarded_preload', 'baseline_fast', 'lattice_guarded_fast'}


def ba_execution_flags(config):
    """Resolve independent execution switches, retaining each variant's defaults."""
    experiment = config.get('ba_experiment', {})
    variant = experiment.get('variant', 'baseline')
    if variant not in VARIANTS:
        raise ValueError(f'Unknown ba_experiment.variant: {variant}')
    flags = dict(
        preload_rays=variant in {'lattice_guarded_preload', 'baseline_fast', 'lattice_guarded_fast'},
        fast_pose_commit=variant in {'baseline_fast', 'lattice_guarded_fast'},
    )
    for name in flags:
        value = experiment.get(name)
        if value is not None:
            if not isinstance(value, bool):
                raise ValueError(f'ba_experiment.{name} must be true, false or null')
            flags[name] = value
    if flags['preload_rays'] and config.get('ba_bucket', {}).get('enabled', False):
        raise ValueError('ba_experiment.preload_rays requires ba_bucket.enabled=false')
    return flags


def _ba_stage(slam, name):
    timer = getattr(slam, 'coslam_timing', None)
    return timer.stage(name) if timer is not None else nullcontext()


def commit_ba_poses_fast(slam, optimized_poses, frame_ids, current_frame_id):
    """Reuse the already updated pose batch instead of rebuilding every camera."""
    # Paper Sec. 3.3 optimizes pose parameters; this is storage of their latest
    # matrices, already refreshed after Adam. See docs/ba_architecture_timing/.
    with _ba_stage(slam, 'BA_COMMIT_BATCH_COPY'):
        poses = optimized_poses.detach().clone()
    with _ba_stage(slam, 'BA_COMMIT_ASSIGN'):
        for index, frame_id in enumerate(frame_ids[1:].tolist()):
            slam.est_c2w_data[int(frame_id)] = poses[index]
        if slam.config['mapping']['optim_cur']:
            slam.est_c2w_data[current_frame_id] = poses[-1]


def prepare_mixed_ba_batches(slam, current_rays):
    """Preload exactly the baseline's per-iteration ray draws, without sorting.

    The ray database is unchanged during a BA invocation. Only Python's RNG is
    consumed by these draws, so moving them before network/smoothness evaluation
    preserves the baseline sampling sequence and Torch perturbation sequence.
    """
    mapping = slam.config['mapping']
    current_sample = max(mapping['sample'] // len(slam.keyframeDatabase.frame_ids), mapping['min_pixels_cur'])
    import random
    rays_all, ids_all = [], []
    # This prepares observations from the paper's global pixel set; it does not
    # replace Sec. 3.2 depth sampling or cache pose-dependent world points.
    with _ba_stage(slam, 'BA_PRELOAD_DRAW'):
        for _ in range(mapping['iters']):
            rays, ids = slam.keyframeDatabase.sample_global_rays(mapping['sample'])
            current_indices = random.sample(range(slam.dataset.H * slam.dataset.W), current_sample)
            rays_all.append(torch.cat([rays, current_rays[current_indices, :]], dim=0))
            ids_all.append(torch.cat([ids // mapping['keyframe_every'], -torch.ones(current_sample)]).to(torch.int64))
    with _ba_stage(slam, 'BA_PRELOAD_PACK_TRANSFER'):
        rays = torch.stack(rays_all).to(slam.device)
        ids = torch.stack(ids_all).to(slam.device)
    return dict(rays_d_cam=rays[..., :3], target_s=rays[..., 3:6], target_d=rays[..., 6:7], ids_all=ids)


def baseline_depth_samples(config, target_d):
    """Match render_rays' dense target-depth path, including RNG draw shape."""
    rays = target_d.shape[0]
    training, camera = config['training'], config['cam']
    centered = torch.linspace(-training['range_d'], training['range_d'],
                              training['n_range_d']).to(target_d)
    centered = centered[None, :].repeat(rays, 1) + target_d
    centered[target_d.squeeze(-1) <= 0] = torch.linspace(
        camera['near'], camera['far'], training['n_range_d']).to(target_d)
    if training['n_samples_d'] > 0:
        uniform = torch.linspace(camera['near'], camera['far'], training['n_samples_d']).to(target_d)
        z_vals = torch.sort(torch.cat([uniform[None, :].repeat(rays, 1), centered], -1), -1).values
    else:
        z_vals = centered
    if training['perturb'] > 0:
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # The baseline draws on CPU and transfers to the ray device.
        z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(target_d)
    return z_vals


def render_ba_lattice(model, rays_o, rays_d, target_d, guarded=True, margin=0.):
    config = model.config
    if config['training']['n_importance']:
        raise ValueError('BA lattice experiments currently require n_importance=0')
    z_vals = baseline_depth_samples(config, target_d)
    truncation = config['training']['trunc'] * config['data']['sc_factor']
    valid = torch.isfinite(target_d) & (target_d > 0)
    keep = ~valid | (z_vals <= target_d + truncation + margin)
    # Preserve the first sample for the renderer's existing no-crossing behavior.
    keep[:, 0] = True
    points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]
    points_flat = points.reshape(-1, 3)
    indices = torch.nonzero(keep.reshape(-1), as_tuple=False).flatten()
    packed_raw = model.run_network(points_flat[indices])
    raw_flat = packed_raw.new_zeros((points_flat.shape[0], 4)).index_copy(0, indices, packed_raw)
    extra_count = 0
    if guarded:
        with torch.no_grad():
            sdf = raw_flat.reshape(*z_vals.shape, 4)[..., 3]
            crossings = (sdf[:, 1:] * sdf[:, :-1] < 0) & keep[:, 1:] & keep[:, :-1]
            first = crossings.long().argmax(dim=1, keepdim=True)
            z_min = z_vals.gather(1, first)
            render_support = z_vals < z_min + truncation
            needs_full = ~crossings.any(dim=1) | (render_support & ~keep).any(dim=1)
            extra_mask = needs_full[:, None] & ~keep
            extra_indices = torch.nonzero(extra_mask.reshape(-1), as_tuple=False).flatten()
        extra_count = extra_indices.numel()
        if extra_count:
            extra_raw = model.run_network(points_flat[extra_indices])
            raw_flat = raw_flat.index_copy(0, extra_indices, extra_raw)
            keep = keep | extra_mask
    raw = raw_flat.reshape(*z_vals.shape, 4)
    rgb, disp, acc, weights, depth, variance = model.raw2outputs(
        raw, z_vals, config['training']['white_bkgd'], sample_mask=keep)
    if config.get('ba_experiment', {}).get('collect_samples', False):
        if not hasattr(model, 'ba_experiment_sample_counts'):
            model.ba_experiment_sample_counts = []
        model.ba_experiment_sample_counts.append(dict(
            dense_samples=points_flat.shape[0], queried_samples=indices.numel() + extra_count,
            fallback_samples=extra_count))
    return dict(rgb=rgb, depth=depth, disp_map=disp, acc_map=acc,
                depth_var=variance, z_vals=z_vals, raw=raw)


def forward_ba_lattice(model, rays_o, rays_d, target_rgb, target_d, guarded=True, margin=0.):
    rendered = render_ba_lattice(model, rays_o, rays_d, target_d, guarded, margin)
    valid_depth = (target_d.squeeze(-1) > 0) & (target_d.squeeze(-1) < model.config['cam']['depth_trunc'])
    # Deliberately reproduce forward's current RGB weighting, including dtype.
    rgb_weight = valid_depth.clone().unsqueeze(-1)
    rgb_weight[rgb_weight == 0] = model.config['training']['rgb_missing']
    rgb_loss = compute_loss(rendered['rgb'] * rgb_weight, target_rgb * rgb_weight)
    depth_loss = masked_mse_loss(rendered['depth'].reshape(-1), target_d.squeeze(-1), valid_depth)
    truncation = model.config['training']['trunc'] * model.config['data']['sc_factor']
    # The dense lattice retains its denominator and global mask counts. All
    # omitted samples lie beyond depth + truncation, where geometry masks are 0.
    fs_loss, sdf_loss = get_sdf_loss(rendered['z_vals'], target_d, rendered['raw'][..., -1],
                                     truncation, 'l2', grad=None)
    return dict(rgb=rendered['rgb'], depth=rendered['depth'], rgb_loss=rgb_loss,
                depth_loss=depth_loss, sdf_loss=sdf_loss, fs_loss=fs_loss, psnr=mse2psnr(rgb_loss))
