"""Optional execution experiments for dense tracking; no sampling changes.

The baseline remains available. Device constants use the existing pose math;
deferred best-pose selection is eligible only when early stopping cannot fire.
"""
import torch

VARIANTS = {'baseline', 'reuse_pose', 'device_pose', 'defer_best', 'baseline_fast',
            'cache_depth', 'cached_fast'}


class DenseDepthLattice:
    """Cache the unchanged dense depth lattice/bins within one tracking frame.

    Construct it anew per fixed pixel batch. Random perturbations remain in the
    renderer, at their original iteration boundary and with the original RNG.
    Target depths must be fixed, without an autograd dependency.
    """

    def __init__(self, config):
        self.config = config
        self.base = self.lower = self.upper = None
        self.key = None

    def prepare(self, rays, target):
        key = (rays.shape[0], rays.device, rays.dtype, id(target))
        if self.base is not None:
            if key != self.key:
                raise ValueError('Depth cache must be reset for a new tracking batch')
            return self.base
        if target is not None and target.requires_grad:
            raise ValueError('Depth caching requires fixed target depths without gradients')
        self.key = key
        training, camera = self.config['training'], self.config['cam']
        if target is not None:
            surface = torch.linspace(-training['range_d'], training['range_d'],
                                     steps=training['n_range_d']).to(target)
            surface = surface[None, :].repeat(rays.shape[0], 1) + target
            surface[target.squeeze() <= 0] = torch.linspace(
                camera['near'], camera['far'], steps=training['n_range_d']).to(target)
            if training['n_samples_d'] > 0:
                uniform = torch.linspace(camera['near'], camera['far'],
                                         training['n_samples_d'])[None, :].repeat(rays.shape[0], 1).to(rays)
                self.base, _ = torch.sort(torch.cat([uniform, surface], -1), -1)
            else:
                self.base = surface
        else:
            self.base = torch.linspace(camera['near'], camera['far'], training['n_samples']).to(rays)
            self.base = self.base[None, :].repeat(rays.shape[0], 1)
        if training['perturb'] > 0:
            mids = .5 * (self.base[..., 1:] + self.base[..., :-1])
            self.upper = torch.cat([mids, self.base[..., -1:]], -1)
            self.lower = torch.cat([self.base[..., :1], mids], -1)
        return self.base


class DevicePoseTransform:
    """The existing transform formula with identities allocated on the device."""

    def __init__(self, reference, representation='axis_angle'):
        self.representation = representation
        self.eye4 = torch.eye(4, device=reference.device, dtype=reference.dtype)
        self.eye3 = torch.eye(3, device=reference.device, dtype=reference.dtype)

    def __call__(self, rotation, translation):
        if self.representation == 'axis_angle':
            theta = torch.norm(rotation, dim=-1, keepdim=True)
            omega = rotation / theta
            x, y, z = omega[..., 0:1], omega[..., 1:2], omega[..., 2:3]
            zero = torch.zeros_like(x)
            skew = torch.cat([
                torch.cat([zero, -z, y], -1)[..., None, :],
                torch.cat([z, zero, -x], -1)[..., None, :],
                torch.cat([-y, x, zero], -1)[..., None, :],
            ], -2)
            identity = self.eye3.expand(*rotation.shape[:-1], 3, 3)
            matrix = (identity + torch.sin(theta).unsqueeze(-1) * skew
                      + (1. - torch.cos(theta).unsqueeze(-1)) * (skew @ skew))
        elif self.representation == 'quat':
            from pytorch3d.transforms import quaternion_to_matrix
            matrix = quaternion_to_matrix(rotation)
        else:
            raise ValueError(f'Unsupported rotation representation: {self.representation}')
        transform = self.eye4[None].repeat(rotation.shape[0], 1, 1)
        transform[:, :3, :3] = matrix
        transform[:, :3, 3] = translation
        return transform


class DeferredBestPose:
    """Record detached pre-update poses and select the first strict minimum.

    Co-SLAM initializes its best loss from iteration 0, including a possible NaN.
    Later NaNs never improve that best. Keep that behavior instead of depending
    on argmin's handling of NaNs. Only the final index crosses to the CPU.
    """

    def __init__(self, iterations, reference):
        self.losses = reference.new_empty(iterations)
        self.poses = reference.new_empty((iterations,) + tuple(reference.shape))
        self.count = 0

    def add(self, loss, pose):
        with torch.no_grad():
            if self.count == 0 and self.losses.dtype != loss.dtype:
                self.losses = loss.new_empty(self.losses.shape)
            self.losses[self.count].copy_(loss.detach())
            self.poses[self.count].copy_(pose.detach())
        self.count += 1

    def best(self):
        if not self.count:
            raise ValueError('No tracking iteration was recorded')
        losses = self.losses[:self.count]
        comparable = torch.where(torch.isnan(losses), float('inf'), losses)
        index = comparable.argmin()
        index = torch.where(torch.isnan(losses[0]), torch.zeros_like(index), index)
        # One host scalar at finalization, instead of per-iteration loss copies.
        return self.poses[int(index.item())]


def tracking_execution_flags(config):
    variant = config.get('tracking_experiment', {}).get('variant', 'baseline')
    if variant not in VARIANTS:
        raise ValueError(f'Unknown tracking_experiment.variant: {variant}')
    return dict(
        reuse_pose=variant in {'reuse_pose', 'baseline_fast', 'cached_fast'},
        device_pose=variant in {'device_pose', 'baseline_fast', 'cached_fast'},
        defer_best=(variant in {'defer_best', 'baseline_fast', 'cached_fast'}
                    and config['tracking']['wait_iters'] >= config['tracking']['iter']),
        cache_depth=(variant in {'cache_depth', 'cached_fast'}
                     and not config.get('training', {}).get('uniform_samples_until_depth', False)
                     and not any(config.get(name, {}).get('enabled', False) for name in
                                 ('tracking_bucket', 'tracking_depth_spatial', 'tracking_naive_pruning'))),
    )
