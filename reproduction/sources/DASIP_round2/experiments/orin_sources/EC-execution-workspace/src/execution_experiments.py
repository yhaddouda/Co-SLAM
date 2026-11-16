"""Optional execution variants; EC-SLAM's baseline code stays selectable."""
import torch

TRACKING_VARIANTS = {'baseline', 'reuse_pose', 'device_pose', 'defer_best', 'baseline_fast'}
MAPPING_VARIANTS = {'baseline', 'cache_poses', 'reuse_inverse', 'baseline_fast'}


def tracking_flags(cfg):
    variant = cfg.get('tracking_experiment', {}).get('variant', 'baseline')
    if variant not in TRACKING_VARIANTS:
        raise ValueError(f'Unknown tracking variant: {variant}')
    return dict(reuse_pose=variant in {'reuse_pose', 'baseline_fast'},
                device_pose=variant in {'device_pose', 'baseline_fast'},
                defer_best=(variant in {'defer_best', 'baseline_fast'}
                            and not cfg['verbose']
                            and not (cfg['debug']['flag'] and cfg['debug']['showTrackerLoss'])))


def mapping_flags(cfg):
    variant = cfg.get('mapping_experiment', {}).get('variant', 'baseline')
    if variant not in MAPPING_VARIANTS:
        raise ValueError(f'Unknown mapping variant: {variant}')
    return dict(cache_poses=variant in {'cache_poses', 'baseline_fast'},
                reuse_inverse=variant in {'reuse_inverse', 'baseline_fast'})


def camera_from_tensor_device(inputs):
    """Match utils.quad2rotation/get_camera_from_tensor without CPU zero copies."""
    singleton = inputs.ndim == 1
    if singleton:
        inputs = inputs.unsqueeze(0)
    quad, translation = inputs[:, :4], inputs[:, 4:]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    # The baseline's torch.zeros uses the default dtype, not quad.dtype.
    rotation = torch.zeros(quad.shape[0], 3, 3, device=quad.device,
                           dtype=torch.get_default_dtype())
    rotation[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rotation[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rotation[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rotation[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rotation[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rotation[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rotation[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rotation[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rotation[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    camera = torch.cat([rotation, translation[:, :, None]], 2)
    return camera[0] if singleton else camera


class DeferredCameraSelection:
    """Preserve EC-SLAM's strict loss minimum with its initial 1e10 ceiling."""

    def __init__(self, iterations, camera, keep_matrix=False):
        self.losses = camera.new_empty(iterations)
        self.cameras = camera.new_empty(iterations, *camera.shape)
        self.matrices = camera.new_empty(iterations, 3, 4) if keep_matrix else None
        self.count = 0

    def add(self, loss, camera, matrix):
        with torch.no_grad():
            if self.count == 0 and self.losses.dtype != loss.dtype:
                self.losses = loss.new_empty(self.losses.shape)
            self.losses[self.count].copy_(loss.detach())
            self.cameras[self.count].copy_(camera.detach())
            if self.matrices is not None:
                self.matrices[self.count].copy_(matrix.detach())
        self.count += 1

    def best(self, last_best=False):
        if not self.count:
            raise ValueError('No tracking iteration was recorded')
        if last_best:
            index = self.count - 1
        else:
            losses = self.losses[:self.count]
            eligible = torch.where(losses < 1e10, losses, float('inf'))
            index = int(eligible.argmin().item())
            if not bool((eligible[index] < 1e10).item()):
                raise RuntimeError('No tracking loss improves EC-SLAM\'s initial 1e10 ceiling')
        matrix = self.matrices[index] if self.matrices is not None else None
        return self.cameras[index], matrix
