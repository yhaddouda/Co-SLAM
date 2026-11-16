"""Run a scene with bucketed BA and fail at the first nonfinite loss.

Run from the repository root in the coslam environment:
    python scripts/Validation/check_ba_bucket.py --frames 100
Use --frames 0 to process the full scene. Validation synchronizes the GPU;
its runtime must not be used as a performance benchmark.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import config
import coslam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/Tum/fr1_desk.yaml')
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--output', default='/tmp/coslam-ba-validation')
    args = parser.parse_args()
    cfg = config.load_config(args.config)
    cfg.setdefault('timing', {}).update(
        mode='none', max_frames=args.frames or None, disable_eval=True,
    )
    cfg['data'].update(output=args.output, exp_name='finite_check')
    cfg['mapping']['first_mesh'] = False
    cfg['ba_bucket']['enabled'] = True
    cfg.setdefault('tracking_depth_spatial', {})['collect_sample_count_stats'] = False
    Path(args.output, 'finite_check').mkdir(parents=True, exist_ok=True)
    # Existing optimizer code in CoSLAM reads the module-level configuration.
    coslam.cfg = cfg
    slam = coslam.CoSLAM(cfg)
    slam.seed_everything(0)
    forward = slam.model.forward
    calls = 0

    def checked_forward(*inputs, **kwargs):
        nonlocal calls
        ret = forward(*inputs, **kwargs)
        calls += 1
        for key in ('rgb_loss', 'depth_loss', 'fs_loss', 'sdf_loss'):
            if key in ret and not torch.isfinite(ret[key]).all():
                depth = inputs[3]
                valid = (depth > 0) & (depth < cfg['cam']['depth_trunc'])
                raise RuntimeError(
                    f'Nonfinite {key} at forward {calls}: '
                    f'uniform_count={kwargs.get("uniform_sample_count")}, '
                    f'valid_depths={valid.sum().item()}/{depth.numel()}'
                )
        return ret

    slam.model.forward = checked_forward
    get_loss = slam.get_loss_from_ret

    def checked_loss(*inputs, **kwargs):
        loss = get_loss(*inputs, **kwargs)
        if not torch.isfinite(loss).all():
            raise RuntimeError(f'Nonfinite total loss at forward {calls}')
        return loss

    slam.get_loss_from_ret = checked_loss
    slam.run()
    for name, param in slam.model.named_parameters():
        if not torch.isfinite(param).all():
            raise RuntimeError(f'Nonfinite model parameter: {name}')
    for frame, pose in slam.est_c2w_data.items():
        if not torch.isfinite(pose).all():
            raise RuntimeError(f'Nonfinite pose: {frame}')
    print(f'PASS: {len(slam.est_c2w_data)} poses, {calls} forwards; '
          'finite losses, final parameters and poses.')


if __name__ == '__main__':
    main()
