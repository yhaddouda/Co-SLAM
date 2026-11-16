"""Paired BA diagnostics or clean frame timing (run each mode in a fresh process).

Example: python scripts/Validation/compare_ba_bucket.py --mode bucket --diagnostic
Diagnostics are intentionally separate from timing: their extra GPU work must
not be interpreted as application performance. Frame wall time includes data
loading, tracking, BA preprocessing/optimization, and keyframe insertion.
"""
import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import config
import coslam


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['baseline', 'bucket'], required=True)
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', default='configs/Tum/fr1_desk.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.load_config(args.config)
    cfg['ba_bucket']['enabled'] = args.mode == 'bucket'
    cfg['timing'].update(mode='none', max_frames=args.frames or None, disable_eval=True)
    cfg['data'].update(output=str(out), exp_name='checkpoint')
    cfg['mapping']['first_mesh'] = False
    cfg.setdefault('tracking_depth_spatial', {})['collect_sample_count_stats'] = False
    (out / 'checkpoint').mkdir(exist_ok=True)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    coslam.cfg = cfg
    slam = coslam.CoSLAM(cfg)
    state = dict(phase='', frame=0, iteration=0, samples=0, ba_calls=0, bucket_calls=0)
    records, tensors, probes, frames = [], [], [], []
    forward = slam.model.forward
    render = slam.model.render_rays
    loss_fn = slam.get_loss_from_ret
    prepare = slam.prepare_ba_buckets

    def prepare_checked(*inputs, **kwargs):
        state['bucket_calls'] += 1
        return prepare(*inputs, **kwargs)
    slam.prepare_ba_buckets = prepare_checked

    def render_checked(*inputs, **kwargs):
        ret = render(*inputs, **kwargs)
        state['samples'] = ret['z_vals'].shape[-1]
        return ret

    def forward_checked(*inputs, **kwargs):
        ret = forward(*inputs, **kwargs)
        if state['phase'] in ('initial', 'tracking', 'ba'):
            depth = inputs[3]
            valid = (depth > 0) & (depth < cfg['cam']['depth_trunc'])
            state['iteration'] += 1
            records.append(dict(phase=state['phase'], frame=state['frame'],
                                iteration=state['iteration'], rays=depth.numel(),
                                samples=state['samples'],
                                pruned=int(bool(kwargs.get('uniform_samples_until_depth')))))
            tensors.append(torch.stack([ret[k].detach() for k in
                           ('rgb_loss', 'depth_loss', 'sdf_loss', 'fs_loss')] +
                           [valid.sum().float()]))
        return ret

    totals, steps = [], []
    def loss_checked(*inputs, **kwargs):
        loss = loss_fn(*inputs, **kwargs)
        if state['phase'] in ('initial', 'tracking', 'ba'):
            totals.append(loss.detach())
        return loss

    # A fixed dense-rendered frame-zero probe makes the comparison independent
    # of the different training batches and per-ray sample counts.
    probe_inputs = None
    def evaluate_probe(label):
        perturb = cfg['training']['perturb']
        cfg['training']['perturb'] = 0
        with torch.no_grad(), torch.random.fork_rng(devices=[slam.device]):
            ret = forward(*probe_inputs, uniform_samples_until_depth=False)
        cfg['training']['perturb'] = perturb
        probes.append(dict(frame=state['frame'], label=label, **{
            k: ret[k].item() for k in ('rgb_loss', 'depth_loss', 'sdf_loss', 'fs_loss')}))

    create_optimizer = slam.create_optimizer
    def create_checked():
        create_optimizer()
        step = slam.map_optimizer.step
        # The embedding parameter provides direct evidence of real map updates.
        param = slam.map_optimizer.param_groups[-1]['params'][-1]
        def step_checked(*inputs, **kwargs):
            if state['phase'] != 'ba':
                return step(*inputs, **kwargs)
            before = param.detach().clone()
            grad_norm = torch.stack([p.grad.detach().float().square().sum()
                for group in slam.map_optimizer.param_groups for p in group['params']
                if p.grad is not None]).sum().sqrt()
            result = step(*inputs, **kwargs)
            steps.append(torch.stack([grad_norm, (param.detach() - before).norm()]))
            return result
        slam.map_optimizer.step = step_checked

    def wrap_phase(name, method):
        def wrapped(batch, *inputs, **kwargs):
            nonlocal probe_inputs
            state.update(phase=name, iteration=0)
            if name != 'initial':
                state['frame'] = int(inputs[0])
            if name == 'ba':
                state['ba_calls'] += 1
            if args.diagnostic and name == 'initial':
                index = torch.randperm(slam.dataset.H * slam.dataset.W,
                                       generator=torch.Generator().manual_seed(2026))[:1024]
                direction = batch['direction'].reshape(-1, 3)[index].to(slam.device)
                rgb = batch['rgb'].reshape(-1, 3)[index].to(slam.device)
                depth = batch['depth'].reshape(-1, 1)[index].to(slam.device)
                pose = batch['c2w'].squeeze(0).to(slam.device)
                rays_d = torch.sum(direction[:, None, :] * pose[:3, :3], -1)
                rays_o = pose[:3, 3].expand_as(rays_d)
                probe_inputs = (rays_o, rays_d, rgb, depth)
                evaluate_probe('before_initial')
            result = method(batch, *inputs, **kwargs)
            if args.diagnostic and name in ('initial', 'ba'):
                evaluate_probe('after_' + name)
            state['phase'] = ''
            return result
        return wrapped

    if args.diagnostic:
        slam.model.render_rays = render_checked
        slam.model.forward = forward_checked
        slam.get_loss_from_ret = loss_checked
        slam.create_optimizer = create_checked
    for name, attr in [('initial', 'first_frame_mapping'), ('tracking', 'tracking_render'), ('ba', 'global_BA')]:
        setattr(slam, attr, wrap_phase(name, getattr(slam, attr)))
    finish = slam.coslam_timing.finish_frame
    last_end = None
    def finish_checked(frame):
        nonlocal last_end
        finish(frame)
        torch.cuda.synchronize()
        now = time.perf_counter()
        if last_end is not None:
            frames.append(dict(frame=frame, wall_ms=(now-last_end)*1000,
                               has_ba=int(frame % cfg['mapping']['map_every'] == 0)))
        last_end = time.perf_counter()
    slam.coslam_timing.finish_frame = finish_checked
    # Keep the comparison artifacts small; the final tensors are checked below.
    slam.save_ckpt = lambda *a, **k: None
    slam.run()
    assert all(torch.isfinite(p).all() for p in slam.model.parameters())
    assert all(torch.isfinite(p).all() for p in slam.est_c2w_data.values())
    write_csv(out / 'frames.csv', frames)
    if args.diagnostic:
        values = torch.stack(tensors).cpu().numpy()
        total_values = torch.stack(totals).cpu().numpy()
        assert len(values) == len(total_values)
        for row, value, total in zip(records, values, total_values):
            row.update(zip(('rgb_loss', 'depth_loss', 'sdf_loss', 'fs_loss', 'valid_depths'), value))
            row['total_loss'] = float(total)
        write_csv(out / 'losses.csv', records)
        write_csv(out / 'probe.csv', probes)
        bad = np.where(~np.isfinite(values).all(axis=1) | ~np.isfinite(total_values))[0]
        print('Nonfinite loss rows:', [(int(i), records[i]) for i in bad[:10]], flush=True)
        assert len(bad) == 0, 'Nonfinite training losses; inspect losses.csv'
        step_values = torch.stack(steps).cpu().numpy()
        assert np.isfinite(step_values).all()
        ba_rows = [r for r in records if r['phase'] == 'ba']
        assert len(ba_rows) == len(step_values)
        write_csv(out / 'updates.csv', [dict(frame=r['frame'], iteration=r['iteration'],
            grad_norm=float(v[0]), embedding_update_norm=float(v[1])) for r,v in zip(ba_rows, step_values)])
    summary = dict(mode=args.mode, diagnostic=args.diagnostic, seed=args.seed,
                   frames=len(slam.est_c2w_data), ba_calls=state['ba_calls'],
                   bucket_prepare_calls=state['bucket_calls'], finite_parameters_and_poses=True,
                   nonfinite_loss_rows=int(len(bad)) if args.diagnostic else None)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('PASS', summary, flush=True)


if __name__ == '__main__':
    main()
