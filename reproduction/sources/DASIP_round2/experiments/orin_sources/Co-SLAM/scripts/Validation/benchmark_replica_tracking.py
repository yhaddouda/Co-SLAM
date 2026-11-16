"""Isolated Co-SLAM tracking experiment; BA stays on the dense baseline path.

Only tracking invocations are timed as phases. Pure tracking-frame wall times
also include loading and the common harness overhead; BA-frame times are not
compared. Each condition runs in a fresh process with paired frame/phase seeds.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import config
import coslam
from optimization.tracking_experiments import VARIANTS, tracking_execution_flags
from scripts.Validation.execution_benchmark_common import (
    device_metadata, rigid_ate, seed_phase, snapshot, statistics, write_csv,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--variant', choices=sorted(VARIANTS), required=True)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'Refusing to overwrite existing output: {args.output}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = config.load_config(f'configs/Replica/{args.scene}.yaml')
    for name in ('tracking_bucket', 'tracking_naive_pruning', 'tracking_depth_spatial'):
        cfg.setdefault(name, {})['enabled'] = False
    cfg['tracking_depth_spatial'].update(collect_sample_count_stats=False, collect_fallback_stats=False, stats_only=False)
    cfg['training']['uniform_samples_until_depth'] = False
    cfg['ba_bucket'].update(enabled=False, debug_log=False, morton_inside_bucket=False)
    cfg['ba_experiment'] = dict(variant='baseline', margin=0., collect_samples=False)
    cfg['tracking_experiment'] = dict(variant=args.variant)
    cfg['grid'].update(hash='CoherentPrime', hash_size=19, morton_sort=False)
    cfg.setdefault('timing', {}).update(mode='none', max_frames=args.frames or None, disable_eval=True)
    cfg.setdefault('profiling', {})['enabled'] = False
    cfg['mapping']['first_mesh'] = False
    cfg['mesh'].update(visualisation=False, vis=1000000000)
    cfg['data'].update(output=str(args.output), exp_name='artifacts')
    (args.output / 'artifacts').mkdir()
    (args.output / 'config.json').write_text(json.dumps(cfg, indent=2))
    snapshot(args.output, ROOT, (
        'coslam.py', 'model/scene_rep.py', 'model/utils.py', 'model/keyframe.py',
        'optimization/utils.py', 'optimization/ba_experiments.py',
        'optimization/tracking_experiments.py',
        'scripts/Validation/benchmark_replica_tracking.py',
        'scripts/Validation/execution_benchmark_common.py',
    ))
    seed_phase(args.seed)
    coslam.cfg = cfg
    slam = coslam.CoSLAM(cfg)
    tracking_times, frame_records = {}, []
    state = dict(phase='', forwards=0)

    def wrap(method, phase, timed=False):
        def call(batch, frame, *inputs, **kwargs):
            seed_phase(args.seed, int(frame), phase)
            state['phase'] = 'tracking' if timed else 'mapping'
            if timed:
                torch.cuda.synchronize()
                started = time.perf_counter()
            result = method(batch, frame, *inputs, **kwargs)
            if timed:
                torch.cuda.synchronize()
                tracking_times[int(frame)] = 1000 * (time.perf_counter() - started)
            state['phase'] = ''
            return result
        return call

    slam.tracking_render = wrap(slam.tracking_render, 1, timed=True)
    # Same BA body and schedule; reseeding is common to every condition.
    slam.global_BA = wrap(slam.global_BA, 2)
    add_keyframe = slam.keyframeDatabase.add_keyframe
    def add_seeded(batch, *inputs, **kwargs):
        seed_phase(args.seed, int(batch['frame_id']), 3)
        return add_keyframe(batch, *inputs, **kwargs)
    slam.keyframeDatabase.add_keyframe = add_seeded
    if args.diagnostic:
        original_loss = slam.get_loss_from_ret
        def checked_loss(result, *inputs, **kwargs):
            loss = original_loss(result, *inputs, **kwargs)
            if state['phase'] == 'tracking':
                state['forwards'] += 1
                for key in ('rgb_loss', 'depth_loss', 'sdf_loss', 'fs_loss'):
                    if not torch.isfinite(result[key]).all():
                        raise RuntimeError(f'Nonfinite tracking {key}')
                if not torch.isfinite(loss).all():
                    raise RuntimeError('Nonfinite tracking loss')
            return loss
        slam.get_loss_from_ret = checked_loss

    finish = slam.coslam_timing.finish_frame
    last_end = None
    def finish_timed(frame):
        nonlocal last_end
        finish(frame)
        torch.cuda.synchronize()
        ended = time.perf_counter()
        if last_end is not None:
            frame_records.append(dict(frame_id=int(frame), wall_ms=1000 * (ended - last_end),
                has_mapping=int(frame % cfg['mapping']['map_every'] == 0),
                tracking_wall_ms=tracking_times.get(int(frame), 0.)))
        last_end = time.perf_counter()
        if frame % 200 == 0:
            print(f'BENCHMARK_PROGRESS system=coslam scene={args.scene} variant={args.variant} frame={frame}', flush=True)
    slam.coslam_timing.finish_frame = finish_timed
    slam.save_ckpt = lambda *inputs, **kwargs: None
    started = time.perf_counter()
    slam.run()
    elapsed = time.perf_counter() - started
    assert all(torch.isfinite(p).all() for p in slam.model.parameters())
    keys = sorted(slam.est_c2w_data)
    assert keys == list(range(args.frames or 2000)), keys[-1]
    ground_truth = np.stack([slam.pose_gt[i].cpu().numpy() for i in keys])
    estimated = np.stack([slam.est_c2w_data[i].detach().cpu().numpy() for i in keys])
    relative_poses = slam.convert_relative_pose()
    relative = np.stack([relative_poses[i].detach().cpu().numpy() for i in keys])
    assert np.isfinite(estimated).all() and np.isfinite(relative).all()
    scale = cfg['data']['sc_factor']
    np.savez(args.output / 'trajectories.npz', ground_truth=ground_truth, estimated=estimated, relative=relative)
    write_csv(args.output / 'frames.csv', frame_records)
    measured = [r for r in frame_records if r['frame_id'] >= args.warmup]
    summary = dict(system='coslam', scene=args.scene, variant=args.variant, seed=args.seed,
        frames=len(keys), warmup_frames=args.warmup, diagnostic=args.diagnostic,
        tracking_variant=args.variant, mapping_variant='baseline',
        execution_flags=tracking_execution_flags(cfg),
        ate_rmse_m=rigid_ate(ground_truth, estimated) / scale,
        ate_relative_rmse_m=rigid_ate(ground_truth, relative) / scale,
        ate_evaluator='float64 Horn rigid alignment; no scale fitting',
        tracking_invocations=statistics([r['tracking_wall_ms'] for r in measured]),
        tracking_frames=statistics([r['wall_ms'] for r in measured if not r['has_mapping']]),
        run_wall_seconds=elapsed, finite_parameters_and_poses=True,
        checked_forwards=state['forwards'], rng_policy='paired frame/phase seeds',
        timing_policy='synchronized tracking calls and pure tracking frames; BA not timed',
        **device_metadata())
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('BENCHMARK_RESULT ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
