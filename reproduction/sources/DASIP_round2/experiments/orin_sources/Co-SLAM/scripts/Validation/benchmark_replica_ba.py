"""BA-only Replica experiment, final aligned ATE and synchronized wall timings.

Run each condition in a fresh process. Tracking and keyframe sampling have
frame-specific seeds shared by every condition, preventing BA RNG consumption
from changing subsequent tracking ray batches. Timed BA invocations include
all preprocessing; BA-frame wall time also includes tracking and data loading.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import config
import coslam
from scripts.Validation.rank_replica_ba_scenes import trajectory_rmse
from optimization.ba_experiments import ba_execution_flags

VARIANTS = ('baseline', 'bucket', 'bucket_dense', 'bucket_reference', 'lattice', 'lattice_guarded',
            'lattice_guarded_preload', 'baseline_fast', 'lattice_guarded_fast')


def write_csv(path, records):
    if not records:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def statistics(values):
    x = np.asarray(values, dtype=float)
    return dict(count=len(x), mean_ms=float(x.mean()), median_ms=float(np.median(x)),
                p95_ms=float(np.percentile(x, 95)), std_ms=float(x.std())) if len(x) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--label', help='Ablation name in summaries; the variant still selects the sampled objective')
    parser.add_argument('--preload-rays', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--fast-pose-commit', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--frames', type=int, default=0, help='0 = full 2,000-frame scene')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--margin', type=float, default=0.0, help='Extra distance beyond depth + truncation')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true', help='Check every loss; do not compare diagnostic timings')
    parser.add_argument('--ba-breakdown', action='store_true',
                        help='Record BA block CUDA intervals/host submission times; separate from headline wall runs')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'Refusing to overwrite experiment directory: {args.output}')
    if not torch.cuda.is_available():
        raise RuntimeError('Replica benchmarking requires the host CUDA device')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = config.load_config(f'configs/Replica/{args.scene}.yaml')
    for name in ('tracking_bucket', 'tracking_naive_pruning', 'tracking_depth_spatial'):
        cfg.setdefault(name, {})['enabled'] = False
    cfg['tracking_depth_spatial'].update(collect_sample_count_stats=False, collect_fallback_stats=False, stats_only=False)
    cfg.setdefault('tracking_experiment', {})['variant'] = 'baseline'
    cfg['training']['uniform_samples_until_depth'] = False
    cfg['ba_bucket']['enabled'] = args.variant in ('bucket', 'bucket_dense', 'bucket_reference')
    cfg['ba_bucket'].update(debug_log=False, morton_inside_bucket=False)
    cfg['ba_experiment'] = dict(variant=args.variant, margin=args.margin, collect_samples=True,
                               preload_rays=args.preload_rays, fast_pose_commit=args.fast_pose_commit)
    execution_flags = ba_execution_flags(cfg)
    cfg['grid'].update(hash='CoherentPrime', hash_size=19, morton_sort=False)
    cfg.setdefault('timing', {}).update(mode='none', max_frames=args.frames or None, disable_eval=True)
    if args.ba_breakdown:
        cfg['timing'].update(mode='ba_breakdown', warmup_frames=args.warmup,
                             ba_output_csv=str(args.output / 'ba_breakdown.csv'))
    cfg.setdefault('profiling', {})['enabled'] = False
    cfg['mapping']['first_mesh'] = False
    cfg['mesh'].update(visualisation=False, vis=1000000000)
    cfg['data'].update(output=str(args.output), exp_name='artifacts')
    (args.output / 'artifacts').mkdir()
    (args.output / 'config.json').write_text(json.dumps(cfg, indent=2))
    source_dir = args.output / 'source'
    source_dir.mkdir()
    hashes = {}
    for name in ('coslam.py', 'model/scene_rep.py', 'model/utils.py', 'optimization/ba_experiments.py',
                 'optimization/utils.py', 'optimization/tracking_experiments.py', 'model/keyframe.py',
                 'model/layer_timing.py', 'scripts/Validation/benchmark_replica_ba.py',
                 'scripts/Validation/rank_replica_ba_scenes.py'):
        path = Path(name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (args.output / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    coslam.cfg = cfg
    slam = coslam.CoSLAM(cfg)
    frames, ba_times = [], {}
    state = dict(phase='', frame=0, checked_forwards=0)

    def phase_seed(frame, phase):
        seed = args.seed * 1000003 + frame * 17 + phase
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def wrap_phase(method, phase, name):
        def wrapped(batch, frame, *inputs, **kwargs):
            phase_seed(frame, phase)
            state.update(phase=name, frame=int(frame))
            if name == 'ba':
                torch.cuda.synchronize()
                start = time.perf_counter()
            result = method(batch, frame, *inputs, **kwargs)
            if name == 'ba':
                torch.cuda.synchronize()
                ba_times[int(frame)] = (time.perf_counter() - start) * 1000
            state['phase'] = ''
            return result
        return wrapped

    slam.tracking_render = wrap_phase(slam.tracking_render, 1, 'tracking')
    slam.global_BA = wrap_phase(slam.global_BA, 2, 'ba')
    add_keyframe = slam.keyframeDatabase.add_keyframe
    def add_seeded(batch, *inputs, **kwargs):
        phase_seed(int(batch['frame_id']), 3)
        return add_keyframe(batch, *inputs, **kwargs)
    slam.keyframeDatabase.add_keyframe = add_seeded
    if args.diagnostic:
        original_loss = slam.get_loss_from_ret
        def checked_loss(ret, *inputs, **kwargs):
            state['checked_forwards'] += 1
            for key in ('rgb_loss', 'depth_loss', 'sdf_loss', 'fs_loss'):
                if not torch.isfinite(ret[key]).all():
                    raise RuntimeError(f'Nonfinite {key} in {state}')
            result = original_loss(ret, *inputs, **kwargs)
            if not torch.isfinite(result).all():
                raise RuntimeError(f'Nonfinite total loss in {state}')
            return result
        slam.get_loss_from_ret = checked_loss

    finish = slam.coslam_timing.finish_frame
    last_end = None
    def finish_timed(frame):
        nonlocal last_end
        finish(frame)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if last_end is not None:
            frames.append(dict(frame_id=int(frame), wall_ms=(end - last_end) * 1000,
                               has_ba=int(frame in ba_times), ba_wall_ms=ba_times.get(frame, 0.)))
        last_end = time.perf_counter()
        if frame % 200 == 0:
            print(f'BENCHMARK_PROGRESS scene={args.scene} variant={args.variant} frame={frame}', flush=True)
    slam.coslam_timing.finish_frame = finish_timed
    # Trajectories suffice for this ATE experiment; no mesh extraction or large map checkpoint.
    slam.save_ckpt = lambda *inputs, **kwargs: None
    start = time.perf_counter()
    slam.run()
    elapsed = time.perf_counter() - start
    assert all(torch.isfinite(p).all() for p in slam.model.parameters())
    assert all(torch.isfinite(p).all() for p in slam.est_c2w_data.values())
    relative = slam.convert_relative_pose()
    gt = np.stack([slam.pose_gt[i].cpu().numpy() for i in sorted(slam.est_c2w_data)])
    scale = cfg['data']['sc_factor']
    ate = trajectory_rmse(gt, slam.est_c2w_data) / scale
    ate_relative = trajectory_rmse(gt, relative) / scale
    np.savez(args.output / 'trajectories.npz', ground_truth=gt,
             estimated=np.stack([p.detach().cpu().numpy() for p in slam.est_c2w_data.values()]),
             relative=np.stack([relative[i].detach().cpu().numpy() for i in sorted(relative)]))
    write_csv(args.output / 'frames.csv', frames)
    measured = [r for r in frames if r['frame_id'] >= args.warmup]
    summary = dict(scene=args.scene, variant=args.label or args.variant, ba_variant=args.variant,
        ba_execution=execution_flags, seed=args.seed, margin=args.margin,
        frames=len(slam.est_c2w_data), diagnostic=args.diagnostic, tracking='baseline',
        ate_rmse_m=ate, ate_relative_rmse_m=ate_relative, warmup_frames=args.warmup,
        ba_invocations=statistics([r['ba_wall_ms'] for r in measured if r['has_ba']]),
        ba_frames=statistics([r['wall_ms'] for r in measured if r['has_ba']]),
        tracking_frames=statistics([r['wall_ms'] for r in measured if not r['has_ba']]),
        run_wall_seconds=elapsed, finite_parameters_and_poses=True,
        checked_forwards=state['checked_forwards'], rng_policy='paired frame/phase seeds',
        timing_mode='ba_breakdown' if args.ba_breakdown else 'none',
        device=torch.cuda.get_device_name(0), torch_version=torch.__version__)
    sample_counts = getattr(slam.model, 'ba_experiment_sample_counts', [])
    if sample_counts:
        summary['network_samples'] = {key: sum(r[key] for r in sample_counts) for key in sample_counts[0]}
        summary['network_samples']['reduction_pct'] = 100 * (1 - summary['network_samples']['queried_samples'] / summary['network_samples']['dense_samples'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('BENCHMARK_RESULT ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
