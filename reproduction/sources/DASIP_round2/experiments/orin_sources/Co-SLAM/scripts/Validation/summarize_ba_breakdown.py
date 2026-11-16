#!/usr/bin/env python3
"""Summarize saved BA wall runs and block profiles without double counting."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

RAY_STAGES = ('BA_RAY_PRELOAD', 'prepare_ba_buckets', 'sample_global_rays_BA', 'cpu_to_gpu_BA')
SETUP_STAGES = ('stack_KF_poses_BA', 'which_poses_to_opt_BA', 'setup_optimizer_BA', 'cat&reshape_BA')
LOOP_STAGES = ('resize_tensors_BA', 'Forward_BA', 'Loss_calculation_BA', 'Backward_BA', 'Optimize_map_pose_BA')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile_frames(path):
    """Validate complete invocations; add only disjoint 'component' rows."""
    grouped = defaultdict(list)
    with path.open() as stream:
        for r in csv.DictReader(stream):
            for key in ('cuda_elapsed_ms', 'host_submit_ms'):
                r[key] = float(r[key])
                if not math.isfinite(r[key]) or r[key] < 0:
                    raise ValueError(f'Invalid {key} in {path}')
            grouped[int(r['frame_id'])].append(r)
    result, stage_rows = {}, []
    for frame, rows in sorted(grouped.items()):
        totals = [r for r in rows if r['stage'] == 'BA_TOTAL']
        if len(totals) != 1:
            raise ValueError(f'Expected one BA_TOTAL at frame {frame} in {path}')
        total = totals[0]
        iterations = sorted(int(r['iteration_id']) for r in rows if r['stage'] == 'BA_ITER_PROFILE_TOTAL')
        if iterations != list(range(int(total['iterations']))):
            raise ValueError(f'Incomplete iteration profile at frame {frame} in {path}')
        metadata = ('variant', 'num_keyframes', 'optimized_poses', 'iterations')
        if any(any(r[k] != total[k] for k in metadata) for r in rows):
            raise ValueError(f'Inconsistent metadata at frame {frame}')
        stages = defaultdict(lambda: [0., 0., 0])
        components = {}
        for r in rows:
            if r['scope'] not in ('inclusive', 'component', 'detail'):
                raise ValueError(f'Unknown timing scope in {path}')
            key = (r['stage'], r['scope'])
            stages[key][0] += r['cuda_elapsed_ms']
            stages[key][1] += r['host_submit_ms']
            stages[key][2] += 1
            if r['scope'] == 'component':
                components[r['stage']] = components.get(r['stage'], 0.) + r['cuda_elapsed_ms']
        accounted = sum(components.values())
        residual = total['cuda_elapsed_ms'] - accounted
        if residual < -0.1:
            raise ValueError(f'Overlapping component accounting at frame {frame}: {residual} ms')
        result[frame] = dict(
            cuda_total_ms=total['cuda_elapsed_ms'],
            host_total_submit_ms=total['host_submit_ms'],
            ray_prepare_cuda_ms=sum(components.get(s, 0.) for s in RAY_STAGES),
            pose_commit_cuda_ms=components.get('BA_POSE_COMMIT', 0.),
            setup_cuda_ms=sum(components.get(s, 0.) for s in SETUP_STAGES),
            loop_compute_cuda_ms=sum(components.get(s, 0.) for s in LOOP_STAGES),
            unattributed_cuda_ms=residual,
            num_keyframes=int(total['num_keyframes']), optimized_poses=int(total['optimized_poses']),
        )
        for (stage, scope), (cuda, host, occurrences) in stages.items():
            stage_rows.append(dict(frame_id=frame, stage=stage, scope=scope,
                                   cuda_elapsed_ms=cuda, host_submit_ms=host, occurrences=occurrences))
    if not result:
        raise ValueError(f'Empty BA profile: {path}')
    return result, stage_rows


def windows(frames, start_frame, window):
    eligible = sorted(r['frame_id'] for r in frames if r['frame_id'] >= start_frame)
    if not eligible:
        raise ValueError('No BA frames at/after --start-frame; lower it for smoke runs.')
    n = min(window, len(eligible) // 2)
    if n < 1:
        raise ValueError('Need at least two BA frames for disjoint early/late windows.')
    return {'early': set(eligible[:n]), 'late': set(eligible[-n:])}


def paired_windows(rows):
    grouped = defaultdict(dict)
    for r in rows:
        grouped[(r['scene'], r['seed'], r['repeat'], r['mode'], r['window'])][r['variant']] = r
    comparisons = []
    metrics = ('ba_wall_ms', 'ba_frame_wall_ms', 'cuda_total_ms', 'ray_prepare_cuda_ms',
               'pose_commit_cuda_ms', 'setup_cuda_ms', 'loop_compute_cuda_ms', 'unattributed_cuda_ms')
    for key, variants in sorted(grouped.items()):
        if not {'baseline', 'baseline_fast'} <= variants.keys():
            continue
        b, f = variants['baseline'], variants['baseline_fast']
        for name in ('first_frame', 'last_frame', 'calls'):
            if b[name] != f[name]:
                raise ValueError(f'Unmatched paired window: {key}')
        row = dict(scene=key[0], seed=key[1], repeat=key[2], mode=key[3], window=key[4],
                   first_frame=b['first_frame'], last_frame=b['last_frame'], calls=b['calls'],
                   baseline_ate_mm=b['ate_mm'], fast_ate_mm=f['ate_mm'],
                   ate_change_pct=100 * (f['ate_mm'] / b['ate_mm'] - 1) if b['ate_mm'] else '')
        for name in metrics:
            row['baseline_' + name], row['fast_' + name] = b[name], f[name]
            row['saved_' + name] = b[name] - f[name] if b[name] != '' and f[name] != '' else ''
        row['ba_wall_reduction_pct'] = 100 * (1 - f['ba_wall_ms'] / b['ba_wall_ms']) if b['ba_wall_ms'] else ''
        comparisons.append(row)
    return comparisons


def summarize(root, start_frame=100, window=50, plot=False):
    all_frames, all_stages, window_rows, report_rows = [], [], [], []
    run_frame_ids = {}
    for path in sorted(root.rglob('summary.json')):
        summary = json.loads(path.read_text())
        if 'ba_invocations' not in summary:
            continue
        d = path.parent
        mode = 'breakdown' if summary.get('timing_mode') == 'ba_breakdown' else 'wall'
        if summary.get('diagnostic'):
            raise ValueError(f'Diagnostic run is not comparable: {d}')
        repeat = json.loads((d / 'run.json').read_text()).get('repeat', 0) if (d / 'run.json').exists() else 0
        meta = dict(scene=summary['scene'], variant=summary['variant'], seed=summary['seed'],
                    repeat=repeat, mode=mode, run=str(d.relative_to(root)), ate_mm=1000 * summary['ate_rmse_m'])
        with (d / 'frames.csv').open() as stream:
            frames = [dict(frame_id=int(r['frame_id']), ba_wall_ms=float(r['ba_wall_ms']),
                           ba_frame_wall_ms=float(r['wall_ms']))
                      for r in csv.DictReader(stream) if int(r['has_ba'])
                      and int(r['frame_id']) >= summary['warmup_frames']]
        if not frames:
            raise ValueError(f'No measured BA frames in {d}')
        profile = {}
        if mode == 'breakdown':
            profile, stage_rows = profile_frames(d / 'ba_breakdown.csv')
            if set(profile) != {r['frame_id'] for r in frames}:
                raise ValueError(f'Profile and wall frame IDs disagree: {d}')
            all_stages.extend(dict(**meta, **r) for r in stage_rows)
        run_frame_ids[meta['run']] = tuple(r['frame_id'] for r in frames)
        selected = windows(frames, start_frame, window)
        metric_names = ('cuda_total_ms', 'host_total_submit_ms', 'ray_prepare_cuda_ms', 'pose_commit_cuda_ms',
                        'setup_cuda_ms', 'loop_compute_cuda_ms', 'unattributed_cuda_ms', 'num_keyframes', 'optimized_poses')
        for r in frames:
            r.update(profile.get(r['frame_id'], {k: '' for k in metric_names}))
            all_frames.append(dict(**meta, **r))
        for label, ids in selected.items():
            chosen = [r for r in frames if r['frame_id'] in ids]
            stats = dict(**meta, window=label, first_frame=min(ids), last_frame=max(ids), calls=len(ids))
            for key in ('ba_wall_ms', 'ba_frame_wall_ms', *metric_names):
                vals = [float(r[key]) for r in chosen if r[key] != '']
                stats[key] = statistics.mean(vals) if vals else ''
            window_rows.append(stats)
        report_rows.append((meta, selected))
    if not all_frames:
        raise ValueError(f'No completed BA runs under {root}')
    # Refuse comparisons of unequal trajectory lengths/windows.
    matched = defaultdict(list)
    for row in all_frames:
        matched[(row['scene'], row['seed'], row['repeat'])].append(row['run'])
    for key, runs in matched.items():
        if len({run_frame_ids[r] for r in set(runs)}) != 1:
            raise ValueError(f'Unmatched frame horizons in {key}; summarize matching lengths separately.')
    write_csv(root / 'ba_blocks_by_frame.csv', all_frames)
    write_csv(root / 'ba_stages_by_frame.csv', all_stages)
    write_csv(root / 'ba_windows.csv', window_rows)
    comparisons = paired_windows(window_rows)
    write_csv(root / 'ba_window_comparisons.csv', comparisons)
    lines = ['# Dense BA timing report', '',
             'Wall passes are the headline timing runs. Breakdown passes are instrumented explanations; '
             'their wall times include event collection, synchronization and CSV flush overhead.', '',
             'CUDA-event intervals are stream elapsed time and may include host-induced idle gaps. '
             'Host submission time includes implicit waits and is not completed GPU time. Never add host and CUDA times.', '',
             'Ray preparation combines preload, per-iteration selection and transfer, so moved work is counted once. '
             'Only disjoint component rows enter the partition; inclusive totals and detail children are excluded. '
             'The residual includes uncovered execution/annotation gaps.', '',
             '| Scene | Variant | Seed/repeat | Pass | Window | Frames | Calls | BA wall ms | CUDA total ms | Ray prep ms | Pose commit ms | Residual ms | ATE mm |',
             '|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in window_rows:
        fmt = lambda k: f'{r[k]:.3f}' if isinstance(r[k], (int, float)) else '—'
        lines.append(f"| {r['scene']} | {r['variant']} | {r['seed']}/{r['repeat']} | {r['mode']} | {r['window']} | "
                     f"{r['first_frame']}–{r['last_frame']} | {r['calls']} | {fmt('ba_wall_ms')} | {fmt('cuda_total_ms')} | "
                     f"{fmt('ray_prepare_cuda_ms')} | {fmt('pose_commit_cuda_ms')} | {fmt('unattributed_cuda_ms')} | {fmt('ate_mm')} |")
    lines += ['', '| Scene | Seed/repeat | Pass | Window | BA wall saved % | CUDA total saved ms | Ray preparation saved ms | Pose commit saved ms |',
              '|---|---|---|---|---:|---:|---:|---:|']
    for r in comparisons:
        fmt = lambda k: f'{r[k]:.3f}' if isinstance(r[k], (int, float)) else '—'
        lines.append(f"| {r['scene']} | {r['seed']}/{r['repeat']} | {r['mode']} | {r['window']} | "
                     f"{fmt('ba_wall_reduction_pct')} | {fmt('saved_cuda_total_ms')} | "
                     f"{fmt('saved_ray_prepare_cuda_ms')} | {fmt('saved_pose_commit_cuda_ms')} |")
    lines += ['', 'Early/late windows are disjoint. If fewer calls are available, both windows shrink equally. '
              'The exact ranges/counts are in the table; use full runs for long-sequence claims.', '',
              'Detailed nested block CUDA and host timings: `ba_stages_by_frame.csv`. '
              'Per-invocation partition and keyframe counts: `ba_blocks_by_frame.csv`. '
              'Window aggregates: `ba_windows.csv`. Paired baseline-minus-fast differences: '
              '`ba_window_comparisons.csv`. Positive saved values mean less time in fast; all ATE values are final-trajectory ATE.', '',
              'Baseline and fast profiles localize observed time changes. They do not alone establish independent '
              'causal effects, lack of interactions, statistical equivalence of ATE, or transfer to other hardware.']
    (root / 'BA_BREAKDOWN_REPORT.md').write_text('\n'.join(lines) + '\n')
    if plot:
        plot_frames(root, all_frames)
    return window_rows


def plot_frames(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for scene in sorted({r['scene'] for r in rows}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
        for run in sorted({r['run'] for r in rows if r['scene'] == scene and r['mode'] == 'breakdown'}):
            frames = sorted((r for r in rows if r['run'] == run), key=lambda r: r['frame_id'])
            label = f"{frames[0]['variant']} s{frames[0]['seed']} r{frames[0]['repeat']}"
            for ax, metric, title in zip(axes, ('cuda_total_ms', 'ray_prepare_cuda_ms', 'pose_commit_cuda_ms'),
                                          ('Complete BA stream interval', 'Ray preparation incl. preload', 'Final pose commit')):
                ax.plot([r['frame_id'] for r in frames], [r[metric] for r in frames], label=label, lw=1)
                ax.set(title=title, xlabel='Frame index', ylabel='CUDA-event elapsed ms')
                ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(scene + ': instrumented passes')
        fig.savefig(root / f'{scene}_ba_block_curves.png', dpi=160)
        fig.savefig(root / f'{scene}_ba_block_curves.pdf')
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        for run in sorted({r['run'] for r in rows if r['scene'] == scene and r['mode'] == 'wall'}):
            frames = sorted((r for r in rows if r['run'] == run), key=lambda r: r['frame_id'])
            label = f"{frames[0]['variant']} s{frames[0]['seed']} r{frames[0]['repeat']}"
            groups = defaultdict(list)
            for frame in frames:
                groups[frame['frame_id'] // 100].append(frame)
            for ax, metric, title in zip(axes, ('ba_wall_ms', 'ba_frame_wall_ms'),
                                         ('Complete BA invocation', 'Complete BA frame')):
                x = [statistics.mean(f['frame_id'] for f in batch) for _, batch in sorted(groups.items())]
                y = [statistics.mean(f[metric] for f in batch) for _, batch in sorted(groups.items())]
                ax.plot(x, y, label=label, lw=1.6)
                ax.set(title=title, xlabel='Frame index', ylabel='Synchronized wall time (ms)')
                ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(scene + ': normal wall passes, means in 100-frame windows')
        fig.savefig(root / f'{scene}_ba_wall_curves.png', dpi=180)
        fig.savefig(root / f'{scene}_ba_wall_curves.pdf')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--start-frame', type=int, default=100)
    parser.add_argument('--window', type=int, default=50, help='BA calls per early/late window')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    if args.start_frame < 0 or args.window < 1:
        parser.error('start-frame must be nonnegative and window must be positive')
    summarize(args.root, args.start_frame, args.window, args.plot)
    print(args.root / 'BA_BREAKDOWN_REPORT.md')


if __name__ == '__main__':
    main()
