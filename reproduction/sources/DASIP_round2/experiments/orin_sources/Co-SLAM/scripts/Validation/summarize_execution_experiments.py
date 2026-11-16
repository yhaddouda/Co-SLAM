"""Compare saved execution experiments with paired baselines; optionally plot."""
import argparse
import csv
import json
import os
from pathlib import Path
from statistics import mean


def reduction(value, reference):
    return 100 * (1 - value / reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    summaries = [json.loads(path.read_text()) for path in args.root.glob('*/*/summary.json')]
    baselines = {(s['system'], s['scene'], s['seed'], s['frames']): s
                 for s in summaries if s['variant'] == 'baseline'}
    records, missing = [], []
    for summary in sorted(summaries, key=lambda s: (s['scene'], s['seed'], s['variant'])):
        key = summary['system'], summary['scene'], summary['seed'], summary['frames']
        if key not in baselines:
            missing.append((summary['scene'], summary['variant'], summary['seed']))
            continue
        baseline = baselines[key]
        if summary['diagnostic'] or baseline['diagnostic']:
            raise ValueError('Diagnostic timings must not be compared')
        if summary['warmup_frames'] != baseline['warmup_frames']:
            raise ValueError('Mismatched timing warmup')
        row = {k:summary[k] for k in ('system','scene','variant','seed','frames','warmup_frames',
                                      'tracking_variant','mapping_variant')}
        row.update(ate_mm=1000 * summary['ate_rmse_m'],
                   ate_change_mm=1000 * (summary['ate_rmse_m'] - baseline['ate_rmse_m']),
                   ate_change_pct=100 * (summary['ate_rmse_m'] / baseline['ate_rmse_m'] - 1),
                   relative_ate_mm=1000 * summary['ate_relative_rmse_m'],
                   run_wall_seconds=summary['run_wall_seconds'],
                   run_wall_reduction_pct=reduction(summary['run_wall_seconds'], baseline['run_wall_seconds']))
        for phase in ('tracking_invocations','tracking_frames','mapping_invocations','mapping_frames'):
            values = summary.get(phase)
            reference = baseline.get(phase)
            for metric in ('count','mean_ms','median_ms','p95_ms','std_ms'):
                row[phase + '_' + metric] = values[metric] if values else ''
            row[phase + '_reduction_pct'] = reduction(values['mean_ms'], reference['mean_ms']) if values else ''
        records.append(row)
    if not records:
        print('No completed paired comparisons yet.')
        return
    with (args.root / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.root / 'summaries.json').write_text(json.dumps(summaries, indent=2))
    system = records[0]['system']
    frames = sorted({r['frames'] for r in records})
    lines = [f'# {system} execution experiments ({frames} frames)', '',
             ('Status: complete suite.' if (args.root / 'SUITE_COMPLETE').exists()
              else 'Status: partial; only completed paired runs appear below.'), '',
             'Positive ATE change means degradation; positive time reduction means faster. '
             'ATE is the final raw trajectory, rigid-aligned in float64 without scale fitting. '
             'Comparisons pair the same scene/seed/trajectory length. Means below average relative '
             'changes per paired scene/seed, rather than pooling time. Prefix runs are screening evidence.', '',
             '| Scene | Variant | Seed | ATE mm | ATE change | Tracking call saved | Tracking frame saved | Mapping call saved | Mapping frame saved |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def percent(value):
        return f'{value:+.2f}%' if value != '' else '—'
    for row in records:
        lines.append(f"| {row['scene']} | {row['variant']} | {row['seed']} | {row['ate_mm']:.3f} | "
                     f"{percent(row['ate_change_pct'])} | {percent(row['tracking_invocations_reduction_pct'])} | "
                     f"{percent(row['tracking_frames_reduction_pct'])} | {percent(row['mapping_invocations_reduction_pct'])} | "
                     f"{percent(row['mapping_frames_reduction_pct'])} |")
    lines += ['', '| Variant | Pairs | Mean ATE change | Worst ATE change | Tracking call saved | Tracking frame saved | Mapping call saved | Mapping frame saved |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for variant in sorted({r['variant'] for r in records}):
        selected = [r for r in records if r['variant'] == variant]
        means = {key:mean(r[key] for r in selected) if selected[0][key] != '' else ''
                 for key in ('ate_change_pct','tracking_invocations_reduction_pct','tracking_frames_reduction_pct',
                             'mapping_invocations_reduction_pct','mapping_frames_reduction_pct')}
        lines.append(f"| {variant} | {len(selected)} | {percent(means['ate_change_pct'])} | "
                     f"{percent(max(r['ate_change_pct'] for r in selected))} | "
                     f"{percent(means['tracking_invocations_reduction_pct'])} | {percent(means['tracking_frames_reduction_pct'])} | "
                     f"{percent(means['mapping_invocations_reduction_pct'])} | {percent(means['mapping_frames_reduction_pct'])} |")
    lines += ['', 'Absolute tracking timings after warmup:', '',
              '| Scene | Variant | Call samples | Call mean ms | Call median ms | Call p95 ms | Frame samples | Frame mean ms | Frame p95 ms |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in records:
        lines.append(f"| {row['scene']} | {row['variant']} | {row['tracking_invocations_count']} | "
                     f"{row['tracking_invocations_mean_ms']:.2f} | {row['tracking_invocations_median_ms']:.2f} | "
                     f"{row['tracking_invocations_p95_ms']:.2f} | {row['tracking_frames_count']} | "
                     f"{row['tracking_frames_mean_ms']:.2f} | {row['tracking_frames_p95_ms']:.2f} |")
    if system == 'ec_slam':
        lines += ['', 'Absolute mapping timings after warmup:', '',
                  '| Scene | Variant | Call samples | Call mean ms | Call median ms | Call p95 ms | Frame mean ms | Frame p95 ms |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for row in records:
            lines.append(f"| {row['scene']} | {row['variant']} | {row['mapping_invocations_count']} | "
                         f"{row['mapping_invocations_mean_ms']:.2f} | {row['mapping_invocations_median_ms']:.2f} | "
                         f"{row['mapping_invocations_p95_ms']:.2f} | {row['mapping_frames_mean_ms']:.2f} | "
                         f"{row['mapping_frames_p95_ms']:.2f} |")
    lines += ['', 'Timing boundaries:', '',
              '- Co-SLAM: complete tracking calls and pure tracking frames; BA stays baseline and is not timed. '
              'Pure tracking frames include data loading and common harness overhead.',
              '- EC-SLAM: complete optimize calls; tracking frame work includes transfers/init/storage but excludes loader and mapping wait; '
              'mapping frame work includes loading/keyframe insertion. All conditions wait before next-frame copies, '
              'and timers end before worker-state publication to prevent cross-worker contamination.',
              '- CUDA synchronized at timer boundaries; frame/phase seeds paired; evaluation, mesh extraction and active profilers excluded. '
              'Profiler traces are separate diagnostic runs and are never headline timing comparisons.',
              '- In EC tracking-only or mapping-only conditions, time changes in the untouched stage are retained as context. '
              'They cannot be attributed to an execution change in that stage; evolving trajectories, field states and run variability can affect them.',
              '- These measurements are on the saved device and seed(s). ATE/depth/reconstruction statistical equivalence '
              'and RTX 4070 generalization are not established by a single-seed Orin experiment.']
    if missing:
        lines += ['', f'Incomplete pairs without a completed baseline: {missing}.']
    (args.root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if args.plot and any(r['variant'] != 'baseline' for r in records):
        os.environ.setdefault('MPLCONFIGDIR', '/tmp/coslam-execution-plots')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.size':9, 'pdf.fonttype':42})
        variants = sorted({r['variant'] for r in records if r['variant'] != 'baseline'})
        phases = ['tracking_invocations_reduction_pct']
        if system == 'ec_slam':
            phases.append('mapping_invocations_reduction_pct')
        fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 3.8), squeeze=False,
                                 constrained_layout=True)
        for ax, phase in zip(axes[0], phases):
            for variant in variants:
                selected = [r for r in records if r['variant'] == variant]
                ax.scatter([r[phase] for r in selected], [r['ate_change_pct'] for r in selected], label=variant, s=35)
            ax.axhline(0,color='gray',lw=.8); ax.axvline(0,color='gray',lw=.8)
            ax.set_xlabel(phase.replace('_invocations_reduction_pct','') + ' call time reduction (%)')
            ax.set_ylabel('Final ATE change (%)')
            ax.grid(alpha=.2)
        axes[0,-1].legend(fontsize=8)
        fig.savefig(args.root / 'tradeoff.png', dpi=180)
        fig.savefig(args.root / 'tradeoff.svg')
        plt.close(fig)
    print(f'Wrote {len(records)} paired comparisons to {args.root}')


if __name__ == '__main__':
    main()
