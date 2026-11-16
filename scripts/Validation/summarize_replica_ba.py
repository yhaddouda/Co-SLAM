"""Compare final ATE and complete BA-frame wall time against paired baselines."""
import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--reevaluate', action='store_true', help='Recompute ATE from saved trajectories with official float64 alignment')
    args = parser.parse_args()
    if args.reevaluate:
        import sys
        import numpy as np
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts.Validation.rank_replica_ba_scenes import trajectory_rmse
        from tools.eval_ate import evaluate_ate
        verified = False
        for path in args.root.glob('*/*/summary.json'):
            data = np.load(path.parent / 'trajectories.npz')
            cfg = json.loads((path.parent / 'config.json').read_text())
            record = json.loads(path.read_text())
            scale = cfg['data']['sc_factor']
            for key, field in (('estimated', 'ate_rmse_m'), ('relative', 'ate_relative_rmse_m')):
                prediction = {i: torch.from_numpy(pose) for i, pose in enumerate(data[key])}
                record[field] = trajectory_rmse(data['ground_truth'], prediction) / scale
                if not verified:
                    # Verify the exact repository evaluator on a 100-frame slice;
                    # its timestamp association is quadratic on full trajectories.
                    count = min(100, len(prediction))
                    gt = {i: data['ground_truth'][i, :3, 3] for i in range(count)}
                    pred = {i: data[key][i, :3, 3] for i in range(count)}
                    official = evaluate_ate(gt, pred)['absolute_translational_error.rmse']
                    actual = trajectory_rmse(data['ground_truth'][:count], {i:prediction[i] for i in range(count)})
                    if abs(official - actual) > 1e-12:
                        raise AssertionError(f'ATE evaluator mismatch: {official} versus {actual}')
                    verified = True
            record['ate_evaluator'] = 'repository Horn alignment with float64 coordinates; verified against evaluate_ate'
            path.write_text(json.dumps(record, indent=2))
    summaries = [json.loads(p.read_text()) for p in args.root.glob('*/*/summary.json')]
    (args.root / 'summaries.json').write_text(json.dumps(summaries, indent=2))
    summaries = [r for r in summaries if not r['diagnostic']]
    baselines = {(r['scene'], r['seed']): r for r in summaries if r['variant'] == 'baseline'}
    records = []
    for r in summaries:
        baseline = baselines.get((r['scene'], r['seed']))
        if baseline is None:
            continue
        if r['frames'] != baseline['frames']:
            raise ValueError('Cannot compare different trajectory lengths')
        records.append(dict(scene=r['scene'], variant=r['variant'], seed=r['seed'], frames=r['frames'],
            ate_mm=1000 * r['ate_rmse_m'], ate_change_mm=1000 * (r['ate_rmse_m'] - baseline['ate_rmse_m']),
            ate_change_pct=100 * (r['ate_rmse_m'] / baseline['ate_rmse_m'] - 1),
            relative_ate_mm=1000 * r['ate_relative_rmse_m'],
            relative_ate_change_pct=100 * (r['ate_relative_rmse_m'] / baseline['ate_relative_rmse_m'] - 1),
            ba_mean_ms=r['ba_invocations']['mean_ms'],
            ba_reduction_pct=100 * (1 - r['ba_invocations']['mean_ms'] / baseline['ba_invocations']['mean_ms']),
            ba_frame_mean_ms=r['ba_frames']['mean_ms'],
            ba_frame_reduction_pct=100 * (1 - r['ba_frames']['mean_ms'] / baseline['ba_frames']['mean_ms']),
            tracking_frame_mean_ms=r['tracking_frames']['mean_ms'],
            network_sample_reduction_pct=r.get('network_samples', {}).get(
                'reduction_pct', None if r['variant'] in {'bucket', 'bucket_reference'} else 0.)))
    records.sort(key=lambda r: (r['scene'], r['seed'], r['variant']))
    if not records:
        raise ValueError('No complete paired results yet')
    with (args.root / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ['# Replica BA-only experiment results', '',
        'Tracking uses the baseline path in every condition. ATE is final rigid-aligned translational RMSE '
        '(no fitted scale). BA invocation wall time includes ray selection, transfers, packing/sorting, '
        'rendering, losses, backward, optimizer updates, and pose finalization. BA-frame wall time also '
        'includes data loading, tracking, and keyframe insertion. Both synchronize CUDA at their boundaries. '
        'Frame zero and frames below the configured warmup are excluded from timing. '
        'Diagnostic timings are excluded. Positive ATE change means degradation; positive time reduction means faster.', '',
        '| Scene | Variant | Seed | Frames | ATE (mm) | ATE change | BA reduction | BA-frame reduction |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['scene']} | {r['variant']} | {r['seed']} | {r['frames']} | "
                     f"{r['ate_mm']:.3f} | {r['ate_change_pct']:+.2f}% | {r['ba_reduction_pct']:+.2f}% | {r['ba_frame_reduction_pct']:+.2f}% |")
    lines += ['', '## Equal-weight means across the paired scene/seed comparisons', '',
              '| Variant | Comparisons | Mean ATE change | Worst ATE change | Mean BA reduction | Mean BA-frame reduction |',
              '|---|---:|---:|---:|---:|---:|']
    for variant in sorted({r['variant'] for r in records}):
        rows = [r for r in records if r['variant'] == variant]
        lines.append(f"| {variant} | {len(rows)} | {fmean(r['ate_change_pct'] for r in rows):+.2f}% | "
                     f"{max(r['ate_change_pct'] for r in rows):+.2f}% | {fmean(r['ba_reduction_pct'] for r in rows):+.2f}% | "
                     f"{fmean(r['ba_frame_reduction_pct'] for r in rows):+.2f}% |")
    (args.root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        variants = sorted({r['variant'] for r in records} - {'baseline'})
        colors = {variant: plt.get_cmap('tab10')(index % 10)
                  for index, variant in enumerate(sorted({r['variant'] for r in records}))}
        for ax, key, xlabel in zip(axes, ('ba_reduction_pct', 'ba_frame_reduction_pct'),
                                   ('BA invocation time reduction (%)', 'Complete BA-frame time reduction (%)')):
            ax.axhline(0, color='black', linewidth=.8)
            ax.axvline(20, color='grey', linestyle='--', linewidth=.8)
            for variant in variants:
                rows = [r for r in records if r['variant'] == variant]
                x = fmean(r[key] for r in rows)
                y = fmean(r['ate_change_pct'] for r in rows)
                ax.errorbar(x, y, xerr=[[x - min(r[key] for r in rows)], [max(r[key] for r in rows) - x]],
                            yerr=[[y - min(r['ate_change_pct'] for r in rows)], [max(r['ate_change_pct'] for r in rows) - y]],
                            fmt='o', capsize=3, label=variant, color=colors[variant])
            ax.set_xlabel(xlabel)
            ax.grid(alpha=.2)
        axes[0].set_ylabel('Final ATE change versus paired baseline (%)')
        fig.suptitle('Replica BA-only variants: means and observed scene/seed ranges')
        if variants:
            fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center',
                       ncol=min(len(variants), 4), fontsize=9)
        fig.tight_layout(rect=(0, .13, 1, .95))
        fig.savefig(args.root / 'tradeoff.png', dpi=180)
        fig.savefig(args.root / 'tradeoff.svg')
        plt.close(fig)
        scenes = sorted({r['scene'] for r in records})
        fig, axes = plt.subplots(1, len(scenes), figsize=(5 * len(scenes), 4), squeeze=False, sharey=True)
        for ax, scene in zip(axes[0], scenes):
            for variant in sorted({r['variant'] for r in records if r['scene'] == scene}):
                samples = []
                for record in summaries:
                    if record['scene'] != scene or record['variant'] != variant:
                        continue
                    path = args.root / scene / f"{variant}_seed{record['seed']}" / 'frames.csv'
                    with path.open(newline='') as handle:
                        samples.extend((int(r['frame_id']), float(r['ba_wall_ms']))
                                       for r in csv.DictReader(handle)
                                       if int(r['has_ba']) and int(r['frame_id']) >= record['warmup_frames'])
                # Average by 100-frame window; this exposes keyframe-dependent
                # finalization cost without treating windows as independent runs.
                windows = sorted({frame // 100 for frame, _ in samples})
                x = [fmean(frame for frame, _ in samples if frame // 100 == window) for window in windows]
                y = [fmean(ms for frame, ms in samples if frame // 100 == window) for window in windows]
                ax.plot(x, y, label=variant, color=colors[variant])
            ax.set_title(scene)
            ax.set_xlabel('Frame index')
            ax.grid(alpha=.2)
        axes[0][0].set_ylabel('Complete BA invocation wall time (ms)')
        axes[0][-1].legend(fontsize=8)
        fig.suptitle('BA wall time over the trajectory (100-frame window means)')
        fig.tight_layout()
        fig.savefig(args.root / 'ba_runtime_by_frame.png', dpi=180)
        fig.savefig(args.root / 'ba_runtime_by_frame.svg')
        plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
