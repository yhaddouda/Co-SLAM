"""Summarize saved timing trends, baseline repeats and read-only telemetry."""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean


def timing_windows(root, width=400):
    rows = []
    summaries = {p.parent: json.loads(p.read_text()) for p in root.glob('*/*/summary.json')}
    baselines = {(s['scene'], s['seed'], s['frames']): path
                 for path, s in summaries.items() if s['variant'] == 'baseline'}
    windows = {}
    for path, summary in summaries.items():
        assert not summary['diagnostic'], path
        grouped = defaultdict(list)
        with (path / 'frames.csv').open() as stream:
            for record in csv.DictReader(stream):
                frame = int(record['frame_id'])
                if frame < summary['warmup_frames']:
                    continue
                start = frame // width * width
                grouped[start, 'tracking_call'].append(float(record['tracking_wall_ms']))
                if summary['system'] == 'ec_slam' and int(record['has_mapping']):
                    grouped[start, 'mapping_call'].append(float(record['mapping_wall_ms']))
        windows[path] = {key: (len(values), mean(values)) for key, values in grouped.items()}
    for path, summary in sorted(summaries.items()):
        reference = baselines.get((summary['scene'], summary['seed'], summary['frames']))
        if reference is None:
            continue
        for (start, phase), (count, value) in sorted(windows[path].items()):
            base_count, base_value = windows[reference][start, phase]
            assert count == base_count, (path, start, phase)
            rows.append(dict(system=summary['system'], scene=summary['scene'],
                variant=summary['variant'], seed=summary['seed'], phase=phase,
                first_frame=max(start, summary['warmup_frames']),
                last_frame=min(start + width, summary['frames']) - 1,
                samples=count, mean_ms=value, baseline_mean_ms=base_value,
                reduction_pct=100 * (1 - value / base_value)))
    return rows


def telemetry(paths):
    groups = defaultdict(lambda: defaultdict(list))
    bounds = {}
    samples = 0
    for path in paths:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                samples += 1
                label = row['active_jobs']
                for group in ('all', label):
                    if group not in bounds:
                        bounds[group] = [row['utc'], row['utc'], 0]
                    bounds[group][1] = row['utc']
                    bounds[group][2] += 1
                    for field, value in row.items():
                        if field.startswith('/sys/') and value:
                            try:
                                groups[group][field].append(int(value))
                            except ValueError:
                                pass
    result = {}
    for group, fields in groups.items():
        result[group] = dict(first_utc=bounds[group][0], last_utc=bounds[group][1],
            samples=bounds[group][2], sensors={field:dict(samples=len(values),
                min=min(values), mean=mean(values), max=max(values))
                for field, values in fields.items()})
    return dict(files=list(map(str, paths)), samples=samples, groups=result,
        scope='Active labels come from the latest driver START line. Repeated labels '
              'can cover separate runs; this is telemetry coverage, not exact phase attribution. '
              'Missing sensors are omitted. Frequencies retain sysfs units; temperatures are millidegrees C.')


def baseline_repeat(full, repeat):
    pairs = []
    for path in repeat.glob('*/*/summary.json'):
        rerun = json.loads(path.read_text())
        original_path = full / path.parent.relative_to(repeat) / 'summary.json'
        original = json.loads(original_path.read_text())
        assert all(original[key] == rerun[key] for key in
                   ('system', 'scene', 'variant', 'seed', 'frames', 'warmup_frames'))
        assert original['variant'] == 'baseline'
        assert json.loads((path.parent / 'source_hashes.json').read_text()) == json.loads(
            (original_path.parent / 'source_hashes.json').read_text())
        original_config = json.loads((original_path.parent / 'config.json').read_text())
        repeat_config = json.loads((path.parent / 'config.json').read_text())
        original_config['data'].pop('output', None)
        repeat_config['data'].pop('output', None)
        assert original_config == repeat_config
        pairs.append(dict(system=original['system'], scene=original['scene'], seed=original['seed'],
            original_ate_mm=1000 * original['ate_rmse_m'],
            repeat_ate_mm=1000 * rerun['ate_rmse_m'],
            ate_difference_mm=1000 * (rerun['ate_rmse_m'] - original['ate_rmse_m']),
            ate_change_pct=100 * (rerun['ate_rmse_m'] / original['ate_rmse_m'] - 1),
            original_tracking_ms=original['tracking_invocations']['mean_ms'],
            repeat_tracking_ms=rerun['tracking_invocations']['mean_ms'],
            tracking_change_pct=100 * (rerun['tracking_invocations']['mean_ms'] /
                                      original['tracking_invocations']['mean_ms'] - 1),
            source_hashes_and_configs='identical except output path'))
    return pairs


def plot_windows(rows, output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/coslam-execution-diagnostic-plots')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for system, phase, variants, filename in (
        ('coslam', 'tracking_call', ('baseline', 'baseline_fast', 'cached_fast'), 'tracking_windows'),
        ('ec_slam', 'mapping_call', ('baseline', 'mapping_fast', 'both_fast'), 'mapping_windows')):
        scenes = sorted({r['scene'] for r in rows if r['system'] == system and r['phase'] == phase})
        if not scenes:
            continue
        fig, axes = plt.subplots(1, len(scenes), figsize=(4.2 * len(scenes), 3.6),
                                 squeeze=False, constrained_layout=True)
        for ax, scene in zip(axes[0], scenes):
            for variant in variants:
                values = sorted((r for r in rows if r['system'] == system and r['phase'] == phase
                    and r['scene'] == scene and r['variant'] == variant), key=lambda r:r['first_frame'])
                ax.plot([(r['first_frame'] + r['last_frame']) / 2 for r in values],
                        [r['mean_ms'] for r in values], marker='o', ms=4, label=variant)
            ax.set_title(scene)
            ax.set_xlabel('Trajectory frame (400-frame windows)')
            ax.set_ylabel('Mean complete call time (ms)')
            ax.grid(alpha=.2)
        axes[0, -1].legend(fontsize=8)
        fig.savefig(output / (filename + '.png'), dpi=180)
        fig.savefig(output / (filename + '.svg'))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', nargs='+', type=Path, required=True)
    parser.add_argument('--telemetry', nargs='+', type=Path, required=True)
    parser.add_argument('--coslam-full', type=Path, required=True)
    parser.add_argument('--coslam-repeat', type=Path, required=True)
    parser.add_argument('--ec-full', type=Path)
    parser.add_argument('--ec-repeat', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for root in [*args.roots, args.coslam_repeat, *([args.ec_repeat] if args.ec_repeat else [])]:
        assert (root / 'SUITE_COMPLETE').exists(), f'Incomplete suite: {root}'
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [row for root in args.roots for row in timing_windows(root)]
    with (args.output / 'timing_windows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sensor_data = telemetry(args.telemetry)
    (args.output / 'telemetry_summary.json').write_text(json.dumps(sensor_data, indent=2) + '\n')
    repeats = baseline_repeat(args.coslam_full, args.coslam_repeat)
    if args.ec_repeat:
        assert args.ec_full is not None
        repeats += baseline_repeat(args.ec_full, args.ec_repeat)
    assert repeats, 'No completed baseline repeat'
    (args.output / 'baseline_repeat.json').write_text(json.dumps(repeats, indent=2) + '\n')
    lines = ['# Timing trends and repeat checks', '',
        'Timing windows describe portions of the same trajectories. They are not independent '
        'replicates or confidence intervals. `timing_windows.csv` retains paired 400-frame '
        'call timings and counts, excluding the first 20 frames. '
        '[Co tracking plot](tracking_windows.png) and [EC mapping plot](mapping_windows.png) '
        'show within-trajectory trends; lines summarize correlated windows, not accuracy '
        'replicates or isolated causal estimates.', '',
        '| System | Baseline scene | First ATE mm | Repeat ATE mm | Difference mm | ATE change | Tracking time change |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in repeats:
        lines.append(f"| {row['system']} | {row['scene']} | {row['original_ate_mm']:.4f} | {row['repeat_ate_mm']:.4f} | "
            f"{row['ate_difference_mm']:+.4f} | {row['ate_change_pct']:+.2f}% | {row['tracking_change_pct']:+.2f}% |")
    lines += ['', 'The repeat uses the same seed, source hashes and resolved configuration '
        '(except output path). Two baseline observations do not establish an equivalence '
        'margin, a confidence interval or the distribution of treatment effects.', '',
        'Read-only telemetry:', '', f"- {sensor_data['samples']} samples across "
        f"{len(args.telemetry)} file(s). Missing sensors remain unavailable."]
    for field, stats in sensor_data['groups']['all']['sensors'].items():
        if 'cpu0/cpufreq/' in field or '.gpu/cur_freq' in field:
            scale = 1000 if 'cpufreq' in field else 1_000_000
            label = 'CPU0' if 'cpufreq' in field else 'GPU'
            lines.append(f"- {label} observed frequency: {stats['min']/scale:.1f}–{stats['max']/scale:.1f} MHz.")
        if field.endswith(('thermal_zone0/temp', 'thermal_zone1/temp')):
            label = 'CPU' if 'thermal_zone0' in field else 'GPU'
            lines.append(f"- {label} observed temperature: {stats['min']/1000:.2f}–{stats['max']/1000:.2f} °C.")
    lines += ['', 'The continuous monitor begins partway through EC screening. Its labels '
        'are driver-level indicators, not exact optimizer ranges. Fixed observed clocks '
        'rule out a recorded frequency change during this coverage; they do not prove '
        'identical memory contention or eliminate all timing variability. No power or '
        'clock setting was changed by the experiment.']
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    plot_windows(rows, args.output)
    print(f'Wrote {len(rows)} timing windows and {len(repeats)} baseline repeat pair(s).')


if __name__ == '__main__':
    main()
