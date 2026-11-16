#!/usr/bin/env python3
"""Export paper-ready plots only for a complete, qualified pruning study."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def export_tables(payload, destination):
    """Export numeric paper tables from the independently audited result."""
    cases = payload['cases']
    columns = ['dataset', 'scene', 'seed', 'frames', 'dense_ate_cm', 'selected_ate_cm',
               'ate_change_pct', 'dense_relative_ate_cm', 'selected_relative_ate_cm',
               'dense_tracking_ms', 'selected_tracking_ms', 'tracking_time_saved_pct',
               'dense_ba_ms', 'selected_ba_ms', 'ba_time_saved_pct',
               'dense_run_s', 'selected_run_s', 'dense_tracking_points',
               'selected_tracking_points', 'dense_ba_points', 'selected_ba_points',
               'dense_feature_points', 'selected_feature_points',
               'dense_pose_blocks', 'selected_pose_blocks']
    paired = [{key: row[key] for key in columns} for row in cases]
    aggregated = []
    for scene in payload['scenes']:
        rows = [row for row in cases if (row['dataset'], row['scene']) ==
                (scene['dataset'], scene['scene'])]
        result = {key: scene[key] for key in ('dataset', 'scene', 'frames',
                  'ate_change_pct', 'tracking_time_saved_pct', 'ba_time_saved_pct',
                  'tracking_target', 'ba_target', 'passed')}
        result['seeds'] = len(rows)
        for metric in ('ate_cm', 'relative_ate_cm', 'tracking_ms', 'ba_ms', 'run_s'):
            for family in ('dense', 'selected'):
                key = family + '_' + metric
                values = [row[key] for row in rows]
                result[key + '_mean'] = statistics.mean(values)
                result[key + '_sd'] = statistics.stdev(values)
        for metric in ('tracking_points', 'ba_points', 'feature_points', 'pose_blocks'):
            dense = sum(row['dense_' + metric] for row in rows)
            selected = sum(row['selected_' + metric] for row in rows)
            result['dense_' + metric + '_sum'] = dense
            result['selected_' + metric + '_sum'] = selected
            result[metric + '_pruned_pct'] = 100 * (1 - selected / dense)
        aggregated.append(result)
    for filename, rows in (('paired_seed_metrics.csv', paired), ('scene_metrics.csv', aggregated)):
        with (destination / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def plot(payload, destination):
    scenes, cases = payload['scenes'], payload['cases']
    if not (payload.get('qualified') is True and payload.get('complete') is True
            and len(scenes) == 8 and len(cases) == 24
            and all(s['passed'] and set(s['seeds']) == {0, 1, 2} for s in scenes)):
        raise ValueError('Plots require the complete eight-scene, three-seed acceptance audit')
    # Present the datasets together; the experimental execution order is archived separately.
    order = {name: i for i, name in enumerate(('Replica', 'Synthetic', 'Tum', 'ScanNet'))}
    scenes = sorted(scenes, key=lambda s: (order[s['dataset']], s['scene']))
    labels = [f"{s['dataset'].replace('Tum', 'TUM')} / {s['scene']}" for s in scenes]
    colors = ['#3973a7', '#3973a7', '#479681', '#479681',
              '#bf8340', '#bf8340', '#8b6ba6', '#8b6ba6']
    panels = [('ate_change_pct', 'ATE RMSE change (%)', 15, None),
              ('tracking_time_saved_pct', 'Tracking wall time reduction (%)', 5, 10),
              ('ba_time_saved_pct', 'BA wall time reduction (%)', 25, 30)]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4), sharey=True)
    y = np.arange(len(scenes))
    for ax, (key, title, gate, target) in zip(axes, panels):
        values = [s[key] for s in scenes]
        ax.barh(y, values, height=.55, color=colors, alpha=.8, zorder=2)
        for i, scene in enumerate(scenes):
            paired = sorted((r for r in cases if (r['dataset'], r['scene']) ==
                             (scene['dataset'], scene['scene'])), key=lambda r: r['seed'])
            dots = [r[key] for r in paired]
            ax.scatter(dots, i + np.array([-.10, 0, .10]), s=17, facecolors='white',
                       edgecolors='#222222', linewidths=.6, zorder=4)
            ax.annotate(f'{values[i]:+.1f}%' if key == 'ate_change_pct' else f'{values[i]:.1f}%',
                        (values[i], i), xytext=(5 if values[i] >= 0 else -5, 0),
                        textcoords='offset points', ha='left' if values[i] >= 0 else 'right',
                        va='center', fontsize=8, zorder=5)
        ax.axvline(0, color='#666666', linewidth=.7, zorder=1)
        ax.axvline(gate, color='#ad352d', linestyle='--', linewidth=1,
                   label=f'Acceptance {"ceiling" if key == "ate_change_pct" else "floor"}: {gate}%')
        if target is not None:
            ax.axvline(target, color='#404040', linestyle=':', linewidth=1,
                       label=f'Nominal target: {target}%')
        ax.set_title(title, fontsize=10, pad=12)
        ax.grid(axis='x', color='#dddddd', linewidth=.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
        ax.margins(x=.22)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, -.20), fontsize=7,
                  frameon=False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.suptitle('Joint ray and keyframe-constraint pruning — full-sequence paired evaluation',
                 fontsize=12, y=.98)
    fig.text(.5, .025,
             'Bars: percentage change of three-seed means. Dots: individual paired-seed changes.\n'
             'Stage timings include all preprocessing and pose work; frames 0–19 are omitted.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .10, 1, .95), w_pad=2)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    export_tables(payload, destination)
    for suffix in ('png', 'pdf'):
        fig.savefig(destination / f'pruning8_tradeoffs.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_history(payload, root, destination):
    """Describe measured BA scaling; this is not a component attribution."""
    order = {name: i for i, name in enumerate(('Replica', 'Synthetic', 'Tum', 'ScanNet'))}
    scenes = sorted(payload['scenes'], key=lambda s: (order[s['dataset']], s['scene']))
    fig, axes = plt.subplots(4, 2, figsize=(11, 11))
    for ax, scene in zip(axes.flat, scenes):
        for family, label, color in (('controls', 'Dense', '#3973a7'),
                                     (payload['candidate'], 'Pruned', '#c17c35')):
            frames, samples = None, []
            for seed in (0, 1, 2):
                directory = Path(root) / family / f"{scene['dataset']}_{scene['scene']}_seed{seed}"
                with (directory / 'frames.csv').open(newline='') as stream:
                    records = list(csv.DictReader(stream))
                if [int(r['frame_id']) for r in records] != list(range(1, scene['frames'])):
                    raise ValueError(f'Incomplete trajectory timing records: {directory}')
                calls = [r for r in records if int(r['has_ba']) and int(r['frame_id']) >= 20]
                ids = np.array([int(r['frame_id']) for r in calls])
                if frames is not None and not np.array_equal(frames, ids):
                    raise ValueError('BA call schedules differ between seeds')
                frames = ids
                samples.append([float(r['ba_wall_ms']) for r in calls])
            values = np.asarray(samples)
            bins = np.array_split(np.arange(len(frames)), min(20, len(frames)))
            x = [np.mean(frames[b] / 5) for b in bins]
            per_seed = np.stack([np.mean(values[:, b], axis=1) for b in bins], axis=1)
            mean, sd = per_seed.mean(axis=0), per_seed.std(axis=0, ddof=1)
            ax.plot(x, mean, color=color, label=label, linewidth=1.5)
            ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=.16)
        ax.set_title(f"{scene['dataset'].replace('Tum', 'TUM')} / {scene['scene']}", fontsize=10)
        ax.set_xlabel('Historical keyframes at BA call', fontsize=8)
        ax.set_ylabel('BA wall time (ms)', fontsize=8)
        ax.grid(color='#dddddd', linewidth=.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .025))
    fig.suptitle('Complete BA invocation time as the keyframe history grows', fontsize=12, y=.985)
    fig.text(.5, .015, '20 bins per scene; lines show three-seed means; shading shows sample SD of seed means. '
             'Frames 0–19 are omitted.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .97), h_pad=2.2)
    for suffix in ('png', 'pdf'):
        fig.savefig(Path(destination) / f'pruning8_ba_history.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('acceptance', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--runs', type=Path, help='Optional audited run root for the BA history figure')
    args = parser.parse_args()
    payload = json.loads(args.acceptance.read_text())
    plot(payload, args.destination)
    if args.runs is not None:
        plot_history(payload, args.runs, args.destination)
