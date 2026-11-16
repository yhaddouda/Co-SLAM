"""Plot prefix ATE from completed full-scene four-dataset BA experiments.

Each point refits Horn rigid alignment on that prefix, without fitting scale.
Prefixes use final saved poses, including any later BA refinement of old poses;
these are not online ATE measurements captured during processing.
Only completed seed-0 full-scene runs are used;
300-frame screening runs are deliberately excluded. No CUDA or Torch needed.

Example: python scripts/Validation/plot_ba_accuracy.py results/ba_accuracy
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'coslam-ba-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SCENES = {'Tum': 'fr1_desk', 'ScanNet': 'scene0000', 'Synthetic': 'tg', 'Replica': 'office0'}
STYLES = {
    'dense': ('Dense BA + depth-spatial tracking', '#3465a4', '-', 'o'),
    'bucket': ('Legacy bucket BA + depth-spatial tracking', '#cf5c36', '-', 's'),
    'guarded': ('Guarded BA + depth-spatial tracking', '#25846b', '-', '^'),
    'ordinary': ('Dense BA + ordinary tracking', '#6850a1', '--', 'D'),
    'selected': ('Balanced tracking + stable dense BA', '#25846b', '-', '^'),
}


def horn_rmse_cm(ground_truth, estimated, scale=1.0):
    """Match tools/eval_ate.py align and its float64 translation arithmetic."""
    valid = np.isfinite(ground_truth).all(axis=(1, 2))
    truth = np.asarray(ground_truth[valid, :3, 3].T, dtype=np.float64)
    prediction = np.asarray(estimated[valid, :3, 3].T, dtype=np.float64)
    if truth.shape[1] < 3:
        return None
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite estimated poses at valid GT frames')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('data.sc_factor must be finite and positive')
    centered_prediction = prediction - prediction.mean(axis=1, keepdims=True)
    centered_truth = truth - truth.mean(axis=1, keepdims=True)
    covariance = np.zeros((3, 3))
    for column in range(prediction.shape[1]):
        covariance += np.outer(centered_prediction[:, column], centered_truth[:, column])
    left, _, right = np.linalg.svd(covariance.T)
    correction = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0:
        correction[2, 2] = -1
    rotation = left @ correction @ right
    translation = truth.mean(axis=1, keepdims=True) - rotation @ prediction.mean(axis=1, keepdims=True)
    residual = rotation @ prediction + translation - truth
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=0))) * 100 / scale)


def condition(summary):
    tracking, variant = summary.get('tracking'), summary.get('variant')
    if tracking == 'balanced_dense' and variant == 'stable':
        return 'selected'
    if tracking == 'baseline' and variant == 'baseline':
        return 'ordinary'
    if tracking != 'depth_spatial':
        return None
    return {'baseline': 'dense', 'bucket': 'bucket',
            'guarded': 'guarded', 'lattice_guarded': 'guarded'}.get(variant)


def completed_runs(root):
    selected = {}
    for family in ('full', 'original_tracking', 'revised_control', 'balanced'):
        for path in sorted((root / family).glob('*/summary.json')):
            summary = json.loads(path.read_text())
            dataset = summary.get('dataset')
            kind = condition(summary)
            trajectory_path = path.with_name('trajectories.npz')
            if (summary.get('seed') != 0 or dataset not in SCENES or kind is None
                    or summary.get('scene') != SCENES[dataset] or not trajectory_path.exists()):
                continue
            with np.load(trajectory_path) as trajectories:
                ground_truth = trajectories['ground_truth']
                estimated = trajectories['estimated']
            frames = summary['frames']
            if (len(estimated) != frames or ground_truth.shape != estimated.shape
                    or estimated.shape[1:] != (4, 4)):
                raise ValueError(f'Incomplete or inconsistent trajectories: {trajectory_path}')
            config_path = path.with_name('config.json')
            config = json.loads(config_path.read_text()) if config_path.exists() else {}
            # A saved nonzero frame cap denotes a screening run, even when
            # the directory was placed under "full".
            if config.get('timing', {}).get('max_frames'):
                continue
            scale = config.get('data', {}).get('sc_factor', 1.0)
            endpoints = sorted(set(range(100, frames + 1, 100)) | {frames})
            curve = [(end, horn_rmse_cm(ground_truth[:end], estimated[:end], scale))
                     for end in endpoints]
            curve = [(end, ate) for end, ate in curve if ate is not None]
            if not curve:
                raise ValueError(f'Fewer than three valid GT poses: {trajectory_path}')
            expected_cm = 100 * summary['ate_rmse_m']
            if not np.isclose(curve[-1][1], expected_cm, rtol=1e-6, atol=1e-6):
                raise ValueError(f'Final prefix ATE disagrees with summary: {path}: '
                                 f'{curve[-1][1]:.9f} versus {expected_cm:.9f} cm')
            key = (dataset, kind)
            previous = selected.get(key)
            # Prefer production guarded runs over an experimental lattice
            # alias, with deterministic selection if both were completed.
            if previous is not None:
                if not (kind == 'ordinary' and family == 'revised_control') and not (kind == 'guarded' and summary['variant'] == 'guarded'
                        and previous['variant'] != 'guarded'):
                    continue
            selected[key] = dict(curve=curve, source=str(path), variant=summary['variant'])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path('results/ba_accuracy'))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    runs = completed_runs(args.root)
    final_comparison = any(kind == 'selected' for _, kind in runs)
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (dataset, scene) in zip(axes.flat, SCENES.items()):
        available = []
        for kind, (label, color, line, marker) in STYLES.items():
            if final_comparison and kind not in {'bucket', 'ordinary', 'selected'}:
                continue
            if final_comparison and kind == 'bucket':
                label = 'Previous depth-pruned tracking + bucket BA'
            if final_comparison and kind == 'ordinary':
                label = 'Ordinary dense reference after shared fixes'
            run = runs.get((dataset, kind))
            if run is None:
                continue
            x, y = zip(*run['curve'])
            axis.plot(x, y, label=label, color=color, linestyle=line,
                      marker=marker, markersize=4, linewidth=1.8)
            available.append(kind)
            print(f'{dataset}/{scene}, {kind}: {x[-1]} frames, '
                  f'final prefix ATE={y[-1]:.6f} cm; {run["source"]}')
        axis.set_title(f'{dataset} / {scene}')
        axis.set_xlabel('Processed frames in prefix')
        axis.set_ylabel('Prefix ATE RMSE (cm)')
        axis.set_ylim(bottom=0)
        axis.grid(True, alpha=.2)
        if available:
            axis.legend(loc='best', fontsize=8)
            expected = ('bucket', 'ordinary', 'selected') if final_comparison else ('dense', 'bucket', 'guarded')
            pending = [kind for kind in expected if kind not in available]
            if pending:
                axis.text(.03, .96, 'Pending: ' + ', '.join(pending),
                          transform=axis.transAxes, va='top', color='#666666', fontsize=9)
        else:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            axis.text(.5, .5, 'Full-scene runs pending', transform=axis.transAxes,
                      ha='center', va='center', color='#666666', fontsize=12)
    figure.suptitle('Prefix ATE on completed full-scene runs (seed 0)\n'
                   'Rigid Horn alignment refitted per prefix; no scale fitting\n'
                   'Prefixes of final saved poses, including later BA refinement', fontsize=13)
    for extension in ('png', 'pdf'):
        output = args.root / f'prefix_ate_seed0.{extension}'
        figure.savefig(output, dpi=180 if extension == 'png' else None)
        print(f'SAVED {output}')
    plt.close(figure)


if __name__ == '__main__':
    main()
