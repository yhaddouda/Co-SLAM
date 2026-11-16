"""Rank paper BA-only ablations or saved checkpoints by ATE degradation."""
import argparse
import csv
import json
import sys
from statistics import fmean
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.eval_ate import align


def trajectory_rmse(gt, estimated):
    """Same rigid Horn alignment as tools/eval_ate.py, with no scale fitting."""
    keys = sorted(estimated)
    # evaluate_ate constructs Python floats before its NumPy matrices, hence
    # float64 even when input poses were float32. Match that arithmetic exactly.
    truth = np.asarray(np.stack([gt[i][:3, 3] for i in keys]).T, dtype=np.float64)
    prediction = np.asarray(np.stack([estimated[i].detach().cpu().numpy()[:3, 3] for i in keys]).T, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite trajectory')
    _, _, errors = align(np.matrix(prediction), np.matrix(truth))
    return float(np.sqrt(np.mean(errors ** 2)))


def checkpoint_metrics(path, cfg):
    checkpoint = torch.load(path, map_location='cpu', mmap=True)
    gt = np.loadtxt(Path(cfg['data']['datadir']) / 'traj.txt').reshape(-1, 4, 4)
    gt[:, :3, 3] *= cfg['data']['sc_factor']
    poses = checkpoint['pose']
    relative = {}
    every = cfg['mapping']['keyframe_every']
    for i in sorted(poses):
        relative[i] = poses[i] if i % every == 0 else checkpoint['pose_rel'][i] @ poses[i // every * every]
    scale = cfg['data']['sc_factor']
    return dict(ate_rmse_m=trajectory_rmse(gt, poses) / scale,
                ate_relative_rmse_m=trajectory_rmse(gt, relative) / scale,
                frames=len(poses))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('paper', 'checkpoints'), default='paper')
    parser.add_argument('--root', type=Path, default=Path('output/Replica'))
    parser.add_argument('--output', type=Path, default=Path('results/ba_experiments/scene_ranking.csv'))
    args = parser.parse_args()
    if args.source == 'paper':
        path = Path('Quality/depth sampling/replica_tracking_ba_ate_results.csv')
        with path.open(newline='') as handle:
            rows = [r for r in csv.DictReader(handle) if 1 <= int(r['Run']) <= 7]
        rankings = []
        for scene in sorted({r['Scene'] for r in rows}):
            baseline = [r for r in rows if r['Scene'] == scene and r['Config'] == 'A']
            bucket = [r for r in rows if r['Scene'] == scene and r['Config'] == 'C']
            if {r['Run'] for r in baseline} != {r['Run'] for r in bucket}:
                raise ValueError(f'Unpaired paper runs for {scene}')
            b = fmean(float(r['ATE_RMSE']) for r in baseline)
            a = fmean(float(r['ATE_RMSE']) for r in bucket)
            rankings.append(dict(scene=scene, baseline_ate_m=b, bucket_ate_m=a,
                                 degradation_pct=100 * (a / b - 1), paired_runs=len(baseline), source=str(path)))
        rankings.sort(key=lambda r: r['degradation_pct'], reverse=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rankings[0]))
            writer.writeheader()
            writer.writerows(rankings)
        print(json.dumps(rankings, indent=2))
        return
    records = []
    # These explicitly named families avoid mixing hash/sampling sweeps.
    for scene in ('office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2'):
        families = [('depth_spatial_three_modes', 'depth_spatial_timing_baseline_Size19_Run{run}',
                     'depth_spatial_timing_depth_spatial_Size19_Run{run}'),
                    ('', 'baseline_Size19_Run{run}', 'pruningD_Size19_Run{run}')]
        for family, baseline, alternative in families:
            for run in range(1, 6):
                parent = args.root / scene / family / f'run_{run}'
                paths = [parent / name.format(run=run) for name in (baseline, alternative)]
                if not all((p / 'config.json').exists() and (p / 'checkpoint1999.pt').exists() for p in paths):
                    continue
                configs = [json.loads((p / 'config.json').read_text()) for p in paths]
                values = [checkpoint_metrics(p / 'checkpoint1999.pt', cfg) for p, cfg in zip(paths, configs)]
                b, a = values
                records.append(dict(scene=scene, family=family or 'legacy_pruningD', run=run,
                    baseline_ate_m=b['ate_rmse_m'], alternative_ate_m=a['ate_rmse_m'],
                    degradation_pct=100 * (a['ate_rmse_m'] / b['ate_rmse_m'] - 1),
                    baseline_relative_ate_m=b['ate_relative_rmse_m'],
                    alternative_relative_ate_m=a['ate_relative_rmse_m'],
                    relative_degradation_pct=100 * (a['ate_relative_rmse_m'] / b['ate_relative_rmse_m'] - 1),
                    tracking_depth_spatial=configs[1].get('tracking_depth_spatial', {}).get('enabled', False),
                    tracking_naive_pruning=configs[1].get('tracking_naive_pruning', {}).get('enabled', False),
                    ba_bucket=configs[1].get('ba_bucket', {}).get('enabled', False),
                    baseline_checkpoint=str(paths[0] / 'checkpoint1999.pt'),
                    alternative_checkpoint=str(paths[1] / 'checkpoint1999.pt')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if records:
        with args.output.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    for family in sorted({r['family'] for r in records}):
        ranking = []
        for scene in sorted({r['scene'] for r in records if r['family'] == family}):
            rows = [r for r in records if r['family'] == family and r['scene'] == scene]
            b = np.mean([r['baseline_ate_m'] for r in rows])
            a = np.mean([r['alternative_ate_m'] for r in rows])
            ranking.append((100 * (a / b - 1), scene, b, a, len(rows)))
        print(family, sorted(ranking, reverse=True), flush=True)


if __name__ == '__main__':
    main()
