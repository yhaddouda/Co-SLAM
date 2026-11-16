"""Report all completed BA accuracy runs, with ATE in centimetres."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    records = []
    for path in sorted(args.root.glob('**/summary.json')):
        row = json.loads(path.read_text())
        if 'dataset' not in row:
            continue
        records.append(dict(suite=str(path.parent.parent.relative_to(args.root)),
                            dataset=row['dataset'], scene=row['scene'], variant=row['variant'],
                            tracking=row['tracking'], seed=row['seed'], frames=row['frames'],
                            ate_cm=row['ate_rmse_m'] * 100,
                            relative_ate_cm=row['ate_relative_rmse_m'] * 100,
                            ba_ms=row['ba_invocations']['mean_ms'],
                            network_reduction_pct=row.get('network_samples', {}).get('reduction_pct', ''),
                            source=str(path)))
    if not records:
        raise ValueError('No completed four-dataset runs')
    with (args.root / 'measurements.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ['# BA accuracy measurements', '',
             'Rigidly aligned translation RMSE; no scale fitting. All completed runs are included.',
             'Each row is mean ± sample SD across seeds (a single seed has no SD).',
             '`baseline` with `depth_spatial` tracking is the tracking-only control.', '',
             '| Suite | Dataset / scene | BA variant | Tracking | Frames | Seeds | ATE (cm) | Relative ATE (cm) | BA (ms) |',
             '| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |']
    keys = sorted({tuple(r[k] for k in ('suite', 'dataset', 'scene', 'variant', 'tracking', 'frames')) for r in records})
    for key in keys:
        group = [r for r in records if tuple(r[k] for k in ('suite', 'dataset', 'scene', 'variant', 'tracking', 'frames')) == key]
        def metric(name):
            values = [r[name] for r in group]
            return f'{mean(values):.4f}' + (f' ± {stdev(values):.4f}' if len(values) > 1 else '')
        suite, dataset, scene, variant, tracking, frames = key
        seeds = ', '.join(str(r['seed']) for r in group)
        lines.append(f'| {suite} | {dataset} / {scene} | {variant} | {tracking} | {frames} | {seeds} | {metric("ate_cm")} | {metric("relative_ate_cm")} | {metric("ba_ms")} |')
    (args.root / 'MEASUREMENTS.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
