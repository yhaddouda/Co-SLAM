"""Export every completed pruning run and reproducible aggregated tables."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def collect(root):
    rows = []
    for path in sorted(root.glob('*/*/summary.json')):
        s = json.loads(path.read_text())
        name = path.parent.name
        preset = name[len(s['dataset'])+1:].rsplit('_seed', 1)[0]
        row = dict(family=path.parent.parent.name, dataset=s['dataset'], scene=s['scene'],
            preset=preset, seed=s['seed'], frames=s['frames'], ate_cm=100*s['ate_rmse_m'],
            relative_ate_cm=100*s['ate_relative_rmse_m'], run_seconds=s['run_wall_seconds'],
            path=str(path.relative_to(root)))
        total_dense, total_queried = 0, 0
        for stage, label in [('Tracking','tracking'), ('Bundle_Adjustment','ba'), ('First_frame_mapping','initialization')]:
            counts = s.get('stage_queries',{}).get(stage,{})
            row[label+'_reduction_pct'] = counts.get('reduction_pct',0.)
            row[label+'_dense'] = counts.get('dense_samples',0)
            row[label+'_queried'] = counts.get('queried_samples',0)
        for counts in s.get('stage_queries',{}).values():
            total_dense += counts['dense_samples']
            total_queried += counts['queried_samples']
        row['all_stages_reduction_pct'] = 100*(1-total_queried/total_dense) if total_dense else 0.
        records = []
        diagnostic = path.parent/'pruning_diagnostics.csv'
        if diagnostic.exists():
            with diagnostic.open() as f:
                records = list(csv.DictReader(f))
        rays = sum(int(r['rays']) for r in records)
        row['recovered_rays_pct'] = 100*sum(int(r['recovered_rays']) for r in records)/rays if rays else 0.
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    args = p.parse_args()
    rows = collect(args.root)
    if not rows:
        raise ValueError('No results found')
    with (args.root/'measurements.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    groups = defaultdict(list)
    full = {'Tum':592, 'Replica':2000, 'Synthetic':395}
    for r in rows:
        if r['frames'] == full.get(r['dataset']):
            groups[(r['preset'],r['dataset'])].append(r)
    text = ['# Full-scene pruning measurements', '',
        'ATE is in cm; mean ± sample SD. Query reduction is mean percent of actual joint RGB/SDF queries saved, including recovery. Initialization stays dense. All completed full-scene runs are included.', '',
        '| Preset | Dataset | n | ATE (cm) | Relative ATE (cm) | Tracking pruning (%) | BA pruning (%) | All-stage pruning (%) | Run time (s) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    aggregate = []
    for (preset, dataset), values in sorted(groups.items()):
        a = dict(preset=preset,dataset=dataset,n=len(values),seeds=[r['seed'] for r in values])
        for metric in ('ate_cm','relative_ate_cm','tracking_reduction_pct','ba_reduction_pct','all_stages_reduction_pct','run_seconds','recovered_rays_pct'):
            a[metric] = statistics.mean(r[metric] for r in values)
            a[metric+'_sd'] = statistics.stdev(r[metric] for r in values) if len(values)>1 else 0.
        aggregate.append(a)
        text.append(f"| {preset} | {dataset} | {a['n']} | {a['ate_cm']:.4f} ± {a['ate_cm_sd']:.4f} | {a['relative_ate_cm']:.4f} | {a['tracking_reduction_pct']:.2f} | {a['ba_reduction_pct']:.2f} | {a['all_stages_reduction_pct']:.2f} | {a['run_seconds']:.2f} |")
    text += ['', '## Individual runs (including smoke checks)', '',
        '| Family | Dataset | Preset | Seed | Frames | ATE (cm) | Tracking pruning (%) | BA pruning (%) |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        text.append(f"| {r['family']} | {r['dataset']} | {r['preset']} | {r['seed']} | {r['frames']} | {r['ate_cm']:.4f} | {r['tracking_reduction_pct']:.2f} | {r['ba_reduction_pct']:.2f} |")
    failures = sorted(args.root.glob('*/*.failure.json'))
    text += ['', f'Failed runs: {len(failures)}.'] + [f'- `{f.relative_to(args.root)}`: {f.read_text()}' for f in failures]
    (args.root/'MEASUREMENTS.md').write_text('\n'.join(text)+'\n')
    (args.root/'aggregates.json').write_text(json.dumps(aggregate,indent=2))
    print('\n'.join(text[:6]+text[6:6+len(groups)]))


if __name__ == '__main__':
    main()
