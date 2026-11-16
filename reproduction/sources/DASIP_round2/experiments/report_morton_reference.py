#!/usr/bin/env python3
"""Audit/pool completed native BASE/MC quality runs across frozen queues."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo
from report_support_reference import audit_job, read_json, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    provenance = []
    runtime_hash = None
    for queue in args.queue:
        identity = read_json(queue / 'queue_identity.json')
        digest = identity['runtime_manifest_sha256']
        if runtime_hash is None:
            runtime_hash = digest
        assert digest == runtime_hash, 'Cannot pool different frozen runtimes'
        jobs = read_json(queue / 'job_manifest.json')['jobs']
        provenance.append(dict(queue=str(queue.resolve()), complete=(queue / 'queue_complete.json').is_file(), identity=identity))
        for job in jobs:
            if job['condition'] not in ('BASE', 'MC') or not (queue / (job['id'] + '.done.json')).is_file():
                continue
            row = audit_job(queue, job, identity)
            cfg = read_json(queue / job['id'] / 'config.json')
            row['hash_size'] = cfg['grid']['hash_size']
            cfg.pop('round2')
            cfg['data'].pop('output')
            assert cfg['grid']['hash'] == ('CoherentPrime' if job['condition'] == 'BASE' else 'Morton')
            assert cfg['grid']['morton_sort'] == (job['condition'] == 'MC')
            cfg['grid'].pop('hash')
            cfg['grid'].pop('morton_sort')
            row['matched_native_config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
            row['queue'] = str(queue.resolve())
            rows.append(row)
    grouped = {}
    for row in rows:
        key = (row['dataset'], row['scene'], row['hash_size'])
        group = grouped.setdefault(key, {}).setdefault(row['condition'], {})
        assert row['seed'] not in group, 'Duplicate run; do not silently select/reuse a different result'
        group[row['seed']] = row
    comparisons = []
    for (dataset, scene, capacity), group in sorted(grouped.items()):
        if any(set(group.get(c, {})) != {0, 1, 2} for c in ('BASE', 'MC')):
            continue
        for seed in (0, 1, 2):
            assert group['BASE'][seed]['matched_native_config_sha256'] == group['MC'][seed]['matched_native_config_sha256']
            assert group['BASE'][seed]['frames'] == group['MC'][seed]['frames']
        row = dict(dataset=dataset, scene=scene, hash_size=capacity, paired_seeds=3)
        for field in ('ate_cm', 'corrected_ate_cm', 'tracking_wall_ms', 'ba_wall_ms', 'tracking_cuda_ms', 'ba_cuda_ms'):
            for condition in ('BASE', 'MC'):
                values = [group[condition][seed][field] for seed in (0, 1, 2)]
                row[condition + '_' + field] = statistics.mean(values)
                row[condition + '_' + field + '_sd'] = statistics.stdev(values)
            base, candidate = row['BASE_' + field], row['MC_' + field]
            row[field + '_change_pct'] = 100 * (candidate / base - 1) if base else None
        comparisons.append(row)
    def value(row, condition, field):
        return f"{row[condition + '_' + field]:.3f} ± {row[condition + '_' + field + '_sd']:.3f}"
    def pct(number):
        return 'undefined' if number is None else f'{number:+.2f}%'
    report = '# Native-reference Morton comparison\n\n'
    report += 'Generated ' + datetime.now(ZoneInfo('Europe/Paris')).isoformat() + ' (Europe/Paris).\n\n'
    report += f'{len(rows)} audited completed BASE/MC runs; {len(comparisons)} scene/capacity comparisons with all three paired seeds. No point pruning or engineering staging/pose-commit changes are active. Hash capacity is an explicit comparison axis. Sources and all other effective native settings match.\n\n'
    report += '| Scene / capacity | BASE direct ATE cm | MC direct ATE cm | Change | BASE corrected ATE cm | MC corrected ATE cm | Change |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in comparisons:
        report += f"| {r['dataset']}/{r['scene']} H{r['hash_size']} | {value(r,'BASE','ate_cm')} | {value(r,'MC','ate_cm')} | {pct(r['ate_cm_change_pct'])} | {value(r,'BASE','corrected_ate_cm')} | {value(r,'MC','corrected_ate_cm')} | {pct(r['corrected_ate_cm_change_pct'])} |\n"
    report += '\n| Scene / capacity | BASE tracking ms | MC tracking ms | Time reduction | BASE BA ms | MC BA ms | Time reduction |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in comparisons:
        report += f"| {r['dataset']}/{r['scene']} H{r['hash_size']} | {value(r,'BASE','tracking_wall_ms')} | {value(r,'MC','tracking_wall_ms')} | {pct(-r['tracking_wall_ms_change_pct'])} | {value(r,'BASE','ba_wall_ms')} | {value(r,'MC','ba_wall_ms')} | {pct(-r['ba_wall_ms_change_pct'])} |\n"
    report += '\nMeans ± sample SD; time reduction is 1 minus the ratio of matched seed means, using complete-call wall time after frames 0–19. CSVs retain CUDA timing. Raw trajectories/ATE and timing CSV coverage were independently checked. Reconstruction metric scoring remains pending; exported meshes are not scored metrics.\n\n'
    report += 'This is a measurement report, not a guarantee of positive gains. MC is distinct from the DP candidate and is not evaluated against a fabricated 30% MC speed requirement. The native-capacity Replica aggregate requires all eight scenes; do not substitute a favourable subset or mix capacity 19 into capacity 16. The two ATE conventions remain visible until the final table convention is fixed.\n'
    write_csv(args.output / 'per_run.csv', rows)
    write_csv(args.output / 'per_scene.csv', comparisons)
    (args.output / 'REPORT.md').write_text(report)
    (args.output / 'ANALYSIS_STATUS.json').write_text(json.dumps(dict(audited_runs=len(rows), complete_comparisons=len(comparisons), provenance=provenance, reconstruction_scoring='pending'), indent=2) + '\n')
    print('MORTON_REPORT', len(rows), 'runs;', len(comparisons), 'complete comparisons', flush=True)


if __name__ == '__main__':
    main()
