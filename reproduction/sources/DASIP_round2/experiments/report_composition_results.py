#!/usr/bin/env python3
"""Audit RTX composition against the three already measured controls; CPU only."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from checked_reconstruction import save_json, sha
from report_support_reference import audit_job, write_csv


def normalized_config(path):
    cfg = json.loads(Path(path).read_text())
    cfg.pop('round2')
    cfg['data'].pop('output')
    cfg['grid'].pop('hash')
    cfg['grid'].pop('morton_sort')
    return cfg


def audit(directory, condition, seed, dataset, scene, identity):
    cfg = json.loads((directory / 'config.json').read_text())
    job = dict(id=directory.name, condition=condition, seed=seed, dataset=dataset,
               scene=scene, export_mesh=dataset in ('Replica', 'Synthetic'))
    row = audit_job(directory.parent, job, identity)
    assert cfg['grid']['hash'] == ('Morton' if condition in ('MC', 'MC_DP') else 'CoherentPrime')
    assert cfg['grid']['morton_sort'] == (condition in ('MC', 'MC_DP'))
    assert cfg['round2']['sampler'] == ('support' if condition in ('DP', 'MC_DP') else 'dense')
    assert cfg['round2']['margin_scene_units'] == .15 and cfg['round2']['anchor_every'] == 3
    assert json.loads((directory / 'summary.json').read_text())['warmup_frames'] == 20
    calls = defaultdict(set)
    with (directory / 'calls.csv').open() as f:
        for value in csv.DictReader(f):
            if int(value['frame_id']) >= 20:
                assert int(value['frame_id']) not in calls[value['stage']]
                calls[value['stage']].add(int(value['frame_id']))
    intervals = defaultdict(dict)
    with (directory / 'iterations.csv').open() as f:
        for value in csv.DictReader(f):
            frame = int(value['frame_id'])
            if frame < 20:
                continue
            group = intervals[(value['iteration_type'], frame)]
            iteration = int(value['iteration_id'])
            assert iteration not in group
            group[iteration] = float(value['full_iteration_cuda_ms'])
    for stage, prefix, updates in [('tracking', 'tracking', cfg['tracking']['iter']),
                                   ('bundle_adjustment', 'ba', cfg['mapping']['iters'])]:
        totals = []
        assert {frame for kind, frame in intervals if kind == stage} == calls[stage]
        for frame in calls[stage]:
            group = intervals[(stage, frame)]
            assert set(group) == set(range(updates))
            totals.append(sum(group.values()))
        row[prefix + '_iteration_total_cuda_ms'] = statistics.mean(totals)
    row.update(queue=str(directory.parent), hash_size=cfg['grid']['hash_size'])
    return row


def report(base):
    manifest = base / 'experiments/manifests/rtx_composition_all_scenes.json'
    if not manifest.exists():
        return dict(completed_runs=0, total_runs=72, complete_scenes=0, groups=[]), []
    jobs = json.loads(manifest.read_text())['jobs']
    queue = base / 'results/composition_v1'
    expected_source = sha(base / 'experiments/coslam_reference_runtime_v1/RUNTIME_MANIFEST.json')
    identity = dict(runtime_manifest_sha256=expected_source)
    sources = [manifest]
    rows = []
    for job in jobs:
        if not (queue / (job['id'] + '.done.json')).exists():
            continue
        directory = queue / job['id']
        paths = dict(job['controls'], MC_DP=str(directory))
        normalized = [normalized_config(Path(path) / 'config.json') for path in paths.values()]
        assert all(cfg == normalized[0] for cfg in normalized)
        frames = []
        for condition, value in paths.items():
            path = Path(value)
            assert (path.parent / (path.name + '.done.json')).exists()
            row = audit(path, condition, job['seed'], job['dataset'], job['scene'], identity)
            assert row['hash_size'] == job['hash_size']
            rows.append(row)
            frames.append(row['frames'])
            sources.extend(path / name for name in ('summary.json', 'config.json', 'source_manifest.json',
                                                    'trajectories.npz', 'calls.csv', 'iterations.csv', 'queries.csv'))
        assert set(frames) == {job['expected_native_frames']}
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[(row['dataset'], row['scene'])].append(row)
    complete = {key: values for key, values in by_scene.items()
                if len(values) == 12 and {(r['condition'], r['seed']) for r in values}
                == {(c, s) for c in ('BASE', 'MC', 'DP', 'MC_DP') for s in (0, 1, 2)}}
    fields = ('ate_cm', 'corrected_ate_cm', 'tracking_wall_ms', 'ba_wall_ms',
              'tracking_iteration_total_cuda_ms', 'ba_iteration_total_cuda_ms')
    groups = []
    scene_rows = []
    for (dataset, scene), values in sorted(complete.items()):
        result = dict(dataset=dataset, scene=scene, paired_seeds=3)
        for condition in ('BASE', 'MC', 'DP', 'MC_DP'):
            for field in fields:
                numbers = [r[field] for r in values if r['condition'] == condition]
                result[condition + '_' + field] = statistics.mean(numbers)
                result[condition + '_' + field + '_sd'] = statistics.stdev(numbers)
        scene_rows.append(result)
    for dataset, expected in [('Replica', 8), ('Synthetic', 7), ('Tum', 3), ('ScanNet', 6)]:
        keys = [key for key in complete if key[0] == dataset]
        if not keys:
            continue
        result = dict(dataset=dataset, scene_count=len(keys), expected_scenes=expected)
        for condition in ('BASE', 'MC', 'DP', 'MC_DP'):
            for field in fields:
                numbers = [statistics.mean(next(r[field] for r in complete[key]
                                                if r['condition'] == condition and r['seed'] == seed)
                                           for key in keys) for seed in (0, 1, 2)]
                result[condition + '_' + field] = statistics.mean(numbers)
                result[condition + '_' + field + '_sd'] = statistics.stdev(numbers)
        groups.append(result)
    result = dict(completed_runs=len(rows) // 4, total_runs=len(jobs), complete_scenes=len(complete),
                  expected_scenes=24, groups=groups, seeds=[0, 1, 2],
                  failures=[json.loads(p.read_text()) for p in queue.glob('*.failed.json')],
                  scope='RTX native scene capacities; MC+DP independent training; BASE/MC/DP controls reused; all outcomes retained',
                  aggregation='Equal scene weights on complete three-seed matched scene groups; sample SD across seed-wise dataset means',
                  input_sha256={str(p.relative_to(base)): sha(p) for p in sources})
    out = base / 'results/composition_v1_analysis'
    out.mkdir(exist_ok=True)
    if rows:
        write_csv(out / 'per_run.csv', rows)
    if scene_rows:
        write_csv(out / 'per_scene.csv', scene_rows)
    if groups:
        write_csv(out / 'dataset_means.csv', groups)
    save_json(out / 'SUMMARY.json', result)
    text = '# RTX MC+DP composition evidence\n\n'
    text += f"{result['completed_runs']}/{len(jobs)} new full runs; {len(complete)}/24 complete three-seed scenes.\n\n"
    text += 'Native BASE/MC/DP outputs are reused after matching sources, configurations, full trajectories/ATE, call coverage and iteration IDs/counts. Complete-call host wall latency and summed iteration CUDA-event intervals stay separate. Both ATE conventions and all measured slowdowns remain visible. Query pruning is verified by the existing raw-output auditor. No composition result is inferred by adding separate gains.\n\n'
    text += '| Dataset | Scenes | MC+DP direct ATE change vs BASE | Tracking reduction | BA reduction | BA reduction vs MC |\n|---|---:|---:|---:|---:|---:|\n'
    for row in groups:
        changes = [100 * (row['MC_DP_ate_cm'] / row['BASE_ate_cm'] - 1),
                   100 * (1 - row['MC_DP_tracking_wall_ms'] / row['BASE_tracking_wall_ms']),
                   100 * (1 - row['MC_DP_ba_wall_ms'] / row['BASE_ba_wall_ms']),
                   100 * (1 - row['MC_DP_ba_wall_ms'] / row['MC_ba_wall_ms'])]
        text += f"| {row['dataset']} | {row['scene_count']}/{row['expected_scenes']} | " + ' | '.join(f'{v:+.2f}%' for v in changes) + ' |\n'
    (out / 'REPORT.md').write_text(text)
    return result, sources + [out / 'SUMMARY.json']


def tex(result, table, val, gain):
    text = r'\subsection{Measured Composition of Morton Execution and Pruning}' + '\n'
    text += f"RTX MC+DP: {result['completed_runs']}/72 full runs; {result['complete_scenes']}/24 complete three-seed scenes. "
    text += r'BASE, MC and DP controls are reused only after configuration/source and raw-output checks. No combined effect is inferred from separate gains. Pending scenes remain explicit.' + '\n'
    quality, timing = [], []
    groups = {r['dataset']: r for r in result['groups']}
    for dataset, total in [('Replica', 8), ('Synthetic', 7), ('Tum', 3), ('ScanNet', 6)]:
        row = groups.get(dataset)
        label = ('TUM' if dataset == 'Tum' else dataset) + f" ({row['scene_count'] if row else 0}/{total})"
        for condition in ('BASE', 'MC', 'DP', 'MC_DP'):
            name = condition.replace('_', '+')
            delta = lambda field: (f"{100*(row[condition+'_'+field]/row['BASE_'+field]-1):+.2f}" if row else r'\pending')
            quality.append([label if condition == 'BASE' else '', name,
                            val(row, condition, 'ate_cm'), delta('ate_cm'),
                            val(row, condition, 'corrected_ate_cm'), delta('corrected_ate_cm')])
            values = [label if condition == 'BASE' else '', name]
            for field in ('tracking_wall_ms', 'ba_wall_ms', 'ba_iteration_total_cuda_ms'):
                values.extend([val(row, condition, field),
                               gain(100*(1-row[condition+'_'+field]/row['BASE_'+field])) if row else r'\pending'])
            timing.append(values)
    text += table(r'RTX composition ATE (cm): equal scene weights, mean $\pm$ sample SD over three paired seeds. Changes are relative to the matched BASE scene subset; positive is degradation. Native capacities: Replica/Synthetic/TUM H16, ScanNet H12/H14/H16. Coverage is complete/full scenes.',
                  'tab:composition_quality', ['Dataset ($n/N$)', 'Condition', 'Direct ATE', r'$\Delta$ (\%)', 'Corrected ATE', r'$\Delta$ (\%)'], quality)
    text += table(r'RTX composition timings, ms per invocation, frames 0--19 excluded. TR and BA complete-call times are synchronized host wall, including preparation, all updates and finalization. BA iteration total sums optimizer-iteration CUDA-event intervals; it is not loop wall time. TR has ten updates; BA has ten except TUM twenty, every fifth frame. Equal scene weights, three-seed mean $\pm$ SD. Bold gains are reductions relative to matched BASE at the same scope; red negative means slower.',
                  'tab:composition_timing', ['Dataset ($n/N$)', 'Condition', 'TR host', r'Gain (\%)', 'BA host', r'Gain (\%)', 'BA iter. CUDA', r'Gain (\%)'], timing)
    return text


if __name__ == '__main__':
    result, _ = report(Path(__file__).resolve().parents[1])
    print('COMPOSITION_AUDIT', result['completed_runs'], 'runs;', result['complete_scenes'], 'scenes')
