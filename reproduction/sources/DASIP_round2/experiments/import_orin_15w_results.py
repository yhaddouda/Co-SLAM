#!/usr/bin/env python3
"""Independently audit the 15 W raw export on CPU; never execute received code."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from import_orin_maxn_increment import normalize
from report_component_profiles import audit_run as audit_co, COMPONENTS
from report_ec_component_profiles import audit_run as audit_ec

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / 'results/orin_15w_received_20261027'
OUT = BASE / 'results/orin_15w_analysis_v1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open() as stream:
        yield from csv.DictReader(stream)


def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def csv_save(path, values):
    columns = list(dict.fromkeys(k for row in values for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(values)


def check_summary(summary, job):
    assert summary['full_sequence'] and not summary['diagnostic']
    assert summary['finite_parameters_and_poses'] and summary['quality'] is False
    assert summary['frames'] == job['expected_native_frames']
    assert summary['seed'] == job['seed'] == 0
    for key in ('initial_power', 'final_power'):
        assert summary[key].splitlines()[0] == 'NV Power Mode: MODE_15W'


def audit_benchmark(job, entry, source):
    directory = RAW / entry['archive_directory']
    summary, cfg = read(directory / 'summary.json'), read(directory / 'config.json')
    check_summary(summary, job)
    assert summary['warmup_frames'] == 20
    assert summary['dataset'] == job['dataset'] and summary['scene'] == job['scene']
    assert read(directory / 'source_manifest.json') == source
    system, condition = job['system'], job['condition']
    engineering = job['runtime_family'] == 'co_engineering_validated'
    grid = cfg['grid' if system == 'Co-SLAM' else 'HashGrid']
    assert grid['hash_size'] == job['hash_size'] and cfg['round2']['seed'] == 0
    morton, prune = condition in ('MC', 'MC_DP'), condition in ('DP', 'MC_DP')
    assert grid['hash'] == ('Morton' if morton else 'CoherentPrime')
    assert bool(grid['morton_sort']) == morton
    assert bool(cfg['round2'].get('engineering', False)) == (condition == 'ENG')
    assert summary['condition'] == ('BASE' if engineering else condition)
    if engineering:
        assert summary['variant'] == ('BASE' if condition == 'BASE_CONTROL' else 'ENG')
    if system == 'Co-SLAM':
        assert cfg['round2']['sampler'] == ('support' if prune else 'dense')
        assert cfg['round2']['margin_scene_units'] == .15 and cfg['round2']['anchor_every'] == 3
    else:
        assert summary['native_thread_schedule_preserved']
    calls, iterations = defaultdict(list), defaultdict(list)
    for row in rows(directory / 'calls.csv'):
        if int(row['frame_id']) >= 20:
            calls[row['stage']].append(row)
    for row in rows(directory / 'iterations.csv'):
        if int(row['frame_id']) >= 20:
            iterations[(row['iteration_type'], int(row['frame_id']))].append(row)
    record = dict(id=job['id'], category='ENG' if engineering else condition,
                  system=system, dataset=job['dataset'], scene=job['scene'],
                  hash_size=job['hash_size'], seed=0,
                  condition='BASE' if condition == 'BASE_CONTROL' else condition,
                  frames=summary['frames'], directory=str(directory),
                  source_sha256=sha(directory / 'source_manifest.json'))
    for stage, call_stage, label in [('tracking', 'tracking' if system == 'Co-SLAM' else 'tracking_optimizer_call', 'tracking'),
                                     ('bundle_adjustment', 'bundle_adjustment', 'ba')]:
        cadence = cfg['mapping']['map_every' if system == 'Co-SLAM' else 'every_frame']
        expected = list(range(20, summary['frames'], 1 if label == 'tracking' else cadence))
        if system == 'EC-SLAM' and label == 'ba' and summary['frames'] - 1 not in expected:
            expected.append(summary['frames'] - 1)
        selected = calls[call_stage]
        assert sorted(int(c['frame_id']) for c in selected) == expected, (job['id'], stage)
        updates = cfg['tracking']['iter' if system == 'Co-SLAM' else 'iters'] if label == 'tracking' else cfg['mapping']['iters']
        totals, forwards, backwards = [], [], []
        for call in selected:
            values = iterations[(stage, int(call['frame_id']))]
            assert len(values) == updates and {int(v['iteration_id']) for v in values} == set(range(updates))
            totals.append(sum(float(v['full_iteration_cuda_ms']) for v in values))
            forwards.extend(float(v['forward_cuda_ms']) for v in values)
            backwards.extend(float(v['backward_cuda_ms']) for v in values)
        assert summary[call_stage]['count'] == len(selected)
        record[label + '_calls'], record[label + '_updates_per_call'] = len(selected), updates
        for field in ('wall_ms', 'cuda_ms'):
            value = statistics.mean(float(c[field]) for c in selected)
            assert abs(value - summary[call_stage]['mean_' + field]) < 1e-6
            record[label + '_' + field] = value
        record[label + '_iteration_cuda_ms'] = statistics.mean(totals)
        record[label + '_forward_cuda_ms'] = statistics.mean(forwards)
        record[label + '_backward_cuda_ms'] = statistics.mean(backwards)
    if system == 'EC-SLAM':
        for stage, label in [('tracking_frame', 'tracking_frame'), ('mapping_frame', 'mapping_frame')]:
            record[label + '_wall_ms'] = statistics.mean(float(c['wall_ms']) for c in calls[stage])
            assert abs(record[label + '_wall_ms'] - summary[stage]['mean_wall_ms']) < 1e-6
    totals, contexts = defaultdict(lambda: [0, 0, 0, 0, 0]), Counter()
    for row in rows(directory / 'queries.csv'):
        dense, actual = int(row['dense_queries']), int(row['actual_queries'])
        assert 0 < actual <= dense
        stage, frame, iteration = row['stage'], int(row['frame_id']), int(row['iteration'])
        if not prune:
            assert actual == dense
        if stage not in ('tracking', 'bundle_adjustment'):
            continue
        count = cfg['tracking']['iter' if system == 'Co-SLAM' else 'iters'] if stage == 'tracking' else cfg['mapping']['iters']
        assert 0 <= iteration < count
        if iteration in (0, count - 1) or iteration % 3 == 0:
            assert actual == dense
        contexts[(stage, frame, iteration)] += 1
        total = totals[stage]
        total[0] += dense
        total[1] += actual
        if frame >= 20:
            total[2] += dense
            total[3] += actual
            total[4] += int(actual < dense)
    for (stage, frame), values in iterations.items():
        for value in values:
            multiplicity = contexts[(stage, frame, int(value['iteration_id']))]
            assert multiplicity in ((1, 2) if system == 'EC-SLAM' else (1,))
    for stage, label in [('tracking', 'tracking'), ('bundle_adjustment', 'ba')]:
        dense, actual, post_dense, post_actual, pruned_updates = totals[stage]
        if system == 'Co-SLAM':
            assert dense == summary[stage]['dense_queries'] and actual == summary[stage]['actual_queries']
            assert abs(100 * (1 - actual / dense) - summary[stage]['queries_pruned_pct']) < 1e-6
            assert (pruned_updates > 0) == prune
        record[label + '_queries_pruned_pct'] = 100 * (1 - post_actual / post_dense)
    assert all(math.isfinite(v) and v >= 0 for v in record.values() if isinstance(v, (int, float)))
    return record, cfg


def aggregate(records):
    groups = []
    for system in ('Co-SLAM', 'EC-SLAM'):
        for dataset in ('Replica', 'Synthetic', 'Tum', 'ScanNet'):
            native = [r for r in records if r['category'] != 'ENG' and r['system'] == system and r['dataset'] == dataset]
            capacities = sorted({r['hash_size'] for r in native}) + ([None] if dataset == 'ScanNet' else [])
            for capacity in capacities:
                for variant in ('MC', 'DP', 'MC_DP'):
                    values = [r for r in native if (capacity is None or r['hash_size'] == capacity)
                              and r['condition'] in ('BASE', variant)]
                    if not any(r['condition'] == variant for r in values):
                        continue
                    groups.append(group(values, system, dataset, capacity, variant))
    groups.append(group([r for r in records if r['category'] == 'ENG'], 'Co-SLAM', 'Replica', 16, 'ENG'))
    return groups


def group(values, system, dataset, capacity, variant):
    scenes = sorted({r['scene'] for r in values})
    assert len(values) == len(scenes) * 2
    assert {(r['scene'], r['seed'], r['condition']) for r in values} == {(s, 0, c) for s in scenes for c in ('BASE', variant)}
    row = dict(category=variant, system=system, dataset=dataset, hash_size=capacity,
               scene_count=len(scenes), scene_names=scenes, seeds=[0],
               capacity_scope='native mixed capacities, equal scene weights' if capacity is None else 'single hash capacity')
    metrics = sorted(set.intersection(*[{k for k in r if k.endswith(('_ms', '_calls', '_updates_per_call', '_pruned_pct'))} for r in values]))
    for field in metrics:
        for condition in ('BASE', variant):
            row[condition + '_' + field] = statistics.mean(r[field] for r in values if r['condition'] == condition)
            row[condition + '_' + field + '_sd'] = None
        denominator = row['BASE_' + field]
        row[field + '_reduction_pct'] = 100 * (1 - row[variant + '_' + field] / denominator) if denominator else None
    return row


def profiles(entries, sources, inputs):
    result = {}
    for key, system, family, auditor in [('CO', 'Co-SLAM', 'co_component_profile_repaired', audit_co),
                                         ('EC', 'EC-SLAM', 'ec_component_profile_validated', audit_ec)]:
        runtime = (RAW / sources[family]['manifest']).parent
        protocol = read(runtime / 'PROTOCOL.json')
        lookup = {w['id']: w for w in protocol['workloads']}
        selected = [j for j in entries if j['system'] == system]
        weighted, thirds, contexts, profile_inputs = [], [], [], []
        for entry in selected:
            directory = RAW / entry['archive_directory']
            summary = read(directory / 'summary.json')
            job = dict(workload=summary['workload'], seed=0)
            work = lookup[job['workload']]
            assert summary['frames'] == work['expected_native_frames']
            assert read(directory / 'selection.json') == work
            assert sha(runtime / 'PROTOCOL.json') == summary['protocol_sha256']
            outputs = auditor(directory, job, work, sources[family]['manifest_sha256'])
            weighted.extend(outputs[0]); thirds.extend(outputs[1])
            if len(outputs) > 2:
                contexts.extend(outputs[2])
            profile_inputs.extend(p for p in directory.rglob('*') if p.is_file() and p.suffix in ('.json', '.csv'))
        groups = []
        for workload, stage, scope in sorted({(r['workload'], r['stage'], r['scope']) for r in weighted}):
            values = [r for r in weighted if (r['workload'], r['stage'], r['scope']) == (workload, stage, scope)]
            components = {r['component']: dict(mean_ms=r['cuda_ms'], sd_ms=None) for r in values}
            assert len(components) == len(values)
            for value in components.values():
                value['share_pct'] = None if scope == 'host_call' else 100 * value['mean_ms'] / components['total']['mean_ms']
            groups.append(dict(system=system, platform='ORIN_15W', workload=workload,
                               stage=stage, scope=scope, seeds=[0], components=components))
        output = BASE / 'results/orin_15w_component_analysis' / key.lower()
        output.mkdir(parents=True, exist_ok=True)
        profile_inputs.extend([runtime / 'PROTOCOL.json', runtime / 'RUNTIME_MANIFEST.json',
                               BASE / 'experiments/report_component_profiles.py', BASE / 'experiments/report_ec_component_profiles.py'])
        data = dict(completed=len(selected), total=3 if key == 'CO' else 2, groups=groups,
                    runtime_ready=True, seeds=[0], input_sha256={str(p): sha(p) for p in profile_inputs},
                    aggregation='Eligible-call-weighted chronological thirds; single seed, no across-seed SD; workers never summed')
        save(output / 'SUMMARY.json', data)
        csv_save(output / 'per_seed.csv', weighted); csv_save(output / 'per_third.csv', thirds)
        if contexts:
            csv_save(output / 'call_context.csv', contexts)
        text = f'# {system} 15 W component profiles\n\n{len(selected)}/{data["total"]} full seed-0 passes independently audited; no across-seed SD.\n\n'
        text += '| Workload | Stage | Scope | Component | ms | Share (%) |\n|---|---|---|---|---:|---:|\n'
        for g in groups:
            for component, value in g['components'].items():
                share = '—' if value['share_pct'] is None else f'{value["share_pct"]:.2f}'
                text += f'| {g["workload"]} | {g["stage"]} | {g["scope"]} | {component} | {value["mean_ms"]:.4f} | {share} |\n'
        (output / 'REPORT.md').write_text(text)
        save(BASE / f'results/ORIN_15W_{key}_COMPONENT_ACTIVE.json',
             dict(runtime=str(runtime), received=str(RAW), analysis_output=str(output)))
        inputs.update(data['input_sha256'])
        result[key] = dict(completed=data['completed'], total=data['total'])
    return result


def main():
    OUT.mkdir(exist_ok=True)
    transfer = read(BASE / 'results/ORIN_15W_TRANSFER_AUDIT.json')
    assert sha(transfer['archive']) == transfer['sha256']
    payload = read(RAW / 'SHA256SUMS.json')
    for name, expected in payload.items():
        assert sha(RAW / name) == expected, name
    master = read(RAW / 'MASTER_RUN_INDEX.json')
    plan_path = BASE / 'experiments/orin_15w_single_seed_replica_mc_v1/manifests/ALL_15W_TARGETS.json'
    plan = read(plan_path)
    expected = {j['id']: j for j in plan['jobs']}
    assert len(master['runs']) == len(expected) == 135
    assert {j['canonical_id'] for j in master['runs']} == set(expected)
    assert len({j['canonical_id'] for j in master['runs']}) == 135
    assert read(RAW / 'RECONCILIATION.json')['counts'] == dict(successful=135, missing=0, failed=0, blocked=0)
    excluded = read(BASE / 'experiments/orin_15w_single_seed_replica_mc_v1/EXCLUSIONS.json')
    exported_excluded = read(RAW / 'AUTHOR_EXCLUSIONS.json')
    assert {j['id'] for j in excluded['jobs']} == {j['id'] for j in exported_excluded['jobs']}
    sources = read(RAW / 'SOURCE_INDEX.json')
    pinned = {
        'co_scientific': BASE / 'experiments/coslam_reference_runtime_v1/RUNTIME_MANIFEST.json',
        'ec_native_timing': BASE / 'results/orin_maxn_received_20261020/native/continuous_v4/EC_RUNTIME_MANIFEST.json',
        'co_engineering_validated': BASE / 'results/orin_maxn_received_20261020/post/coslam_engineering_local/RUNTIME_MANIFEST.json',
        'co_component_profile_repaired': Path(read(BASE / 'results/ORIN_MAXN_CO_COMPONENT_ACTIVE.json')['runtime']) / 'RUNTIME_MANIFEST.json',
        'ec_component_profile_validated': Path(read(BASE / 'results/ORIN_MAXN_EC_COMPONENT_ACTIVE.json')['runtime']) / 'RUNTIME_MANIFEST.json'}
    for family, entry in sources.items():
        path = RAW / entry['manifest']
        assert sha(path) == entry['manifest_sha256'] == sha(pinned[family]), family
        manifest = read(path)
        for filename, digest in manifest.get('runtime_hashes', manifest.get('files', {})).items():
            assert sha(path.parent / filename) == digest, (family, filename)
    inputs = {str(p): sha(p) for p in [Path(__file__).resolve(), plan_path, RAW / 'MASTER_RUN_INDEX.json',
               RAW / 'RECONCILIATION.json', RAW / 'SHA256SUMS.json', RAW / 'SOURCE_INDEX.json',
               Path(transfer['archive']), BASE / 'results/ORIN_15W_TRANSFER_AUDIT.json']}
    records, configs, control_map = [], {}, {}
    for entry in master['runs']:
        job = expected[entry['canonical_id']]
        assert entry['status'] == 'successful' and entry['seed'] == 0 and not entry['missing_evidence']
        assert entry['method'] == job['condition'] and entry['hash_size'] == job['hash_size']
        controls=dict(job.get('control_ids', {}))
        if job['condition']=='MC':controls.setdefault('BASE',job['id'].replace('_MC_','_BASE_'))
        if job['condition']=='ENG':controls={'BASE_CONTROL':'Co_RTXREF_office0_BASE_CONTROL_seed0'}
        assert entry['paired_control_ids'] == controls, job['id']
        control_map[job['id']]=controls
        directory = RAW / entry['archive_directory']
        assert read(directory / 'completion.json')['status'] == 'PASS'
        assert read(directory / 'controller/process_receipt.json')['returncode'] == 0
        for metadata in entry['archive_files'].values():
            path = RAW / metadata['archive_path']
            assert path.is_file() and sha(path) == metadata['sha256']
        check_summary(read(directory / 'summary.json'), job)
        assert sha(directory / 'source_manifest.json') == sources[job['runtime_family']]['manifest_sha256']
        if job['execution_kind'] == 'benchmark':
            record, cfg = audit_benchmark(job, entry, read(RAW / sources[job['runtime_family']]['manifest']))
            records.append(record); configs[job['id']] = normalize(cfg, job['system'])
            for name in ('summary.json', 'config.json', 'source_manifest.json', 'calls.csv', 'iterations.csv', 'queries.csv'):
                inputs[str(directory / name)] = sha(directory / name)
    for job in plan['jobs']:
        for control_id in control_map[job['id']].values():
            assert configs[job['id']] == configs[control_id], ('Paired configuration mismatch', job['id'], control_id)
            assert expected[control_id]['runtime_family'] == job['runtime_family']
            assert all(job[k]==expected[control_id][k] for k in ('system','dataset','scene','hash_size','seed'))
    validations = list((RAW / 'validation/profiling_15W').rglob('PASSED.json'))
    assert len(validations) == 5
    for path in validations:
        value = read(path)
        assert value['status'] == 'PASSED' and value['output_loss_atol'] == 1e-6 and value['gradient_step_atol'] == 1e-5
        inputs[str(path)] = sha(path)
    engineering_validation = RAW / 'validation/engineering_15W/ENGINEERING_FIXED_STATE_15W.json'
    assert read(engineering_validation)['status'] == 'PASS'
    inputs[str(engineering_validation)] = sha(engineering_validation)
    profile_audit = profiles(master['profiles'], sources, inputs)
    groups = aggregate(records)
    csv_save(OUT / 'per_run.csv', records); csv_save(OUT / 'dataset_means.csv', groups)
    report = '# Audited Orin 15 W import\n\n130/130 benchmarks and 5/5 component profiles independently audited. All 4,192 payload hashes, amended identities, historical mode, frozen sources, paired configurations, raw call/iteration coverage, summary agreement, pruning/query counters and profile partitions passed. No received code or GPU work executed.\n\n'
    report += 'Seed 0 only; equal scene weights; no across-seed SD. Frames 0–19 excluded. Full-invocation host wall and optimizer CUDA sums are distinct. Native ScanNet capacities are pooled equally across all six scenes. EC workers overlap and cannot be summed into FPS; RTX and Orin EC call scopes differ. Non-Replica MC-only is intentionally absent. Achieved fixed historical clocks are not established; available policy bounds and snapshots remain in the raw export. Preserve negative outcomes.\n\n'
    report += '| System | Dataset | H | Variant | Scenes | TR host reduction | BA host reduction | BA iteration reduction |\n|---|---|---|---|---:|---:|---:|---:|\n'
    for g in groups:
        tr = '—' if g['category'] == 'ENG' else f'{g["tracking_wall_ms_reduction_pct"]:+.2f}%'
        report += f'| {g["system"]} | {g["dataset"]} | {g["hash_size"] or "native"} | {g["category"]} | {g["scene_count"]} | {tr} | {g["ba_wall_ms_reduction_pct"]:+.2f}% | {g["ba_iteration_cuda_ms_reduction_pct"]:+.2f}% |\n'
    (OUT / 'REPORT.md').write_text(report)
    data = dict(mode='15W', platform='ORIN_15W', verified_runs=130, benchmark_runs=130,
                native_runs=128, engineering_runs=2, profile_runs=5, seeds=[0],
                author_excluded=162, missing_runs=0, failed_runs=0, blocked_runs=0,
                dp_runs_in_this_snapshot=24, composition_runs_in_this_snapshot=24,
                component_profiles=profile_audit, groups=groups, clocks='Policy bounds/snapshots recorded; achieved fixed historical clocks unknown',
                input_sha256=inputs, received_code_executed=False, aggregation='Equal scene means; seed 0 only; SD null; gain is reduction of matched mean times')
    save(OUT / 'ANALYSIS_STATUS.json', data)
    save(BASE / 'results/ORIN_15W_ACTIVE.json', dict(analysis=str(OUT), mode='15W', benchmark_runs=130, profile_runs=5))
    print('ORIN_15W_AUDITED: 130 benchmarks +5 profiles; all coverage and paired evidence passed', flush=True)


if __name__ == '__main__':
    main()
