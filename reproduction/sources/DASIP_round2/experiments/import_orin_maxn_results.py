#!/usr/bin/env python3
"""Audit the received MAXN snapshot and aggregate its original raw CSVs on CPU."""
import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import zipfile

BASE = Path(__file__).resolve().parents[1]
RECEIVED = BASE / 'results/orin_maxn_received_20261020'
OUT = BASE / 'results/orin_maxn_analysis'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open() as stream:
        yield from csv.DictReader(stream)


def resolve(job):
    original = job['output']
    if job['status'] == 'reused':
        return RECEIVED / 'native/reused' / Path(original).name
    for token, local in [('orin_continuous_v4/', 'native/continuous_v4'),
                         ('orin_after_v4_maxn_v1/', 'post')]:
        if token in original:
            return RECEIVED / local / original.split(token, 1)[1]
    raise ValueError(original)


def normalized(config, system):
    config = copy.deepcopy(config)
    for key in ('datadir', 'output', 'exp_name', 'input_folder'):
        config.get('data', {}).pop(key, None)
    for key in ('condition', 'seed', 'engineering'):
        config.get('round2', {}).pop(key, None)
    for key in ('hash', 'morton_sort'):
        config['grid' if system == 'Co-SLAM' else 'HashGrid'].pop(key, None)
    return config


def main():
    OUT.mkdir(exist_ok=True)
    receipt = read(RECEIVED / 'RECEIPT.json')
    assert sha(Path(receipt['archive'])) == receipt['archive_sha256']
    declared = read(RECEIVED / 'transfer/RTX_IMPORT_MANIFEST.json')
    jobs = declared['completed_identity_mapping']
    assert len(jobs) == 172 and len({j['id'] for j in jobs}) == 172
    assert all(j['required_for_completion'] and j['status'] in ('done', 'reused') for j in jobs)
    inputs = {str(RECEIVED / 'RECEIPT.json'): sha(RECEIVED / 'RECEIPT.json')}
    inputs[receipt['archive']] = receipt['archive_sha256']
    inputs[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    inventory = read(BASE / 'experiments/ALL_CONFIGURED_NONREPLICA_INVENTORY.json')
    lengths = {(r['dataset'], r['scene']): r['frames'] for r in inventory['scenes']}
    native_manifest = read(BASE / 'experiments/coslam_reference_runtime_v1/RUNTIME_MANIFEST.json')
    native_zip = RECEIVED / 'native/continuous_v4/orin_coslam_runtime_v1.zip'
    assert sha(native_zip) == sha(BASE/'orin_coslam_runtime_v1.zip')
    with zipfile.ZipFile(native_zip) as runtime_zip:
        shipped_native = json.loads(runtime_zip.read('runtime/RUNTIME_MANIFEST.json'))
        for name,value in shipped_native['runtime_hashes'].items():
            assert hashlib.sha256(runtime_zip.read('runtime/'+name)).hexdigest() == value,name
    shipped_engineering = read(RECEIVED/'post/coslam_engineering_local/RUNTIME_MANIFEST.json')
    for name,value in shipped_engineering['runtime_hashes'].items():
        assert sha(RECEIVED/'post/coslam_engineering_local'/name) == value,name
    ec_source = RECEIVED / 'native/continuous_v4'
    ec_manifest = read(ec_source / 'EC_RUNTIME_MANIFEST.json')
    for name, expected in ec_manifest['files'].items():
        assert sha(ec_source / name) == expected, name
    records = []; configs = {}; resolved = set(); source_ids = defaultdict(set)
    for job in jobs:
        directory = resolve(job)
        assert directory.is_dir(), directory
        assert directory not in resolved, ('Repeated raw run', directory)
        resolved.add(directory)
        summary = read(directory / 'summary.json'); cfg = read(directory / 'config.json')
        source = read(directory / 'source_manifest.json')
        system = job['system']; category = 'ENG' if job['condition'] in ('ENG', 'BASE_CONTROL') else 'MC'
        condition = 'BASE' if job['condition'] == 'BASE_CONTROL' else job['condition']
        assert summary['full_sequence'] and not summary['diagnostic'] and summary['finite_parameters_and_poses']
        assert not summary['quality'] and summary['warmup_frames'] == 20
        assert summary['dataset'] == job['dataset'] and summary['scene'] == job['scene'] and summary['seed'] == job['seed']
        assert summary['initial_power'].splitlines()[0].strip() == 'NV Power Mode: MAXN'
        assert summary['final_power'].splitlines()[0].strip() == 'NV Power Mode: MAXN'
        frames = summary['frames']; assert frames == (2000 if job['dataset'] == 'Replica' else lengths[(job['dataset'], job['scene'])])
        assert cfg['round2']['seed'] == job['seed']
        grid = cfg['grid' if system == 'Co-SLAM' else 'HashGrid']
        assert grid['hash_size'] == job['hash_size']
        assert grid['hash'] == ('Morton' if condition == 'MC' else 'CoherentPrime')
        assert bool(grid['morton_sort']) == (condition == 'MC')
        assert bool(cfg['round2'].get('engineering', False)) == (condition == 'ENG')
        if system == 'Co-SLAM':
            assert source == (shipped_engineering if category == 'ENG' else shipped_native)
            assert source['public_revision'] == native_manifest['public_revision']
            assert source['public_file_hashes'] == native_manifest['public_file_hashes']
            # Estimator, losses, datasets and geometry encoding are the pinned path.
            for name in ('coslam.py', 'model/scene_rep.py', 'model/encodings.py', 'datasets/dataset.py'):
                assert source['runtime_hashes'][name] == native_manifest['runtime_hashes'][name], name
            assert summary['condition'] == ('BASE' if category == 'ENG' else condition)
            assert summary.get('engineering', False) == (condition == 'ENG')
            if category == 'ENG': assert summary['variant'] == condition
            stages = [('tracking', 'tracking', 'tracking'), ('bundle_adjustment', 'bundle_adjustment', 'ba')]
        else:
            assert source['public_revision'] == ec_manifest['public_revision']
            # Reused v2 runs have an older wrapper; native estimator/timer hashes agree.
            for name, value in ec_manifest['files'].items():
                if name != 'run_ec_round2.py': assert source['files'][name] == value, name
            assert summary['condition'] == condition and summary['hash_size'] == job['hash_size']
            assert summary['native_thread_schedule_preserved']
            stages = [('tracking_optimizer_call', 'tracking', 'tracking'), ('bundle_adjustment', 'bundle_adjustment', 'ba')]
        key = (category, system, job['dataset'], job['scene'], job['hash_size'])
        canon = normalized(cfg, system)
        if key in configs: assert canon == configs[key], ('Configuration mismatch', job['id'])
        configs[key] = canon
        source_ids[(category, system)].add(sha(directory / 'source_manifest.json'))
        calls = defaultdict(list)
        for row in csv_rows(directory / 'calls.csv'):
            if int(row['frame_id']) >= 20: calls[row['stage']].append(row)
        iterations = defaultdict(list)
        for row in csv_rows(directory / 'iterations.csv'):
            if int(row['frame_id']) >= 20: iterations[(row['iteration_type'], int(row['frame_id']))].append(row)
        record = dict(id=job['id'], category=category, system=system, dataset=job['dataset'], scene=job['scene'],
                      hash_size=job['hash_size'], seed=job['seed'], condition=condition, status=job['status'],
                      frames=frames, directory=str(directory.relative_to(BASE)), source_sha256=sha(directory / 'source_manifest.json'))
        for call_stage, iteration_stage, label in stages:
            selected = calls[call_stage]
            expected_frames = list(range(20, frames)) if label == 'tracking' else list(range(20, frames, cfg['mapping']['map_every'] if system == 'Co-SLAM' else cfg['mapping']['every_frame']))
            if system == 'EC-SLAM' and label == 'ba' and frames - 1 not in expected_frames: expected_frames.append(frames - 1)
            assert sorted(int(c['frame_id']) for c in selected) == expected_frames, (job['id'], call_stage)
            updates = cfg['tracking']['iter' if system == 'Co-SLAM' else 'iters'] if label == 'tracking' else cfg['mapping']['iters']
            totals = []; forwards = []; backwards = []
            for call in selected:
                values = iterations[(iteration_stage, int(call['frame_id']))]
                assert len(values) == updates and {int(v['iteration_id']) for v in values} == set(range(updates)), (job['id'],call_stage,call['frame_id'])
                totals.append(sum(float(v['full_iteration_cuda_ms']) for v in values))
                forwards.extend(float(v['forward_cuda_ms']) for v in values)
                backwards.extend(float(v['backward_cuda_ms']) for v in values)
            record[label + '_calls'] = len(selected)
            record[label + '_updates_per_call'] = updates
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
        for query in csv_rows(directory / 'queries.csv'):
            assert int(query['dense_queries']) == int(query['actual_queries']) > 0
        for name in ('summary.json', 'config.json', 'source_manifest.json', 'calls.csv', 'iterations.csv', 'queries.csv'):
            path = directory / name; inputs[str(path)] = sha(path)
        assert all(math.isfinite(v) and v >= 0 for v in record.values() if isinstance(v, (float, int)))
        records.append(record)
    grouped = defaultdict(list)
    for r in records: grouped[(r['category'], r['system'], r['dataset'], r['hash_size'])].append(r)
    groups = []
    for (category, system, dataset, capacity), values in grouped.items():
        alternative = category; scenes = sorted({r['scene'] for r in values})
        assert len(values) == len(scenes) * 4
        assert {(r['scene'], r['seed'], r['condition']) for r in values} == {(s, seed, c) for s in scenes for seed in (0, 1) for c in ('BASE', alternative)}
        row = dict(category=category, system=system, dataset=dataset, hash_size=capacity, scene_count=len(scenes), scene_names=scenes, seeds=[0,1])
        metrics = [k for k in values[0] if k.endswith(('_ms', '_calls', '_updates_per_call'))]
        for field in metrics:
            for condition in ('BASE', alternative):
                means = [statistics.mean(r[field] for r in values if r['seed'] == seed and r['condition'] == condition) for seed in (0,1)]
                row[condition + '_' + field] = statistics.mean(means)
                row[condition + '_' + field + '_sd'] = statistics.stdev(means)
            row[field + '_reduction_pct'] = 100 * (1-row[alternative + '_' + field] / row['BASE_' + field])
        groups.append(row)
    for name, values in [('per_run.csv',records), ('dataset_means.csv',groups)]:
        columns = list(dict.fromkeys(k for row in values for k in row))
        with (OUT/name).open('w',newline='') as f:
            writer = csv.DictWriter(f,columns); writer.writeheader(); writer.writerows(values)
    status = dict(archive_sha256=receipt['archive_sha256'], verified_runs=len(records), native_runs=168, engineering_runs=4,
                  reused_runs=sum(r['status']=='reused' for r in records), seeds=[0,1], mode='MAXN',
                  aggregation='mean within run, equal scene weights within seed, two-seed mean and sample SD; ratios of matched means',
                  ec_scope='Orin tracking_optimizer_call / optimize_map method calls; excludes outer frame preparation/finalization. RTX active worker-body timing has different boundaries; no direct EC cross-hardware latency ratio.',
                  clocks='Author-selected policy retained; full jetson_clocks lock state unknown',
                  pending_scannet_runs=24, dp_runs_in_this_snapshot=0, composition_runs_in_this_snapshot=0,
                  excluded='EC engineering', groups=groups, input_sha256=inputs)
    (OUT/'ANALYSIS_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    report = '# Audited Orin MAXN import\n\n172 full runs: 168 native BASE/MC and four Co-SLAM BASE-control/ENG runs. Sixteen reused outputs are present and counted once. All transfer and nested-payload SHA-256 checks passed. Full call coverage, iteration counts/IDs, source identities, paired effective configurations, seeds, MAXN records and raw-to-summary agreement were checked. No GPU work or received code was executed.\n\n'
    report += 'Means give scenes equal weight; sample SD is across the two seed-wise scene means. All stage averages exclude frames 0–19. H19 is separate. The two dedicated engineering BASE controls are distinct from the native BASE population.\n\n'
    report += '**EC scope:** Orin optimizer-method call times exclude outer frame preparation/finalization; outer frame timers separately include native waits. RTX EC measures a larger active worker body with handshake waits removed. These are explicitly different boundaries. Within-platform BASE/MC comparisons remain paired; no direct EC hardware speed ratio or FPS is inferred. Native thread/shared-stream overlap can affect intervals.\n\n'
    report += '| Comparison | System | Dataset | H | Scenes | BASE TR ms | Variant TR ms | TR reduction | BASE BA/Map ms | Variant BA/Map ms | Reduction |\n|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in groups:
        c=r['category']
        report += f"| {c} | {r['system']} | {r['dataset']} | {r['hash_size']} | {r['scene_count']} | {r['BASE_tracking_wall_ms']:.3f} | {r[c+'_tracking_wall_ms']:.3f} | {r['tracking_wall_ms_reduction_pct']:+.2f}% | {r['BASE_ba_wall_ms']:.3f} | {r[c+'_ba_wall_ms']:.3f} | {r['ba_wall_ms_reduction_pct']:+.2f}% |\n"
    report += '\nScanNet MAXN is absent from this snapshot and is now running according to the author. No full DP/MC+DP results are included: the v4 execution notes refer to a separate earlier MAXN archive. Retrieve that compatible evidence before requesting repeats. Disjoint component and kernel A/B/C/D profiling is not provided here. Co engineering is the only engineering comparison. Recorded clocks do not establish locked frequencies.\n'
    (OUT/'REPORT.md').write_text(report)
    print('ORIN_MAXN_AUDITED',len(records),'runs;',len(groups),'dataset comparisons')
    for r in groups:
        print(r['category'],r['system'],r['dataset'],r['hash_size'],r['scene_count'],f"TR {r['tracking_wall_ms_reduction_pct']:+.2f}% BA/Map {r['ba_wall_ms_reduction_pct']:+.2f}%")


if __name__ == '__main__': main()
