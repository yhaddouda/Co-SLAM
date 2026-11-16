#!/usr/bin/env python3
"""CPU audit: amendment subset, unchanged execution fields, controls and hashes."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path


def validate(root):
    root = Path(root)
    for name, expected in json.loads((root / 'SHA256SUMS.json').read_text()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    source_path = root / 'provenance/ORIGINAL_15W_TARGETS.json'
    original = json.loads(source_path.read_text())
    master = json.loads((root / 'manifests/ALL_15W_TARGETS.json').read_text())
    exclusions = json.loads((root / 'EXCLUSIONS.json').read_text())
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == master['supersedes_ledger_sha256']
    expected = []
    excluded = {}
    for item in original['jobs']:
        if item['seed'] == 1:
            excluded[item['id']] = 'second_seed_removed_by_author'
        elif item['condition'] == 'MC' and item['dataset'] != 'Replica':
            excluded[item['id']] = 'mc_only_replica_scope'
        else:
            job = copy.deepcopy(item)
            if job['condition'] == 'MC_DP' and job['dataset'] != 'Replica':
                del job['control_ids']['MC']
                job['unavailable_comparisons'] = {'MC': '15W_MC_only_outside_Replica_excluded_by_author'}
            job['ordinal'] = len(expected) + 1
            expected.append(job)
    assert master['jobs'] == expected, 'Unexpected change to execution fields/order'
    assert {j['id']: j['reason'] for j in exclusions['jobs']} == excluded
    assert len(original['jobs']) == 297 and len(excluded) == 162
    assert Counter(excluded.values()) == {'second_seed_removed_by_author': 146, 'mc_only_replica_scope': 16}
    jobs = master['jobs']
    assert len(jobs) == len({j['id'] for j in jobs}) == 135
    assert master['benchmark_seeds'] == master['profiling_seeds'] == [0]
    assert all(j['seed'] == 0 for j in jobs)
    assert [p['count'] for p in master['phases']] == [48, 56, 26, 5]
    combined = []
    for phase in master['phases']:
        part = json.loads((root / 'manifests' / (phase['id'] + '.json')).read_text())
        assert len(part['jobs']) == phase['count']
        combined.extend(part['jobs'])
    assert combined == jobs
    assert Counter(j['execution_kind'] for j in jobs) == {'benchmark': 130, 'component_profile': 5}
    coverage = {'Replica': 8, 'Synthetic': 7, 'Tum': 3, 'ScanNet': 6}
    for condition in ('BASE', 'DP', 'MC_DP'):
        native = [j for j in jobs if j['runtime_family'] == 'co_scientific'
                  and j['condition'] == condition and j['hash_size'] != 19]
        assert Counter(j['dataset'] for j in native) == coverage
    mc = [j for j in jobs if j['condition'] == 'MC']
    assert len(mc) == 32 and all(j['dataset'] == 'Replica' for j in mc)
    assert Counter((j['system'], j['hash_size']) for j in mc) == {
        ('Co-SLAM', 16): 8, ('Co-SLAM', 19): 8, ('EC-SLAM', 16): 8, ('EC-SLAM', 19): 8}
    lookup = {j['id']: j for j in jobs}
    for job in jobs:
        for control_id in job.get('control_ids', {}).values():
            control = lookup[control_id]
            keys = ('system', 'dataset', 'scene', 'hash_size', 'seed')
            assert all(job[k] == control[k] for k in keys)
            assert control['ordinal'] < job['ordinal']
    print('VERIFIED: 135 full slots; seed 0 only; Replica MC H16/H19; 162 exclusions; execution fields, coverage, controls and hashes passed')


if __name__ == '__main__':
    validate(Path(__file__).resolve().parent)
