#!/usr/bin/env python3
"""CPU-only coverage, ordering, comparison-dependency and package audit."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def validate(root):
    root=Path(root)
    for name,expected in json.loads((root/'SHA256SUMS.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    master=json.loads((root/'manifests/ALL_15W_TARGETS.json').read_text())
    jobs=master['jobs']
    assert len(jobs)==297 and len({j['id'] for j in jobs})==297
    counts=[48,72,26,5,48,72,26]
    assert [p['count'] for p in master['phases']]==counts
    combined=[]
    for phase,count in zip(master['phases'],counts):
        part=json.loads((root/'manifests'/(phase['id']+'.json')).read_text())
        assert len(part['jobs'])==count
        assert all(j['phase']==phase['id'] and j['priority']==phase['priority'] for j in part['jobs'])
        combined.extend(part['jobs'])
    assert combined==jobs
    assert [j['ordinal'] for j in jobs]==list(range(1,298))
    assert all(j['seed']==0 for j in jobs[:151])
    assert all(j['seed']==1 and j['execution_kind']=='benchmark' for j in jobs[151:])
    benchmark=[j for j in jobs if j['execution_kind']=='benchmark']
    profiles=[j for j in jobs if j['execution_kind']=='component_profile']
    assert len(benchmark)==292 and len(profiles)==5
    assert Counter(j['seed'] for j in benchmark)=={0:146,1:146}
    assert Counter(j['system'] for j in profiles)=={'Co-SLAM':3,'EC-SLAM':2}
    assert all(j['seed']==0 for j in profiles)
    expected={'Replica':8,'Synthetic':7,'Tum':3,'ScanNet':6}
    for seed in (0,1):
        native=[j for j in benchmark if j['seed']==seed and j['runtime_family']=='co_scientific'
                and not(j['dataset']=='Replica' and j['hash_size']==19)]
        assert len(native)==96
        for condition in ('BASE','DP','MC','MC_DP'):
            part=[j for j in native if j['condition']==condition]
            assert Counter(j['dataset'] for j in part)==expected
            assert len({(j['dataset'],j['scene']) for j in part})==24
        for system in ('Co-SLAM','EC-SLAM'):
            h19=[j for j in benchmark if j['seed']==seed and j['system']==system and j['hash_size']==19]
            assert len(h19)==16 and Counter(j['condition'] for j in h19)=={'BASE':8,'MC':8}
        native_ec=[j for j in benchmark if j['seed']==seed and j['system']=='EC-SLAM' and j['hash_size']==16]
        assert len(native_ec)==16 and Counter(j['condition'] for j in native_ec)=={'BASE':8,'MC':8}
        eng=[j for j in benchmark if j['seed']==seed and j['runtime_family']=='co_engineering_validated']
        assert len(eng)==2 and {j['condition'] for j in eng}=={'BASE_CONTROL','ENG'}
        assert all(j['system']=='Co-SLAM' and j['reported_comparison_stage']=='bundle_adjustment' for j in eng)
    lookup={j['id']:j for j in jobs}
    for job in jobs:
        assert job['frames']==0 and job['quality'] is False and job['export_mesh'] is False
        assert job['warmup_frames']==20 and job['expected_native_frames']>20
        assert job['power_mode_required']=='AUTHOR_SELECTED_15W_EXACT_LOCAL_NAME'
        for control_id in job.get('control_ids',{}).values():
            c=lookup[control_id]
            assert (c['system'],c['dataset'],c['scene'],c['hash_size'],c['seed'])==(
                job['system'],job['dataset'],job['scene'],job['hash_size'],job['seed'])
            # Reversed BASE/DP seed-1 order is allowed within the same phase.
            assert c['priority']<=job['priority']
    print('15W_PLAN_VERIFIED: 297 unique full slots; seed 0 first (146 benchmarks +5 profiles), then seed 1 (146 benchmarks); 24-scene DP/MC coverage and controls, package hashes passed')


if __name__=='__main__':validate(Path(__file__).resolve().parent)
