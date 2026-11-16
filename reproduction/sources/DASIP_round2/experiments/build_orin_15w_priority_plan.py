#!/usr/bin/env python3
"""Prepare the complete author-prioritized 15 W workload; no remote execution."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    base=Path(__file__).resolve().parents[1]
    package=base/'experiments/orin_15w_priority_v1'
    target=base/'bundles/to_orin/orin_15w_complete_priority_v1.zip'
    assert not target.exists(), 'Preserve delivered bundles'
    co=json.loads((base/'experiments/manifests/orin_continuous_coslam.json').read_text())['jobs']
    ec=json.loads((base/'experiments/manifests/orin_continuous_ec.json').read_text())['jobs']
    inventory=json.loads((base/'experiments/ALL_CONFIGURED_NONREPLICA_INVENTORY.json').read_text())
    scene_info={(j['dataset'],j['scene']):j for j in inventory['scenes']}
    scenes={}
    for job in co:
        if job['dataset']=='Replica' and job['hash_size']!=16:continue
        scenes.setdefault((job['dataset'],job['scene']),job)
    assert len(scenes)==24
    # Dataset rotation and scene order are fixed independently of results.
    replica_order=['office0','room0','office1','office2','office3','office4','room1','room2']
    preferred={'Replica':replica_order,'Synthetic':['tg','ck'],'Tum':['fr1_desk','fr3_office'],
               'ScanNet':['scene0106','scene0000']}
    groups={}
    for dataset in ('Replica','Synthetic','Tum','ScanNet'):
        names=[s for d,s in scenes if d==dataset]
        first=[s for s in preferred[dataset] if s in names]
        groups[dataset]=first+sorted(set(names)-set(first))
    ordered=[]
    while any(groups.values()):
        for dataset,names in groups.items():
            if names:ordered.append(copy.deepcopy(scenes[(dataset,names.pop(0))]))
    phases=[]
    def phase(number,name):
        item=dict(priority=number,id=name,jobs=[])
        phases.append(item)
        return item['jobs']
    def benchmark(template,system,condition,seed,**extra):
        job=copy.deepcopy(template)
        dataset,scene,h=job['dataset'],job['scene'],job['hash_size']
        prefix='EC_' if system=='EC-SLAM' else ''
        job.update(id=f'{prefix}{dataset}_{scene}_{condition}_H{h}_seed{seed}',
                   system=system,condition=condition,seed=seed,execution_kind='benchmark',
                   runtime_family='co_scientific' if system=='Co-SLAM' else 'ec_native_timing',
                   frames=0,quality=False,export_mesh=False,warmup_frames=20,
                   expected_native_frames=2000 if dataset=='Replica' else scene_info[(dataset,scene)]['frames'])
        job.pop('requires',None)
        job.update(extra)
        return job
    dp=phase(1,'P1_DP_ALL_DATASETS')
    for scene in ordered:
        for seed in (0,1):
            for condition in (('BASE','DP') if seed==0 else ('DP','BASE')):
                job=benchmark(scene,'Co-SLAM',condition,seed)
                if condition=='DP':job['control_ids']={'BASE':job['id'].replace('_DP_','_BASE_')}
                dp.append(job)
    mc=phase(2,'P2_MC_ALL_REQUIRED')
    for scene in ordered:
        for seed in (0,1):
            job=benchmark(scene,'Co-SLAM','MC',seed)
            job['control_ids']={'BASE':job['id'].replace('_MC_','_BASE_')}
            mc.append(job)
    ec_lookup={(j['scene'],j['hash_size']):j for j in ec}
    co_lookup={(j['scene'],j['hash_size']):j for j in co if j['dataset']=='Replica'}
    # Main EC H16 comparison precedes the capacity sensitivity.
    for system,capacity,lookup in [('EC-SLAM',16,ec_lookup),('Co-SLAM',19,co_lookup),('EC-SLAM',19,ec_lookup)]:
        for scene in replica_order:
            template=lookup[(scene,capacity)]
            for seed in (0,1):
                for condition in (('BASE','MC') if seed==0 else ('MC','BASE')):
                    mc.append(benchmark(template,system,condition,seed))
    remaining=phase(3,'P3_COMPOSITION_AND_CO_BA_ENGINEERING')
    for scene in ordered:
        for seed in (0,1):
            job=benchmark(scene,'Co-SLAM','MC_DP',seed)
            job['control_ids']={c:job['id'].replace('_MC_DP_',f'_{c}_') for c in ('BASE','DP','MC')}
            remaining.append(job)
    for seed in (0,1):
        for condition in (('BASE_CONTROL','ENG') if seed==0 else ('ENG','BASE_CONTROL')):
            job=benchmark(co_lookup[('office0',16)],'Co-SLAM',condition,seed,
                          runtime_family='co_engineering_validated',reported_comparison_stage='bundle_adjustment',
                          native_scientific_base_is_not_engineering_control=True)
            job['id']=f'Co-SLAM_office0_ENG_H16_seed{seed}' if condition=='ENG' else f'Co_RTXREF_office0_BASE_CONTROL_seed{seed}'
            remaining.append(job)
    profiles=phase(4,'P4_FINAL_COMPONENT_PROFILES')
    for system,filename in [('Co-SLAM','orin_component_profile_15w_single_seed_v1.json'),
                            ('EC-SLAM','orin_ec_component_profile_15w_single_seed_v1.json')]:
        source=json.loads((base/'experiments/manifests'/filename).read_text())
        for original in source['jobs']:
            job=copy.deepcopy(original)
            dataset='Tum' if job['workload'].startswith('Tum_') else 'Replica'
            job.update(system=system,dataset=dataset,scene='fr1_desk' if dataset=='Tum' else 'office0',
                       hash_size=16 if job['workload'].endswith('H16') else 19,
                       execution_kind='component_profile',runtime_family='co_component_profile_repaired' if system=='Co-SLAM' else 'ec_component_profile_validated',
                       frames=0,quality=False,export_mesh=False,warmup_frames=20,
                       expected_native_frames=592 if dataset=='Tum' else 2000)
            profiles.append(job)
    # Latest author steering: cover every workload with seed 0 before any
    # second-seed benchmark. Profile once at the end of that first sweep.
    original_phases=phases
    phases=[]
    for seed in (0,1):
        for p in original_phases[:3]:
            phases.append(dict(priority=len(phases)+1,id=f'S{seed}_'+p['id'],
                               jobs=[j for j in p['jobs'] if j['seed']==seed]))
        if seed==0:
            phases.append(dict(priority=len(phases)+1,id='S0_P4_COMPONENT_PROFILES',jobs=profiles))
    jobs=[]
    for p in phases:
        for job in p['jobs']:
            job['priority']=p['priority'];job['phase']=p['id'];job['ordinal']=len(jobs)+1
            job['power_mode_required']='AUTHOR_SELECTED_15W_EXACT_LOCAL_NAME'
            jobs.append(job)
    assert [len(p['jobs']) for p in phases]==[48,72,26,5,48,72,26]
    assert len({j['id'] for j in jobs})==297
    assert Counter(j['dataset'] for j in dp if j['condition']=='DP')=={'Replica':16,'Synthetic':14,'Tum':6,'ScanNet':12}
    common=dict(author_date='2026-10-22',platform='ORIN_15W',power_mode=None,
                power_mode_binding='Resolve exact local nvpmodel label for author-manually-selected 15 W; never change mode',
                runtime_binding='Bind to matching completed MAXN local execution identities; profiler metadata repair is preserved',
                benchmark_seeds=[0,1],profiling_seeds=[0],warmup_frames=20,
                seed_policy='Complete seed 0 for all benchmarks and all five single-run profiles, then seed 1 benchmarks; no second profiling seed',
                launch_status='PLAN_PREPARED_ON_RTX_NOT_INSTALLED_OR_LAUNCHED_ON_ORIN',
                execution_interface='Ordered heterogeneous target ledger; bind to existing validated local serial supervisor; do not feed all jobs to the Co-only queue',
                gate='All required MAXN profiles/exports normally complete, actual manual switch to verified 15 W, exclusive GPU idle ownership',
                continuation='All ready jobs automatically, no human gate at phase/night boundaries, no arbitrary elapsed-time cap',
                current_maxn_interrupted=False,quality=False,export_mesh=False,
                frame_count_policy='Complete native sequences; no diagnostics counted as full jobs',
                primary_native_source_manifest_sha256='9786e843d6856dcc443fb08ebb9646a5d759384e19ab3456552b51e2281e616d')
    master=dict(**common,benchmark_slots=292,component_profile_slots=5,total_full_slots=297,
                phases=[dict(priority=p['priority'],id=p['id'],count=len(p['jobs'])) for p in phases],jobs=jobs)
    dump(package/'manifests/ALL_15W_TARGETS.json',master)
    for p in phases:
        dump(package/'manifests'/(p['id']+'.json'),dict(**common,phase=p['id'],priority=p['priority'],jobs=p['jobs']))
    dump(package/'COVERAGE.json',dict(expected_scene_counts={'Replica':8,'Synthetic':7,'Tum':3,'ScanNet':6},
        native_scene_order=[dict(dataset=j['dataset'],scene=j['scene'],hash_size=j['hash_size']) for j in ordered],
        benchmark_conditions=dict(Counter(j['condition'] for j in jobs if j['execution_kind']=='benchmark')),
        profile_preflights='3 Co +2 EC bounded fixed-state/50-frame checks before their respective final full profile series; separate from 297 full slots'))
    entries={str(p.relative_to(package)):p.read_bytes() for p in (package/'manifests').glob('*.json')}
    for name in ('README.md','validate_plan.py','COVERAGE.json'):
        entries[name]=(package/name).read_bytes()
    entries['NOTIFICATION_POLICY.md']=(base/'docs/protocols/NOTIFICATION_POLICY.md').read_bytes()
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in sorted(entries.items())}
    entries['SHA256SUMS.json']=(json.dumps(hashes,indent=2)+'\n').encode()
    for name,data in entries.items():
        path=package/name;path.parent.mkdir(exist_ok=True,parents=True);path.write_bytes(data)
    subprocess.run([sys.executable,str(package/'validate_plan.py')],check=True)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as saved:
        for name,data in sorted(entries.items()):saved.writestr(name,data)
    with zipfile.ZipFile(target) as saved:
        assert saved.testzip() is None
        assert all(hashlib.sha256(saved.read(n)).hexdigest()==v for n,v in hashes.items())
    receipt=dict(archive=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 phase_full_slots=[48,72,26,5,48,72,26],seed0_full_slots=151,seed1_full_slots=146,
                 benchmark_slots=292,profile_slots=5,remote_installed=False,remote_launched=False)
    dump(base/'results/ORIN_15W_PRIORITY_HANDOFF.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
