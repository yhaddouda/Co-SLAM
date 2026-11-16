#!/usr/bin/env python3
"""Audit the MAXN increment on CPU, retaining the earlier 172-run evidence."""
import ast
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from round2_paths import local_path

BASE=Path(__file__).resolve().parents[1]
RECEIVED=BASE/'results/orin_maxn_received_20261022'
OUT=BASE/'results/orin_maxn_complete_analysis_v1'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):
    with Path(path).open() as stream:yield from csv.DictReader(stream)
def save(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')
def csv_save(path,data):
    columns=list(dict.fromkeys(k for row in data for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,columns);writer.writeheader();writer.writerows(data)


def normalize(cfg,system):
    cfg=copy.deepcopy(cfg)
    for key in ('datadir','output','exp_name','input_folder'):cfg.get('data',{}).pop(key,None)
    for key in ('condition','seed','engineering','sampler'):cfg.get('round2',{}).pop(key,None)
    for key in ('hash','morton_sort'):cfg['grid' if system=='Co-SLAM' else 'HashGrid'].pop(key,None)
    return cfg


def verify_profile_sources():
    from build_component_profile_runtime import verify_ast as co_ast
    from build_ec_component_profile_runtime import verify_ast as ec_ast
    proof=[]
    for name,parent,driver,system in [
        ('co_bundle','component_profile_runtime_v2','run_component_profile.py','Co-SLAM'),
        ('bundle','ec_component_profile_runtime_v2','run_ec_component_profile.py','EC-SLAM')]:
        runtime=RECEIVED/'component_profiles'/name/'runtime'
        manifest=read(runtime/'RUNTIME_MANIFEST.json')
        native=BASE/'experiments'/parent
        old=read(native/'RUNTIME_MANIFEST.json')
        assert manifest['metadata_reader_recovery']['parent_runtime_manifest_sha256']==sha(native/'RUNTIME_MANIFEST.json')
        for filename,expected in manifest['runtime_hashes'].items():assert sha(runtime/filename)==expected,filename
        assert set(manifest['runtime_hashes'])==set(old['runtime_hashes'])|{'optional_sysfs_metadata.py'}
        for filename,expected in old['runtime_hashes'].items():
            if filename!=driver:assert manifest['runtime_hashes'][filename]==expected,(system,filename)
        class IgnoreOptionalEnvironment(ast.NodeTransformer):
            def visit_FunctionDef(self,node):
                if node.name=='environment':node.body=[ast.Pass()];return node
                return self.generic_visit(node)
        left=IgnoreOptionalEnvironment().visit(ast.parse((native/driver).read_text()))
        right=IgnoreOptionalEnvironment().visit(ast.parse((runtime/driver).read_text()))
        assert ast.dump(left,include_attributes=False)==ast.dump(right,include_attributes=False)
        if system=='Co-SLAM':
            for filename in ('coslam.py','model/scene_rep.py'):
                co_ast((runtime/'reference'/filename).read_text(),(runtime/filename).read_text())
        else:
            for filename in ('src/Tracker.py','src/Mapper.py','src/Model.py'):
                ec_ast((runtime/'reference'/filename).read_text(),(runtime/filename).read_text())
        proof.append(dict(system=system,runtime_manifest_sha256=sha(runtime/'RUNTIME_MANIFEST.json'),
                          parent_manifest_sha256=sha(native/'RUNTIME_MANIFEST.json'),
                          nonmetadata_driver_AST_identical=True,all_model_config_timer_validator_selection_bytes_identical=True))
    return proof


def audit_increment(job,scientific):
    directory=RECEIVED/'benchmarks'/job['id']
    summary,cfg,source=(read(directory/n) for n in ('summary.json','config.json','source_manifest.json'))
    assert summary['full_sequence'] and not summary['diagnostic'] and summary['finite_parameters_and_poses']
    assert summary['quality'] is False and summary['warmup_frames']==20
    for key in ('dataset','scene','condition','seed'):assert summary[key]==job[key],(job['id'],key)
    assert summary['initial_power'].splitlines()[0]=='NV Power Mode: MAXN'
    assert summary['final_power'].splitlines()[0]=='NV Power Mode: MAXN'
    assert source==scientific and cfg['grid']['hash_size']==job['hash_size']
    assert cfg['round2']['seed']==job['seed'] and not cfg['round2']['engineering']
    morton=job['condition'] in ('MC','MC_DP');prune=job['condition'] in ('DP','MC_DP')
    assert cfg['grid']['hash']==('Morton' if morton else 'CoherentPrime')
    assert bool(cfg['grid']['morton_sort'])==morton
    assert cfg['round2']['sampler']==('support' if prune else 'dense')
    assert cfg['round2']['margin_scene_units']==.15 and cfg['round2']['anchor_every']==3
    frames=summary['frames'];assert frames==job['native_frames']
    calls=defaultdict(list);iterations=defaultdict(list)
    for row in rows(directory/'calls.csv'):
        if int(row['frame_id'])>=20:calls[row['stage']].append(row)
    for row in rows(directory/'iterations.csv'):
        if int(row['frame_id'])>=20:iterations[(row['iteration_type'],int(row['frame_id']))].append(row)
    record=dict(id=job['id'],category=job['condition'] if prune else 'MC',system='Co-SLAM',
                dataset=job['dataset'],scene=job['scene'],hash_size=job['hash_size'],seed=job['seed'],
                condition=job['condition'],status=job['status'],frames=frames,
                directory=str(directory.relative_to(BASE)),source_sha256=sha(directory/'source_manifest.json'))
    for stage,label in [('tracking','tracking'),('bundle_adjustment','ba')]:
        selected=calls[stage]
        expected=list(range(20,frames)) if stage=='tracking' else list(range(20,frames,cfg['mapping']['map_every']))
        assert sorted(int(c['frame_id']) for c in selected)==expected,(job['id'],stage)
        updates=cfg['tracking']['iter'] if stage=='tracking' else cfg['mapping']['iters']
        totals=[];forwards=[];backwards=[]
        for call in selected:
            values=iterations[(stage,int(call['frame_id']))]
            assert len(values)==updates and {int(v['iteration_id']) for v in values}==set(range(updates))
            totals.append(sum(float(v['full_iteration_cuda_ms']) for v in values))
            forwards.extend(float(v['forward_cuda_ms']) for v in values)
            backwards.extend(float(v['backward_cuda_ms']) for v in values)
        record[label+'_calls']=len(selected);record[label+'_updates_per_call']=updates
        assert summary[stage]['count']==len(selected)
        for field in ('wall_ms','cuda_ms'):
            value=statistics.mean(float(c[field]) for c in selected)
            assert abs(value-summary[stage]['mean_'+field])<1e-6
            record[label+'_'+field]=value
        record[label+'_iteration_cuda_ms']=statistics.mean(totals)
        record[label+'_forward_cuda_ms']=statistics.mean(forwards)
        record[label+'_backward_cuda_ms']=statistics.mean(backwards)
    query_totals=defaultdict(lambda:[0,0,0,0,0])
    for row in rows(directory/'queries.csv'):
        stage=row['stage'];dense,actual=int(row['dense_queries']),int(row['actual_queries'])
        assert 0<actual<=dense
        frame,i=int(row['frame_id']),int(row['iteration'])
        total=cfg['tracking']['iter'] if stage=='tracking' else cfg['mapping']['iters']
        assert 0<=i<total
        if not prune or i==0 or i==total-1 or i%3==0:assert actual==dense
        counts=query_totals[stage];counts[0]+=dense;counts[1]+=actual
        if frame>=20:
            counts[2]+=dense;counts[3]+=actual;counts[4]+=int(actual<dense)
    for stage,label in [('tracking','tracking'),('bundle_adjustment','ba')]:
        dense,actual,post_dense,post_actual,pruned_updates=query_totals[stage]
        assert dense==summary[stage]['dense_queries'] and actual==summary[stage]['actual_queries']
        assert abs(100*(1-actual/dense)-summary[stage]['queries_pruned_pct'])<1e-6
        assert (pruned_updates>0)==prune,(job['id'],stage)
        record[label+'_queries_pruned_pct']=100*(1-post_actual/post_dense)
    assert all(math.isfinite(v) and v>=0 for v in record.values() if isinstance(v,(int,float)))
    return record,cfg


def aggregate(records):
    groups=[]
    for system in ('Co-SLAM','EC-SLAM'):
        for dataset in ('Replica','Synthetic','Tum','ScanNet'):
            if system=='EC-SLAM' and dataset!='Replica':continue
            capacities=sorted({r['hash_size'] for r in records if r['system']==system and r['dataset']==dataset and r['category']!='ENG'})
            if dataset=='ScanNet':capacities+=[None]
            for capacity in capacities:
                for variant in (('MC','DP','MC_DP') if system=='Co-SLAM' else ('MC',)):
                    selected=[r for r in records if r['category']!='ENG' and r['system']==system and r['dataset']==dataset
                              and (capacity is None or r['hash_size']==capacity) and r['condition'] in ('BASE',variant)]
                    if not any(r['condition']==variant for r in selected):continue
                    groups.append(group(selected,system,dataset,capacity,variant))
    eng=[r for r in records if r['category']=='ENG']
    groups.append(group(eng,'Co-SLAM','Replica',16,'ENG'))
    return groups


def group(values,system,dataset,capacity,variant):
    scenes=sorted({r['scene'] for r in values})
    assert len(values)==len(scenes)*4
    assert {(r['scene'],r['seed'],r['condition']) for r in values}=={(s,seed,c) for s in scenes for seed in (0,1) for c in ('BASE',variant)}
    metrics=sorted(set.intersection(*[{k for k,v in r.items() if v is not None and k.endswith(('_ms','_calls','_updates_per_call'))} for r in values]))
    row=dict(category=variant,system=system,dataset=dataset,hash_size=capacity,scene_count=len(scenes),scene_names=scenes,seeds=[0,1],
             capacity_scope='native mixed capacities, equal scene weights' if capacity is None else 'single hash capacity')
    for field in metrics:
        for condition in ('BASE',variant):
            means=[statistics.mean(r[field] for r in values if r['seed']==seed and r['condition']==condition) for seed in (0,1)]
            row[condition+'_'+field]=statistics.mean(means);row[condition+'_'+field+'_sd']=statistics.stdev(means)
        row[field+'_reduction_pct']=100*(1-row[variant+'_'+field]/row['BASE_'+field])
    return row


def main():
    OUT.mkdir(exist_ok=True)
    receipt=read(RECEIVED/'RECEIPT.json')
    assert sha(receipt['archive'])==receipt['sha256']
    for name,expected in read(RECEIVED/'SHA256SUMS.json').items():assert sha(RECEIVED/name)==expected,name
    master=read(RECEIVED/'MAXN_MASTER_INDEX.json')
    expected=read(BASE/'experiments/orin_15w_priority_v1/manifests/ALL_15W_TARGETS.json')['jobs']
    assert len(master['benchmarks'])==292
    assert {j['id'] for j in master['benchmarks']}=={j['id'] for j in expected if j['execution_kind']=='benchmark'}
    assert len({j['id'] for j in master['benchmarks']})==292
    # Recheck the earlier audit at its original recorded hashes; do not edit it.
    old_status=read(BASE/'results/orin_maxn_analysis/ANALYSIS_STATUS.json')
    for path,digest in old_status['input_sha256'].items():assert sha(local_path(BASE,path))==digest,path
    records=[]
    for row in rows(BASE/'results/orin_maxn_analysis/per_run.csv'):
        for key,value in row.items():
            if key in ('hash_size','seed','frames') or key.endswith(('_calls','_updates_per_call')):row[key]=int(value) if value else None
            elif key.endswith('_ms'):row[key]=float(value) if value else None
        records.append(row)
    old={r['id']:r for r in records}
    inputs={str(RECEIVED/'RECEIPT.json'):sha(RECEIVED/'RECEIPT.json'),str(Path(receipt['archive'])):receipt['sha256'],
            str(Path(__file__).resolve()):sha(Path(__file__).resolve()),
            str(RECEIVED/'MAXN_MASTER_INDEX.json'):sha(RECEIVED/'MAXN_MASTER_INDEX.json'),
            str(RECEIVED/'SHA256SUMS.json'):sha(RECEIVED/'SHA256SUMS.json')}
    inputs.update(old_status['input_sha256'])
    scientific=read(RECEIVED/'scientific_runtime/RUNTIME_MANIFEST.json')
    assert sha(RECEIVED/'scientific_runtime/RUNTIME_MANIFEST.json')==sha(BASE/'experiments/coslam_reference_runtime_v1/RUNTIME_MANIFEST.json')
    for name,expected in scientific['runtime_hashes'].items():assert sha(RECEIVED/'scientific_runtime'/name)==expected,name
    configs={}
    for row in records:
        if row['category']=='ENG':continue
        cfg=read(BASE/row['directory']/'config.json')
        key=(row['system'],row['dataset'],row['scene'],row['hash_size'])
        normalized=normalize(cfg,row['system'])
        if key in configs:assert normalized==configs[key]
        configs[key]=normalized
    for job in master['benchmarks']:
        if job['delivery']=='previous_172_archive':
            row=old[job['id']];directory=BASE/row['directory']
            for filename,key in [('summary.json','summary_sha256'),('config.json','config_sha256'),('source_manifest.json','source_manifest_sha256')]:
                assert sha(directory/filename)==job[key]
            continue
        record,cfg=audit_increment(job,scientific)
        key=('Co-SLAM',job['dataset'],job['scene'],job['hash_size'])
        normalized=normalize(cfg,'Co-SLAM')
        if key in configs:assert configs[key]==normalized,('Paired config mismatch',job['id'])
        configs[key]=normalized
        records.append(record)
        for filename in ('summary.json','config.json','source_manifest.json','calls.csv','iterations.csv','queries.csv'):
            path=RECEIVED/'benchmarks'/job['id']/filename;inputs[str(path)]=sha(path)
    assert len(records)==292 and len({r['id'] for r in records})==292
    groups=aggregate(records)
    profile_source_audit=verify_profile_sources()
    status=dict(verified_runs=292,native_runs=192,engineering_runs=4,reused_runs=sum(r['status']=='reused' for r in records),
                seeds=[0,1],mode='MAXN',pending_scannet_runs=0,dp_runs_in_this_snapshot=48,composition_runs_in_this_snapshot=48,
                aggregation='Within-run post-warmup means; equal scene weights per seed, two-seed mean/sample SD; ratios of matched means',
                ec_scope=old_status['ec_scope'],clocks=old_status['clocks'],excluded='EC engineering',groups=groups,
                profiles_received=5,profile_seeds=[0],profile_source_audit=profile_source_audit,
                archive_sha256=receipt['sha256'],previous_verified_runs=172,incremental_verified_runs=120,
                input_sha256=inputs,received_code_executed=False)
    csv_save(OUT/'per_run.csv',records);csv_save(OUT/'dataset_means.csv',groups)
    save(OUT/'ANALYSIS_STATUS.json',status)
    text='# Complete MAXN benchmark import\n\n292/292 benchmark identities audited: earlier 172 plus 120 new. No experiments repeated. Five single-seed profiles received; their raw partition audit is recorded separately. All increment CRC/payload hashes, native sources, paired configs, full call/iteration coverage and pruning anchor/query counters passed. Recorded local numerical profiling checks are preserved; no received code is executed.\n\n'
    text+='Equal scene weights; sample SD across two seed-wise means. Frames 0--19 excluded. Whole invocation host wall and iteration-sum CUDA intervals remain separate. ScanNet whole-dataset rows pool its native H12/H14/H16 capacities with equal scene weights; capacity-specific rows remain in CSV. EC worker scopes differ from RTX as documented. Clock-lock status is not inferred.\n\n'
    text+='| Dataset | Comparison | Scenes | TR host reduction | BA host reduction | BA iteration-sum reduction |\n|---|---|---:|---:|---:|---:|\n'
    for r in groups:
        if r['system']!='Co-SLAM' or (r['dataset']=='Replica' and r['hash_size']!=16) or (r['dataset']=='ScanNet' and r['hash_size'] is not None):continue
        tracking='---' if r['category']=='ENG' else f"{r['tracking_wall_ms_reduction_pct']:+.2f}%"
        text+=f"| {r['dataset']} | {r['category']} | {r['scene_count']} | {tracking} | {r['ba_wall_ms_reduction_pct']:+.2f}% | {r['ba_iteration_cuda_ms_reduction_pct']:+.2f}% |\n"
    (OUT/'REPORT.md').write_text(text)
    print('MAXN_INCREMENT_AUDITED',len(records),'benchmark identities;',len(groups),'comparison groups; profile source/metadata-only AST checks passed')


if __name__=='__main__':main()
