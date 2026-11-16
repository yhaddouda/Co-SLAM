"""Audit worker-local EC partitions and aggregate calls/thirds/seeds separately."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from report_component_profiles import COMPONENTS, read_csv, write_csv


EC_COMPONENTS = dict(COMPONENTS, call=[*COMPONENTS['call'], 'handshake_wait'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_run(directory, job, work, source_sha):
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['system']=='EC-SLAM' and summary['condition']=='BASE'
    assert summary['full_sequence'] and not summary['diagnostic'] and summary['finite_parameters_and_poses']
    assert summary['workload']==job['workload'] and summary['seed']==job['seed']
    assert summary['frames']==work['expected_native_frames']
    assert summary['source_manifest_sha256']==source_sha==sha(directory/'source_manifest.json')
    assert json.loads((directory/'selection.json').read_text())==work
    cfg=json.loads((directory/'config.json').read_text())
    assert cfg['HashGrid']['hash_size']==work['hash_size']
    assert cfg['tracking']['iters']==8 and cfg['mapping']['iters']==10 and cfg['mapping']['every_frame']==5
    assert summary['selected_calls']=={'tracking':300,'bundle_adjustment':60}
    weighted,thirds=[],[]
    for stage in ('tracking','bundle_adjustment'):
        root=directory/stage
        parents={int(r['id']):r for r in read_csv(root/'parents.csv')}
        parts=defaultdict(dict)
        calls=read_csv(root/'calls.csv')
        for row in read_csv(root/'components.csv'):
            pid=int(row['id']);parent=parents[pid]
            assert row['stage']==stage==parent['stage'] and row['scope']==parent['scope']
            assert row['component'] in EC_COMPONENTS[row['scope']]
            assert row['component'] not in parts[pid]
            parts[pid][row['component']]=float(row['cuda_ms'])
        grouped=defaultdict(list)
        for pid,parent in parents.items():
            total=float(parent['cuda_ms'])
            assert abs(sum(parts[pid].values())-total)<=.02+.005*total
            grouped[(int(parent['frame_id']),parent['scope'])].append((pid,parent))
        selected={f for t in work['thirds'] for f in t[stage]['selected_frames']}
        assert {f for f,scope in grouped if scope=='call'}==selected
        assert {int(r['frame_id']) for r in calls}==selected and len(calls)==len(selected)
        call_lookup={int(r['frame_id']):r for r in calls}
        for frame in selected:
            iterations=grouped[(frame,'iteration')]
            count=work['max_tracking_updates'] if stage=='tracking' else work['native_ba_updates']
            assert sorted(int(p['iteration_id']) for _,p in iterations)==list(range(count))
            forwards=grouped[(frame,'forward')]
            assert len(forwards)==count
            assert sorted(int(p['iteration_id']) for _,p in forwards)==list(range(count))
            assert len(grouped[(frame,'call')])==1
            # Waiting is represented explicitly only in the call envelope.
            for scope in ('iteration','forward'):
                assert all('handshake_wait' not in parts[pid] for pid,_ in grouped[(frame,scope)])
        for scope,names in EC_COMPONENTS.items():
            windows=[]
            for third in work['thirds']:
                rows=[]
                for frame in third[stage]['selected_frames']:
                    values=grouped[(frame,scope)]
                    row=dict(total=statistics.mean(float(p['cuda_ms']) for _,p in values))
                    row.update({name:statistics.mean(parts[pid].get(name,0.) for pid,_ in values) for name in names})
                    rows.append(row)
                means={name:statistics.mean(r[name] for r in rows) for name in ['total',*names]}
                windows.append((third[stage]['scene_weight'],means))
                for component,value in means.items():
                    thirds.append(dict(workload=job['workload'],seed=job['seed'],stage=stage,scope=scope,
                                       third=third['name'],component=component,cuda_ms=value,
                                       selected_calls=len(rows),eligible_calls=third[stage]['eligible_calls'],weight=third[stage]['scene_weight']))
            for component in ['total',*names]:
                weighted.append(dict(workload=job['workload'],seed=job['seed'],stage=stage,scope=scope,
                                     component=component,cuda_ms=sum(w*m[component] for w,m in windows)))
        for component in ('wall_ms','handshake_wait_wall_ms','active_body_wall_ms'):
            value=sum(t[stage]['scene_weight']*statistics.mean(float(call_lookup[f][component]) for f in t[stage]['selected_frames']) for t in work['thirds'])
            weighted.append(dict(workload=job['workload'],seed=job['seed'],stage=stage,scope='host_call',component=component,cuda_ms=value))
    return weighted,thirds


def report(base, active=None):
    active=Path(active) if active else base/'results/EC_COMPONENT_PROFILE_ACTIVE.json'
    if not active.exists():return dict(completed=0,total=6,groups=[],runtime_ready=False),[]
    spec=json.loads(active.read_text())
    runtime,queue,manifest_path=(Path(spec[k]) for k in ('runtime','queue','manifest'))
    manifest=json.loads(manifest_path.read_text())
    protocol=json.loads((runtime/'PROTOCOL.json').read_text())
    lookup={w['id']:w for w in protocol['workloads']}
    source_sha=sha(runtime/'RUNTIME_MANIFEST.json')
    weighted,thirds,inputs=[],[],[active,manifest_path,runtime/'PROTOCOL.json',runtime/'RUNTIME_MANIFEST.json']
    done=0
    for job in manifest['jobs']:
        marker=queue/(job['id']+'.done.json')
        if not marker.exists():continue
        directory=queue/job['id']
        assert sha(directory/'summary.json')==json.loads(marker.read_text())['summary_sha256']
        rows,windows=audit_run(directory,job,lookup[job['workload']],source_sha)
        weighted.extend(rows);thirds.extend(windows);done+=1
        inputs.extend(directory/name for name in ('summary.json','config.json','source_manifest.json','selection.json'))
        for stage in ('tracking','bundle_adjustment'):
            inputs.extend(directory/stage/name for name in ('parents.csv','components.csv','segments.csv','calls.csv','query_calls.csv'))
    groups=[]
    scopes=dict(EC_COMPONENTS,host_call=['wall_ms','handshake_wait_wall_ms','active_body_wall_ms'])
    for work in protocol['workloads']:
        seeds={j['seed'] for j in manifest['jobs'] if j['workload']==work['id']}
        rows=[r for r in weighted if r['workload']==work['id']]
        if {r['seed'] for r in rows}!=seeds:continue
        for stage in ('tracking','bundle_adjustment'):
            for scope,names in scopes.items():
                group=dict(system='EC-SLAM',platform=manifest['platform'],workload=work['id'],stage=stage,scope=scope,seeds=sorted(seeds),components={})
                for component in (names if scope=='host_call' else ['total',*names]):
                    values=[r['cuda_ms'] for r in rows if r['stage']==stage and r['scope']==scope and r['component']==component]
                    assert len(values)==len(seeds)
                    group['components'][component]=dict(mean_ms=statistics.mean(values),sd_ms=statistics.stdev(values) if len(values)>1 else None)
                for values in group['components'].values():
                    values['share_pct']=None if scope=='host_call' else 100*values['mean_ms']/group['components']['total']['mean_ms']
                groups.append(group)
    result=dict(completed=done,total=len(manifest['jobs']),groups=groups,runtime_ready=True,
                status=json.loads((queue/'status.json').read_text()) if (queue/'status.json').exists() else {},
                aggregation='Worker-specific call means, eligible-call-weighted chronological thirds, seed mean/sample SD. Never add overlapping workers. Host and CUDA quantities remain separate.',
                input_sha256={str(p):sha(p) for p in inputs})
    output=base/'results/ec_component_profile_analysis_v1';output.mkdir(exist_ok=True)
    write_csv(output/'per_seed.csv',weighted);write_csv(output/'per_third.csv',thirds)
    (output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    text='# EC-SLAM component profiling\n\n'+f'{done}/{len(manifest["jobs"])} complete full passes; complete seed groups only.\n\n'
    text+='Native concurrent workers/streams retained. CUDA-event stream intervals include contention/host idle and can overlap across workers. No summed worker FPS or isolated-kernel claim. Raw call envelope includes the explicitly named native handshake interval; host active-body wall subtracts actual host waiting only. Loader yield, publish/logger and outer handshakes are outside the selected worker body. EC has no smoothness computation in its pinned forward despite accepting a smooth argument.\n\n'
    text+='| Workload | Worker | Scope | Component | ms ± seed SD | Share |\n|---|---|---|---|---:|---:|\n'
    for g in groups:
        for name,v in g['components'].items():
            share='—' if v['share_pct'] is None else f"{v['share_pct']:.2f}%"
            dispersion='' if v['sd_ms'] is None else f" ± {v['sd_ms']:.4f}"
            text+=f"| {g['workload']} | {g['stage']} | {g['scope']} | {name} | {v['mean_ms']:.4f}{dispersion} | {share} |\n"
    (output/'REPORT.md').write_text(text)
    return result,inputs+[output/'SUMMARY.json']


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--active',type=Path)
    args=parser.parse_args()
    result,_=report(args.base,args.active)
    print('EC_COMPONENT_PROFILE_AUDIT',result['completed'],'/',result['total'])
