"""Audit selected-call partitions and form temporally weighted seed summaries."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics


COMPONENTS = {
    'forward': ['hash_grid', 'decoder', 'sampling_rendering', 'other_forward'],
    'iteration': ['ray_pose_preparation', 'model_forward', 'external_loss', 'dense_smoothness',
                  'pose_selection', 'backward', 'optimizer_update', 'iteration_other'],
    'call': ['call_setup', 'iterations', 'loop_gaps', 'call_finalization'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def audit_run(directory, job, work, source_sha):
    result = json.loads((directory/'summary.json').read_text())
    assert result['full_sequence'] and not result['diagnostic'] and result['finite_parameters_and_poses']
    assert result['workload']==job['workload'] and result['seed']==job['seed']
    assert result['source_manifest_sha256']==source_sha
    assert result['selected_calls']=={'tracking':300, 'bundle_adjustment':60}
    source = json.loads((directory/'source_manifest.json').read_text())
    assert sha(directory/'source_manifest.json')==source_sha
    assert source['computational_ast_preserved'] and source['baseline_only']
    cfg = json.loads((directory/'config.json').read_text())
    assert cfg['grid']['hash_size']==work['hash_size'] and cfg['grid']['hash']=='CoherentPrime'
    assert cfg['grid']['morton_sort'] is False and cfg['round2']['engineering'] is False
    assert cfg['round2']['sampler']=='dense'
    parents = {int(row['id']):row for row in read_csv(directory/'parents.csv')}
    components = defaultdict(dict)
    for row in read_csv(directory/'components.csv'):
        parent = parents[int(row['id'])]
        assert row['scope']==parent['scope']
        assert row['component'] in COMPONENTS[row['scope']]
        assert row['component'] not in components[int(row['id'])]
        components[int(row['id'])][row['component']]=float(row['cuda_ms'])
    for parent_id,parent in parents.items():
        total = sum(components[parent_id].values())
        elapsed = float(parent['cuda_ms'])
        assert abs(total-elapsed)<=.02+.005*elapsed
    per_call = defaultdict(list)
    for parent_id,parent in parents.items():
        per_call[(parent['stage'],int(parent['frame_id']),parent['scope'])].append((parent_id,parent))
    weighted, thirds, contexts = [], [], []
    for stage in ('tracking','bundle_adjustment'):
        observed = {frame for kind,frame,scope in per_call if kind==stage and scope=='call'}
        selected = {frame for third in work['thirds'] for frame in third[stage]['selected_frames']}
        assert observed==selected
        for frame in selected:
            historical=(frame+cfg['mapping']['keyframe_every']-1)//cfg['mapping']['keyframe_every']
            contexts.append(dict(workload=job['workload'],seed=job['seed'],stage=stage,frame_id=frame,
                                 historical_keyframes=historical,
                                 stage_pose_population=historical+1 if stage=='bundle_adjustment' else 1,
                                 optimized_poses=(historical-1+int(cfg['mapping']['optim_cur'])) if stage=='bundle_adjustment' else 1,
                                 count_source='Derived from verified native full-sequence insertion cadence and pose rules; not a sampled runtime counter'))
            iterations = per_call[(stage,frame,'iteration')]
            ids = [int(parent['iteration_id']) for _,parent in iterations]
            assert sorted(ids)==list(range(len(ids)))
            assert len(ids)<=work['max_tracking_updates'] if stage=='tracking' else len(ids)==work['native_ba_updates']
        for scope,names in COMPONENTS.items():
            summaries = []
            for third in work['thirds']:
                call_means = []
                for frame in third[stage]['selected_frames']:
                    values = per_call[(stage,frame,scope)]
                    assert values
                    row = {'total':statistics.mean(float(parent['cuda_ms']) for _,parent in values)}
                    row.update({name:statistics.mean(components[pid].get(name,0.) for pid,_ in values) for name in names})
                    call_means.append(row)
                means = {key:statistics.mean(row[key] for row in call_means) for key in ['total',*names]}
                summaries.append((third[stage]['scene_weight'],means))
                for component,value in means.items():
                    thirds.append(dict(workload=job['workload'],seed=job['seed'],stage=stage,scope=scope,
                                       third=third['name'],component=component,cuda_ms=value,
                                       selected_calls=len(call_means),eligible_calls=third[stage]['eligible_calls'],
                                       weight=third[stage]['scene_weight']))
            for component in ['total',*names]:
                mean = sum(weight*means[component] for weight,means in summaries)
                weighted.append(dict(workload=job['workload'],seed=job['seed'],stage=stage,scope=scope,
                                     component=component,cuda_ms=mean))
    return weighted,thirds,contexts


def report(base, active=None):
    active = Path(active) if active else base/'results/COMPONENT_PROFILE_ACTIVE.json'
    if not active.exists():
        return dict(completed=0,total=9,groups=[],runtime_ready=False),[]
    spec = json.loads(active.read_text())
    queue,runtime = Path(spec['queue']),Path(spec['runtime'])
    manifest = json.loads(Path(spec['manifest']).read_text())
    protocol = json.loads((runtime/'PROTOCOL.json').read_text())
    lookup = {work['id']:work for work in protocol['workloads']}
    source_sha = sha(runtime/'RUNTIME_MANIFEST.json')
    rows,thirds,contexts,inputs = [],[],[],[active,Path(spec['manifest']),runtime/'RUNTIME_MANIFEST.json',runtime/'PROTOCOL.json']
    done=0
    for job in manifest['jobs']:
        marker=queue/(job['id']+'.done.json')
        if not marker.exists():
            continue
        directory=queue/job['id']
        assert sha(directory/'summary.json')==json.loads(marker.read_text())['summary_sha256']
        weighted,windows,context=audit_run(directory,job,lookup[job['workload']],source_sha)
        rows.extend(weighted);thirds.extend(windows);contexts.extend(context);done+=1
        inputs.extend(directory/name for name in ('summary.json','parents.csv','components.csv','calls.csv','config.json','source_manifest.json','selection.json','query_calls.csv'))
    groups=[]
    for work in protocol['workloads']:
        case_rows=[r for r in rows if r['workload']==work['id']]
        expected_seeds={j['seed'] for j in manifest['jobs'] if j['workload']==work['id']}
        if {r['seed'] for r in case_rows}!=expected_seeds:
            continue
        for stage in ('tracking','bundle_adjustment'):
            for scope,names in COMPONENTS.items():
                group=dict(platform=manifest['platform'],workload=work['id'],stage=stage,scope=scope,seeds=sorted(expected_seeds),components={})
                for component in ['total',*names]:
                    values=[r['cuda_ms'] for r in case_rows if r['stage']==stage and r['scope']==scope and r['component']==component]
                    assert len(values)==len(expected_seeds)
                    group['components'][component]=dict(mean_ms=statistics.mean(values),sd_ms=statistics.stdev(values))
                for component,values in group['components'].items():
                    values['share_pct']=100*values['mean_ms']/group['components']['total']['mean_ms']
                groups.append(group)
    result=dict(completed=done,total=len(manifest['jobs']),groups=groups,runtime_ready=True,
                status=json.loads((queue/'status.json').read_text()) if (queue/'status.json').exists() else {},
                aggregation='Within-call mean per executed forward/iteration; within-third call mean; eligible-call-weighted thirds; seed mean/sample SD; shares are ratios of matched means',
                input_sha256={str(p):sha(p) for p in inputs})
    output=base/'results/component_profile_analysis_v1';output.mkdir(exist_ok=True)
    write_csv(output/'per_seed.csv',rows);write_csv(output/'per_third.csv',thirds);write_csv(output/'call_context.csv',contexts)
    (output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    text='# Co-SLAM component profiling\n\n'+f'{done}/{len(manifest["jobs"])} full {manifest["platform"]} passes; only complete seed groups enter summaries.\n\n'
    text+='Forward and iteration numbers are CUDA-event stream intervals, not isolated active-kernel times. Call host wall measurements remain separate. No component rescaling or overhead subtraction. Early/middle/late raw summaries are retained.\n\n'
    for scope,title in [('forward','Forward'),('iteration','Iteration'),('call','Call envelope')]:
        text+=f'## {title}\n\n| Workload | Stage | Component | ms ± seed SD | Share |\n|---|---|---|---:|---:|\n'
        for group in groups:
            if group['scope']!=scope:continue
            for component,value in group['components'].items():
                text+=f"| {group['workload']} | {group['stage']} | {component} | {value['mean_ms']:.4f} ± {value['sd_ms']:.4f} | {value['share_pct']:.2f}% |\n"
        text+='\n'
    (output/'REPORT.md').write_text(text)
    return result,inputs+[output/'SUMMARY.json']


def table_tex(result):
    groups={(g['platform'],g['workload'],g['stage']):g for g in result.get('groups',[]) if g['scope']=='forward'}
    text=r'\begin{table}[htbp]\centering\scriptsize\caption{Forward decomposition. Components are CUDA-event ms per JointEncoding.forward evaluation (mean $\pm$ seed SD) and share of the matched total. Co-SLAM BASE uses office0 H16/H19 and fr1\_desk H16, full online sequences, and 100 tracking/20 BA calls spread across each post-warmup third. Third means are weighted by eligible native calls; three RTX/two Orin seeds. Includes model-internal supervision, excludes external smoothness, backward, optimizer and call setup/finalization. Other contains positional encoding, normalization/glue and internal loss assembly. CUDA intervals can include host-induced stream idle; profiling times are not primary uninstrumented speed measurements. Existing EC placeholders remain separate; unmeasured cells are TBD.}\label{tab:profiling_regimes}\resizebox{\linewidth}{!}{\begin{tabular}{llllrrrrr}\toprule System&Platform&Workload&Stage&Hash grid&Decoder&Sampling/rendering&Other&Total ms\\\midrule'+'\n'
    for platform,label in [('RTX_4070','RTX 4070'),('ORIN_MAXN','MAXN'),('ORIN_15W','15 W')]:
        for workload,name in [('Replica_office0_H16','office0 H16'),('Replica_office0_H19','office0 H19'),('Tum_fr1_desk_H16',r'fr1\_desk H16')]:
            for stage,short in [('tracking','TR'),('bundle_adjustment','BA')]:
                group=groups.get((platform,workload,stage));values=[]
                for component in ['hash_grid','decoder','sampling_rendering','other_forward','total']:
                    if group:
                        value=group['components'][component]
                        cell=f"${value['mean_ms']:.3f}\\pm{value['sd_ms']:.3f}$"
                        if component!='total':cell=r'\shortstack{'+cell+r'\\'+f"${value['share_pct']:.1f}\\%$"+'}'
                    else:cell=r'\pending'
                    values.append(cell)
                text+='&'.join(['Co',label,name,short,*values])+r'\\'+'\n'
    text+=r'\midrule'+'\n'
    for platform in ('RTX 4070','MAXN','15 W'):
        for stage in ('TR','Map'):
            text+='&'.join(['EC',platform,'Replica H16',stage,*([r'\pending']*5)])+r'\\'+'\n'
    return text+r'\bottomrule\end{tabular}}\end{table}'+'\n'


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--active',type=Path)
    args=parser.parse_args()
    result,_=report(args.base,args.active)
    print('COMPONENT_PROFILE_AUDIT',result['completed'],'/',result['total'])
