"""Audited post-SLAM metrics and the separate optional implementation ablation."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS=['accuracy_cm','completion_cm','completion_ratio_pct','depth_l1_cm']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as file:
        w=csv.DictWriter(file,fields);w.writeheader();w.writerows(rows)


def timing_gain(value):
    colour='green!35!black' if float(value)>0 else 'red!70!black' if float(value)<0 else 'black'
    return r'\textbf{\textcolor{'+colour+'}{'+f'{float(value):+.2f}'+'}}'


def reconstruction(base):
    import numpy as np
    declarations=[];inputs=[]
    for queue_name,manifest_name in [('reconstruction_v1','rtx_reconstruction_all.json'),('reconstruction_dp_extension_v1','rtx_reconstruction_dp_extension.json'),('reconstruction_composition_v1','rtx_reconstruction_composition.json')]:
        queue=base/'results'/queue_name;manifest=base/'experiments/manifests'/manifest_name
        if not manifest.exists():continue
        declarations.extend((queue,job) for job in json.loads(manifest.read_text())['jobs']);inputs.append(manifest)
    total=len(declarations)
    if not total:return dict(scored=0,total=0,comparisons=[],groups=[],failures=[]),[]
    assert len({job['id'] for _,job in declarations})==total,'Duplicate scoring job across manifests'
    rows=[];failures=[]
    for queue,job in declarations:
        done=queue/(job['id']+'.done.json');failed=queue/(job['id']+'.failed.json')
        if failed.exists():failures.append(json.loads(failed.read_text()))
        if not done.exists():continue
        path=queue/job['id']/'metrics.json';record=json.loads(path.read_text())
        assert sha(path)==json.loads(done.read_text())['metrics_sha256']
        assert record['scorer_sha256']==sha(base/'experiments/reconstruction_runtime_v2/checked_reconstruction.py')
        assert all(record[k]==job[k] for k in ('id','system','dataset','scene','condition','hash_size','seed'))
        assert all(math.isfinite(v) and v>=0 for v in record['metrics'].values())
        assert record['metrics']['completion_ratio_pct']<=100
        error_path=queue/job['id']/'depth_errors.npz'
        with np.load(error_path) as errors:
            assert len(errors['valid_view_ids'])==record['depth_valid_views']
            assert abs(float(errors['absolute_depth_error_m'].mean())*100-record['metrics']['depth_l1_cm'])<1e-6
        rows.append(dict(**{k:record[k] for k in ('id','system','dataset','scene','condition','hash_size','seed')},**record['metrics'],wall_seconds=record['wall_seconds']))
        inputs.extend([path,error_path])
    byscene={}
    for row in rows:
        key=(row['system'],row['dataset'],row['scene'],row['hash_size'])
        group=byscene.setdefault(key,{}).setdefault(row['condition'],{})
        assert row['seed'] not in group;group[row['seed']]=row
    comparisons=[]
    for key,group in sorted(byscene.items()):
        for variant in ('MC','DP','MC_DP'):
            if set(group.get('BASE',{}))!={0,1,2} or set(group.get(variant,{}))!={0,1,2}:continue
            row=dict(system=key[0],dataset=key[1],scene=key[2],hash_size=key[3],variant=variant)
            for metric in METRICS:
                for c in ('BASE',variant):
                    values=[group[c][s][metric] for s in (0,1,2)]
                    row[c+'_'+metric]=statistics.mean(values);row[c+'_'+metric+'_sd']=statistics.stdev(values)
                row[metric+'_absolute_change']=row[variant+'_'+metric]-row['BASE_'+metric]
            comparisons.append(row)
    grouped={}
    for row in comparisons:grouped.setdefault((row['system'],row['dataset'],row['hash_size'],row['variant']),[]).append(row)
    aggregates=[]
    def aggregate(key,scenes,sensitivity=None):
        system,dataset,capacity,variant=key
        result=dict(system=system,dataset=dataset,hash_size=capacity,variant=variant,scene_count=len(scenes),expected_scenes=8 if dataset=='Replica' else 7,scenes=sorted(r['scene'] for r in scenes),sensitivity=sensitivity)
        for metric in METRICS:
            for c in ('BASE',variant):
                seed_means=[statistics.mean(byscene[(system,dataset,r['scene'],capacity)][c][s][metric] for r in scenes) for s in (0,1,2)]
                result[c+'_'+metric]=statistics.mean(seed_means);result[c+'_'+metric+'_sd']=statistics.stdev(seed_means)
            result[metric+'_absolute_change']=result[variant+'_'+metric]-result['BASE_'+metric]
        return result
    for key,scenes in sorted(grouped.items()):
        aggregates.append(aggregate(key,scenes))
        if key[1]=='Replica' and len(scenes)==8:
            aggregates.append(aggregate(key,[r for r in scenes if r['scene']!='office4'],sensitivity='exclude_office4_only'))
    result=dict(scored=len(rows),total=total,declared_variants=sorted({job['condition'] for _,job in declarations}),complete_scene_comparisons=len(comparisons),groups=aggregates,failures=failures,protocol='checked common double-sided EGL, GT+virtual culling, threshold-0.1m vertex ICP, 200k surface samples with fixed seed 0, 1000 fixed metric poses, GT valid-depth mask, missing prediction penalized as zero',legacy_depth_caches_reused=False)
    output=base/'results/reconstruction_v1_analysis';output.mkdir(exist_ok=True)
    write_csv(output/'per_run.csv',rows);write_csv(output/'per_scene.csv',comparisons);write_csv(output/'dataset_aggregates.csv',aggregates)
    (output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    text=f'# Common reconstruction results\n\n{len(rows)}/{total} meshes scored; {len(comparisons)} complete three-seed scene comparisons; {len(failures)} explicit failures.\n\n'
    text+='Means give scenes equal weight. SD is across three seed-wise dataset averages. Comparison conditions use exactly the same scenes; incomplete/failed comparisons stay explicit. Completion-ratio differences are percentage points.\n\n'
    text+='Both systems share identical Replica raw GT camera matrices. All geometry is in metres. The EGL renderer contains one mesh at a time, renders both face directions, converts the supplied metric OpenCV poses explicitly and uses fresh hash-bound GT caches. It is a declared common evaluator adaptation, not proof of exact numerical equivalence to old published/evaluator tables. No causal explanation of a historical discrepancy is inferred from this change.\n\n'
    text+='| System / dataset / capacity / comparison | Coverage | Condition | Accuracy cm | Completion cm | Completion % | Depth L1 cm |\n|---|---:|---|---:|---:|---:|---:|\n'
    for row in aggregates:
        name=f"{row['system']}/{row['dataset']}/H{row['hash_size']}/{row['variant']}"+(('/'+row['sensitivity']) if row['sensitivity'] else '')
        for c in ('BASE',row['variant']):
            values=[f"{row[c+'_'+m]:.3f} ± {row[c+'_'+m+'_sd']:.3f}" for m in METRICS]
            text+='| '+name+f" | {row['scene_count']}/{row['expected_scenes']} | {c} | "+' | '.join(values)+' |\n'
    (output/'REPORT.md').write_text(text)
    return result,inputs


def engineering(base):
    queue=base/'results/engineering_v1';manifest=base/'experiments/manifests/rtx_engineering_office0.json'
    if not manifest.exists():return None,[]
    rows=[];inputs=[];configs=[]
    from report_support_reference import rigid_rmse
    import numpy as np
    for job in json.loads(manifest.read_text())['jobs']:
        if not (queue/(job['id']+'.done.json')).exists():continue
        directory=queue/job['id'];p=directory/'summary.json';summary=json.loads(p.read_text())
        assert summary['full_sequence'] and summary['frames']==2000 and summary['finite_parameters_and_poses']
        assert summary['seed']==job['seed'] and summary['variant']==job['variant'] and summary['warmup_frames']==20
        config=json.loads((directory/'config.json').read_text())
        assert config['round2']['engineering']==job['engineering']
        config['round2'].pop('engineering');config['round2'].pop('seed');config['data'].pop('output');configs.append(config)
        assert sha(directory/'source_manifest.json')==sha(base/'experiments/coslam_engineering_runtime_v1/RUNTIME_MANIFEST.json')
        with np.load(directory/'trajectories.npz') as poses:
            assert abs(rigid_rmse(poses['gt'],poses['estimated'])-summary['ate_rmse_m'])<1e-6
            assert abs(rigid_rmse(poses['gt'],poses['corrected'])-summary['corrected_ate_rmse_m'])<1e-6
        with (directory/'calls.csv').open() as f:calls=list(csv.DictReader(f))
        row=dict(variant=job['variant'],seed=job['seed'],ate_cm=100*summary['ate_rmse_m'],corrected_ate_cm=100*summary['corrected_ate_rmse_m'])
        for stage,field in [('tracking','tracking_wall_ms'),('bundle_adjustment','ba_wall_ms')]:
            chosen=[r for r in calls if r['stage']==stage and int(r['frame_id'])>=20]
            value=statistics.mean(float(r['wall_ms']) for r in chosen)
            assert abs(value-summary[stage]['mean_wall_ms'])<1e-6
            assert summary[stage]['queries_pruned_pct']==0
            row[field]=value
        rows.append(row);inputs.extend([p,directory/'calls.csv',directory/'trajectories.npz'])
    if any(set(r['seed'] for r in rows if r['variant']==c)!={0,1,2} for c in ('BASE','ENG')):return None,inputs
    assert all(c==configs[0] for c in configs)
    result=dict(scene='office0',hash_size=16,seeds=[0,1,2],scope='separate engineering ablation, native pose aliases preserved, no pruning/Morton')
    for field in ('ate_cm','corrected_ate_cm','tracking_wall_ms','ba_wall_ms'):
        for c in ('BASE','ENG'):
            values=[r[field] for r in rows if r['variant']==c]
            result[c+'_'+field]=statistics.mean(values);result[c+'_'+field+'_sd']=statistics.stdev(values)
        result[field+'_change_pct']=100*(result['ENG_'+field]/result['BASE_'+field]-1)
    output=base/'results/engineering_v1_analysis';output.mkdir(exist_ok=True)
    write_csv(output/'per_run.csv',rows);(output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'REPORT.md').write_text('# Separate Co engineering ablation\n\nThree paired seeds; native office0 H16. No pruning or Morton. Configuration parity, raw-pose ATE and complete-call wall times were checked. Public pose aliases are preserved. Full values are in SUMMARY.json and per_run.csv. No expected gain or online bitwise-equivalence claim is imposed.\n\n'+json.dumps(result,indent=2)+'\n')
    return result,inputs


def reconstruction_tex(result):
    groups={(r['system'],r['dataset'],r['hash_size'],r['variant']):r for r in result['groups'] if not r['sensitivity']}
    planned=[('Co-SLAM','Replica',16,'MC'),('Co-SLAM','Replica',19,'MC'),('Co-SLAM','Synthetic',16,'MC'),('EC-SLAM','Replica',16,'MC'),('EC-SLAM','Replica',19,'MC'),('Co-SLAM','Replica',16,'DP'),('Co-SLAM','Synthetic',16,'DP')]
    if 'MC_DP' in result.get('declared_variants',[]):planned += [('Co-SLAM','Replica',16,'MC_DP'),('Co-SLAM','Synthetic',16,'MC_DP')]
    text=r'\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}'+'\n'
    text+=r'\caption{Common reconstruction evaluation: equal scene weights, three-seed mean $\pm$ sample SD. Coverage $n/N$ is complete matched scenes/full suite. Co/EC denote Co-SLAM/EC-SLAM. Each BASE uses the same subset as its MC/DP/MC+DP comparison. H19 is a sensitivity; pruning is under author review. Completion ratio is a percentage.}'+'\n'
    text+=r'\label{tab:quality_round2}\resizebox{\linewidth}{!}{\begin{tabular}{llrrrrr}\toprule System / dataset / pair &Cond.&$n/N$&Acc. (cm)&Comp. (cm)&Comp. (\%)&Depth (cm)\\\midrule'+'\n'
    for key in planned:
        row=groups.get(key);system,dataset,cap,variant=key;n=row['scene_count'] if row else 0;expected=8 if dataset=='Replica' else 7
        name=f"{'Co' if system=='Co-SLAM' else 'EC'}/{dataset}/H{cap} {variant.replace('_','+')}"
        for c in ('BASE',variant):
            values=[f"${row[c+'_'+m]:.2f}\\pm{row[c+'_'+m+'_sd']:.2f}$" if row else r'\pending' for m in METRICS]
            text+='&'.join([name,c.replace('_','+'),f'{n}/{expected}',*values])+r'\\'+'\n'
    text+=r'\bottomrule\end{tabular}}\end{table}'+'\n'
    text+=f"{result['scored']}/{result['total']} saved meshes have checked scores; {len(result['failures'])} recorded failures. "
    text+=r'All-eight-scene Replica is primary; a seven-scene office4-exclusion sensitivity is kept separately in the supporting report, never substituted silently. The common EGL evaluator uses isolated double-sided geometry and fresh hash-bound GT caches. It is a declared adaptation; agreement with an old evaluator is not assumed.'+'\n'
    return text


def engineering_tex(result):
    text=r'\begin{table}[htbp]\centering\small\caption{Co engineering: BA comparison, office0 H16, three paired seeds. Host wall ms per complete BA invocation (ten iterations, setup and pose finalization). BA every fifth frame; frames 0--19 excluded. No pruning or Morton.}\label{tab:engineering}\resizebox{\linewidth}{!}{\begin{tabular}{lrrr}\toprule Condition&BA (ms/call)&Direct ATE (cm)&Corrected ATE (cm)\\\midrule'+'\n'
    for c in ('BASE','ENG'):
        values=[f"${result[c+'_'+f]:.3f}\\pm{result[c+'_'+f+'_sd']:.3f}$" if result else r'\pending' for f in ('ba_wall_ms','ate_cm','corrected_ate_cm')]
        text+='&'.join([c,*values])+r'\\'+'\n'
    if result:
        text+=r'\midrule '+r'\textbf{BA time gain (\%)}&'+timing_gain(-result['ba_wall_ms_change_pct'])+r'&---&---\\'+'\n'
    text+=r'\bottomrule\end{tabular}}\end{table}'+'\n'
    if result:text+='Bold gains are time reductions; negative red values are slowdowns. These implementation benefits are not attributed to scientific pruning or Morton.\n'
    return text


def ec_timing(base):
    queue=base/'results/ec_timing_v1';manifest=queue/'job_manifest.json'
    if not manifest.exists():return dict(completed=0,total=96,groups=[]),[]
    rows=[];inputs=[]
    for job in json.loads(manifest.read_text())['jobs']:
        if not (queue/(job['id']+'.done.json')).exists():continue
        directory=queue/job['id'];path=directory/'summary.json';summary=json.loads(path.read_text())
        assert summary['purpose']=='native_complete_call_profile' and summary['warmup_frames']==20 and summary['full_sequence'] and summary['finite_parameters_and_poses']
        assert summary['condition']==job['condition'] and summary['seed']==job['seed'] and summary['hash_size']==job['hash_size']
        assert sha(directory/'source_manifest.json')==sha(base/'experiments/ec_timing_runtime_v1/RUNTIME_MANIFEST.json')
        with (directory/'calls.csv').open() as f:calls=list(csv.DictReader(f))
        row=dict(scene=job['scene'],hash_size=job['hash_size'],condition=job['condition'],seed=job['seed'],profiled_ate_cm=100*summary['ate_rmse_m'],profiled_corrected_ate_cm=100*summary['corrected_ate_rmse_m'])
        for stage,prefix,count in [('tracking','tracking',1980),('bundle_adjustment','mapping',397)]:
            selected=[r for r in calls if r['stage']==stage and int(r['frame_id'])>=20]
            assert len(selected)==count
            for source,label in [('wall_ms','wall_ms'),('cuda_ms','cuda_ms')]:
                value=statistics.mean(float(r[source]) for r in selected)
                assert math.isfinite(value) and abs(value-summary[stage]['mean_'+source])<1e-6
                row[prefix+'_'+label]=value
        rows.append(row);inputs.extend([path,directory/'calls.csv',directory/'source_manifest.json'])
    grouped={}
    for r in rows:grouped.setdefault((r['scene'],r['hash_size']),{}).setdefault(r['condition'],{})[r['seed']]=r
    groups=[]
    for cap in (16,19):
        scenes=[s for (s,h),g in grouped.items() if h==cap and all(set(g.get(c,{}))=={0,1,2} for c in ('BASE','MC'))]
        if not scenes:continue
        result=dict(hash_size=cap,scene_count=len(scenes),expected_scenes=8,scenes=sorted(scenes))
        for field in ('tracking_wall_ms','mapping_wall_ms','tracking_cuda_ms','mapping_cuda_ms','profiled_ate_cm','profiled_corrected_ate_cm'):
            for c in ('BASE','MC'):
                values=[statistics.mean(grouped[(s,cap)][c][seed][field] for s in scenes) for seed in (0,1,2)]
                result[c+'_'+field]=statistics.mean(values);result[c+'_'+field+'_sd']=statistics.stdev(values)
            result[field+'_change_pct']=100*(result['MC_'+field]/result['BASE_'+field]-1)
        groups.append(result)
    result=dict(completed=len(rows),total=96,groups=groups,scope='Native active worker calls, synchronized boundaries including preparation/pose finalization; handshake waits separate; native threads retained; overlapping intervals not additive',quality_scope='Instrumented ATE retained separately for timing-protocol sensitivity; primary reconstruction/quality remains the native quality campaign')
    output=base/'results/ec_timing_v1_analysis';output.mkdir(exist_ok=True)
    write_csv(output/'per_run.csv',rows);(output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'REPORT.md').write_text('# Native EC complete-call timing\n\n'+json.dumps(result,indent=2)+'\n\nPer-run active-call times were recomputed from CSVs. Native frame coverage, source identity and three paired seeds were checked. Timers can perturb native thread interleaving; profiling ATE is separate sensitivity evidence and never replaces primary unprofiled quality/mesh results. Do not sum worker intervals into scene FPS. All measurements, including slowdowns, are retained.\n')
    return result,inputs


def ec_timing_tex(result):
    groups={r['hash_size']:r for r in result['groups']}
    text=r'\begin{table}[htbp]\centering\scriptsize\caption{EC: host wall ms per active tracking/mapping worker invocation (all iterations, setup and pose finalization), excluding handshake waits. Overlapping worker intervals are not frame latency or FPS. Equal scene weights, three-seed SD; frames 0--19 excluded. Bold gains are time reductions; red negative values are slowdowns. Coverage is complete/full Replica scenes.}\label{tab:ec_calls}\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrr}\toprule Capacity ($n/N$)&BASE TR&MC TR&\textbf{Gain (\%)}&BASE map&MC map&\textbf{Gain (\%)}\\\midrule'+'\n'
    for cap in (16,19):
        row=groups.get(cap);values=[]
        for phase in ('tracking','mapping'):
            for c in ('BASE','MC'):values.append(f"${row[c+'_'+phase+'_wall_ms']:.3f}\\pm{row[c+'_'+phase+'_wall_ms_sd']:.3f}$" if row else r'\pending')
            values.append(timing_gain(-row[phase+'_wall_ms_change_pct']) if row else r'\pending')
        text+='&'.join([f"H{cap} ({row['scene_count'] if row else 0}/8)",*values])+r'\\'+'\n'
    return text+r'\bottomrule\end{tabular}}\end{table}'+'\n'
