#!/usr/bin/env python3
"""Update the working paper from audited complete comparisons; no GPU work."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parents[1]
PAPER = BASE/'paper_round2_placeholder'
REPLICA = ['office0','office1','office2','office3','office4','room0','room1','room2']
INVENTORY=json.loads((BASE/'experiments/ALL_CONFIGURED_NONREPLICA_INVENTORY.json').read_text())
SCENES = [('Replica', s,16) for s in REPLICA]+[(r['dataset'],r['scene'],r['native_hash_size']) for r in INVENTORY['scenes']]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as file:
        data=list(csv.DictReader(file))
    for row in data:
        assert int(row['paired_seeds'])==3
        for key,value in row.items():
            if key.startswith(('BASE_','MC_','DP_')) and key not in ('BASE_source','MC_source','DP_source'):
                assert math.isfinite(float(value)), (path,key)
    return data


def esc(value):
    return str(value).replace('_',r'\_').replace('%',r'\%')


def val(row, condition, field):
    if row is None:
        return r'\pending'
    mean=float(row[condition+'_'+field]);sd=row.get(condition+'_'+field+'_sd')
    return f"${mean:.3f}$" if sd is None else f"${mean:.3f}\\pm{float(sd):.3f}$"


def pct(value, red=False):
    text=f'{float(value):+.2f}'
    return r'\textcolor{red!70!black}{'+text+'}' if red else text


def gain(value):
    colour='green!35!black' if float(value)>0 else 'red!70!black' if float(value)<0 else 'black'
    return r'\textbf{\textcolor{'+colour+'}{'+f'{float(value):+.2f}'+'}}'


def dp_timing_scopes_table(phases,expected,orin=None,orin15=None):
    from paper_orin_results import dp_phase_rows
    all_rows=[dict(r,platform='RTX') for r in phases]+dp_phase_rows(orin)+dp_phase_rows(orin15)
    lookup={(r['dataset'],r['stage'],r['platform']):r for r in all_rows}
    platforms=['RTX']+[p for p in ('MAXN','15W') if any(r['platform']==p for r in all_rows)]
    text=r'\begin{table}[htbp]\centering\scriptsize\setlength{\tabcolsep}{3pt}'+'\n'
    text+=r'\caption{DP timing at two scopes, ms per stage invocation. Iteration total sums optimizer-iteration CUDA-event intervals; it excludes setup/finalization and gaps, and is not loop wall time. Complete invocation is synchronized host wall including setup, all updates and pose finalization. Stage ($I$) gives update count. Tracking once/frame; BA every fifth. Frames 0--19 excluded; equal scene weights, mean $\pm$ seed SD: three RTX/two MAXN seeds; 15 W uses seed 0 only without seed SD. Bold gains are time reductions, green positive and red slower. Replica/Synthetic/TUM H16; ScanNet pools all six native H12/H14/H16 scenes equally.}'+'\n'
    text+=r'\label{tab:dp_wall}\resizebox{\linewidth}{!}{\begin{tabular}{lllrrrrrr}\toprule Dataset ($n/N$)&GPU&Stage ($I$)&\multicolumn{3}{c}{Optimizer iterations (CUDA ms/call)}&\multicolumn{3}{c}{Complete invocation (host ms/call)}\\\cmidrule(lr){4-6}\cmidrule(lr){7-9}&&&BASE&DP&\textbf{Gain (\%)}&BASE&DP&\textbf{Gain (\%)}\\\midrule'+'\n'
    for dataset,total in expected.items():
        name='TUM' if dataset=='Tum' else dataset
        for platform in platforms:
            for stage,short in [('tracking','TR'),('bundle_adjustment','BA')]:
                row=lookup.get((dataset,stage,platform))
                label=esc(name)+f' ({total if row else 0}/{total})' if stage=='tracking' else ''
                if row:
                    updates=int(row['BASE_iterations_per_call']);assert updates==int(row['DP_iterations_per_call'])
                    body=[label,platform if stage=='tracking' else '',f'{short} ({updates})']
                    for field in ('summed_iteration_cuda_ms_per_call','call_host_wall_ms'):
                        body.extend([val(row,'BASE',field),val(row,'DP',field),gain(row[field+'_reduction_pct'])])
                else:body=[label,platform,short]+[r'\pending']*6
                text+='&'.join(body)+r'\\'+'\n'
        if dataset!='ScanNet':text+=r'\addlinespace[2pt]'+'\n'
    return text+r'\bottomrule\end{tabular}}\end{table}'+'\n'


def table(caption,label,header,body):
    cols='l'+'r'*(len(header)-1)
    return '\n'.join([r'\begin{table}[htbp]\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\resizebox{\linewidth}{!}{\begin{tabular}{'+cols+'}',r'\toprule','&'.join(header)+r'\\\midrule',*['&'.join(row)+r'\\' for row in body],r'\bottomrule\end{tabular}}',r'\end{table}',''])


def aggregate(scene_rows,run_rows,dataset,condition,capacity=None,fields=None):
    """Equal scene weights; sample SD over three seed-wise dataset means."""
    fields=fields or ['ate_cm','corrected_ate_cm','tracking_wall_ms','ba_wall_ms']
    selected=[r for r in scene_rows if r['dataset']==dataset and (capacity is None or int(r['hash_size'])==capacity)]
    if not selected:return None
    names={r['scene'] for r in selected}
    assert len(names)==len(selected), 'Do not mix two capacities of the same scene'
    found={}
    for row in run_rows:
        if row['dataset']!=dataset or row['scene'] not in names or row['condition'] not in ('BASE',condition):continue
        if capacity is not None and int(row['hash_size'])!=capacity:continue
        key=(row['scene'],row['condition'],int(row['seed']))
        assert key not in found, 'Duplicate run in aggregate'
        found[key]=row
    result=dict(dataset=dataset,hash_size=capacity,scene_count=len(names),scene_names=sorted(names))
    for field in fields:
        for c in ('BASE',condition):
            seed_means=[statistics.mean(float(found[(scene,c,seed)][field]) for scene in names) for seed in (0,1,2)]
            result[c+'_'+field]=statistics.mean(seed_means)
            result[c+'_'+field+'_sd']=statistics.stdev(seed_means)
        change=100*(result[condition+'_'+field]/result['BASE_'+field]-1)
        result[field+'_change_pct']=change
        result[field+'_increase_pct']=change
        result[field+'_reduction_pct']=-change
    return result


def raw_rows(path):
    with path.open() as file:return list(csv.DictReader(file))


def compact_tables(dp,mc,ec,dp_runs,mc_runs,ec_runs,now,timed=None,dp_phases=None,orin=None,orin15=None):
    from paper_orin_results import find,co_morton_table
    expected={'Replica':8,'Synthetic':7,'Tum':3,'ScanNet':6}
    dpa={d:aggregate(list(dp.values()),dp_runs,d,'DP') for d in expected}
    mca={(d,cap):aggregate(list(mc.values()),mc_runs,d,'MC',cap) for d,cap in [('Replica',16),('Replica',19),('Synthetic',None),('Tum',None),('ScanNet',None)]}
    ec_rows=[dict(dataset='Replica',scene=s,hash_size=h,**r) for (s,h),r in ec.items()]
    eca={cap:aggregate(ec_rows,ec_runs,'Replica','MC',cap,fields=['ate_cm','corrected_ate_cm']) for cap in (16,19)}
    def label(dataset,row,cap=None):
        name={'Tum':'TUM'}.get(dataset,dataset)+(f' H{cap}' if cap else '')
        return esc(name)+f" ({row['scene_count'] if row else 0}/{expected[dataset]})"
    def quality_panel(groups,condition):
        body=[]
        for dataset,cap,row in groups:
            body.append([label(dataset,row,cap),val(row,'BASE','ate_cm'),val(row,condition,'ate_cm'),pct(row['ate_cm_change_pct']) if row else r'\pending',val(row,'BASE','corrected_ate_cm'),val(row,condition,'corrected_ate_cm'),pct(row['corrected_ate_cm_change_pct']) if row else r'\pending'])
        return body
    def timing_panel(groups,condition):
        body=[]
        for dataset,cap,row in groups:
            body.append([label(dataset,row,cap),val(row,'BASE','tracking_wall_ms'),val(row,condition,'tracking_wall_ms'),gain(row['tracking_wall_ms_reduction_pct']) if row else r'\pending',val(row,'BASE','ba_wall_ms'),val(row,condition,'ba_wall_ms'),gain(row['ba_wall_ms_reduction_pct']) if row else r'\pending'])
        return body
    text=r'\paragraph{Verified progress ('+now+').} '
    text+=f'{len(dp)}/24 Co BASE/DP, {len(mc)}/32 Co BASE/MC and {len(ec)}/16 EC quality scene/capacity comparisons have three paired RTX seeds. '
    text+=(f"MAXN: {orin['verified_runs']} audited full runs, two paired seeds; ScanNet pending: {orin['pending_scannet_runs']}.\n" if orin else 'Orin results await import.\n')
    if orin15:text+='15 W: 130 audited benchmarks, seed 0 only; five audited component profiles. No across-seed SD.\n'
    text+=r'\paragraph{Coverage and aggregation.} The suite contains Replica 8, Synthetic 7, TUM 3 and ScanNet 6 scenes. Means use matched subsets and equal scene weights; SD is across seed-wise dataset means. Coverage $n/N$ denotes complete/full scenes. Negative outcomes remain visible; H19 is a sensitivity. Synthetic ma corrects the duplicated public data path.'+'\n'
    text+=r'\subsection{Stage Timing Across Platforms}'+'\n'
    row=mca[('Replica',16)];runtime=[]
    for c in ('BASE','MC'):
        value=r'\pending' if row is None else r'\shortstack{TR: '+val(row,c,'tracking_wall_ms')+r'\\BA: '+val(row,c,'ba_wall_ms')+'}'
        embedded=find(orin,'Co-SLAM','Replica',16)
        maxn=r'\pending' if embedded is None else r'\shortstack{TR: '+val(embedded,c,'tracking_wall_ms')+r'\\BA: '+val(embedded,c,'ba_wall_ms')+'}'
        low=find(orin15,'Co-SLAM','Replica',16)
        low_text=r'\pending' if low is None else r'\shortstack{TR: '+val(low,c,'tracking_wall_ms')+r'\\BA: '+val(low,c,'ba_wall_ms')+'}'
        runtime.append(['Co-SLAM '+c,value,maxn,low_text])
    ec_time=next((r for r in (timed or {}).get('groups',[]) if r['hash_size']==16),None)
    for c in ('BASE','MC'):
        value=r'\pending' if ec_time is None else r'\shortstack{TR: '+val(ec_time,c,'tracking_wall_ms')+r'\\Map: '+val(ec_time,c,'mapping_wall_ms')+'}'
        embedded=find(orin,'EC-SLAM','Replica',16)
        maxn=r'\pending' if embedded is None else r'\shortstack{TR opt.: '+val(embedded,c,'tracking_wall_ms')+r'\\Map opt.: '+val(embedded,c,'ba_wall_ms')+'}'
        low=find(orin15,'EC-SLAM','Replica',16)
        low_text=r'\pending' if low is None else r'\shortstack{TR opt.: '+val(low,c,'tracking_wall_ms')+r'\\Map opt.: '+val(low,c,'ba_wall_ms')+'}'
        runtime.append(['EC-SLAM '+c+(f" ({ec_time['scene_count']}/8)" if ec_time else ''),value,maxn,low_text])
    text+=table(r'Replica H16, host wall ms/call. Co: complete tracking/BA invocations. EC RTX: active worker bodies excluding handshake waits; EC both Orin modes: optimizer-method calls excluding outer frame preparation/finalization (different scopes; see Table~\ref{tab:ec_calls}). Worker intervals overlap and are not FPS. Equal scene weights; mean $\pm$ SD over three RTX/two MAXN seeds; 15 W seed 0 only, no seed SD. Frames 0--19 excluded.', 'tab:runtime_round2',['System / condition','RTX 4070','Orin MAXN','Orin 15 W'],runtime)
    quality_header=['Dataset ($n/N$)','BASE direct','Variant direct',r'$\Delta$ (\%)','BASE corr.','Variant corr.',r'$\Delta$ (\%)']
    time_header=['Dataset ($n/N$)','BASE TR','Variant TR',r'\textbf{Gain (\%)}','BASE BA','Variant BA',r'\textbf{Gain (\%)}']
    text+=r'\subsection{Conservative Pruning: Quality and Cost}'+'\n'
    groups=[(d,None,r) for d,r in dpa.items()]
    quality_body=quality_panel(groups,'DP')
    decision=[]
    for index,(dataset,_,row) in enumerate(groups):
        scenes=[r for r in dp.values() if r['dataset']==dataset]
        count=len(scenes)
        failed=sum(float(r['ate_cm_increase_pct'])>15 or float(r['corrected_ate_cm_increase_pct'])>15 for r in scenes)
        tr_pass=sum(float(r['tracking_wall_ms_reduction_pct'])>=5 for r in scenes)
        ba_pass=sum(float(r['ba_wall_ms_reduction_pct'])>=25 for r in scenes)
        quality_body[index].append((r'\textcolor{red!70!black}{' if failed else '')+f'{failed}/{count}'+('}' if failed else '') if count else r'\pending')
        decision.append(dict(dataset=dataset,tested_scenes=count,ate_failed_scenes=failed,tracking_passed_scenes=tr_pass,ba_passed_scenes=ba_pass))
    dp_quality_header=[h.replace('Variant','DP') for h in quality_header]+[r'ATE fails']
    text+=table(r'Conservative DP versus dense BASE: ATE (cm), equal-weight scene means and three-seed sample SD. Replica, Synthetic and TUM use H16; ScanNet retains scene-specific H12/H14/H16. Positive changes denote degradation. Coverage is tested/full configured scenes. ATE fails counts tested scenes exceeding the previous 15\% reference in either saved convention, using per-scene three-seed means; it is advisory.','tab:ate_round2',dp_quality_header,quality_body)
    text+=dp_timing_scopes_table(dp_phases or [],expected,orin,orin15)
    if dp:
        failures=sum(float(r['ate_cm_increase_pct'])>15 or float(r['corrected_ate_cm_increase_pct'])>15 for r in dp.values())
        worst=max(float(r['corrected_ate_cm_increase_pct']) for r in dp.values())
        lo=min(float(r['ba_wall_ms_reduction_pct']) for r in dp.values());hi=max(float(r['ba_wall_ms_reduction_pct']) for r in dp.values())
        text+=r'\textbf{Advisory reference checks.} '
        text+=f'{failures}/{len(dp)} scenes exceed the previous 15\\% ATE reference in at least one saved convention; worst corrected degradation is {worst:.2f}\\%. '
        text+=f'Individual-scene BA reductions span {lo:.2f}--{hi:.2f}\\%; tracking slowdowns remain visible. '
        text+=r'All scenes use the same frozen DP; acceptance awaits author review of complete dataset means and scene exceptions.'+'\n'
        outliers=[(f"{r['dataset']}/{r['scene']}",float(r['ate_cm_increase_pct']),float(r['corrected_ate_cm_increase_pct'])) for r in dp.values() if float(r['ate_cm_increase_pct'])>15 or float(r['corrected_ate_cm_increase_pct'])>15]
        text+='ATE exceptions (direct/corrected): '+', '.join(esc(name)+f' {direct:+.2f}\\%/{corrected:+.2f}\\%' for name,direct,corrected in outliers)+'.\n'
    text+=r'\subsection{Morton Execution: Quality and Integrated Cost}'+'\n'
    groups=[(d,cap,r) for (d,cap),r in mca.items()]
    text+=table(r'Co-SLAM BASE/MC ATE (cm), equal scene weights and three-seed sample SD. Coverage $n/N$ includes only complete paired scenes. H19 is a separate sensitivity.','tab:morton_ate',quality_header,quality_panel(groups,'MC'))
    if orin:text+=co_morton_table(groups,orin,val,gain,table,orin15)
    else:text+=table(r'Co BASE/MC: host wall ms per complete tracking/BA call (all iterations, setup, pose finalization and ordering). Tracking once/frame after initialization; BA every fifth frame. Equal scene weights, three-seed mean $\pm$ SD; frames 0--19 excluded. Bold gains are reductions; red negative values are slowdowns.','tab:morton_wall',time_header,timing_panel(groups,'MC'))
    if mc and all(float(r['tracking_wall_ms_change_pct'])>0 and float(r['ba_wall_ms_change_pct'])>0 for r in mc.values()):
        text+=r'\textbf{RTX Morton comparisons are slower in tracking and BA.} MAXN outcomes are shown separately; kernel gains cannot replace call measurements.'+'\n'
    text+=r'\subsection{EC-SLAM Quality Progress}'+'\n'
    text+=table(r'EC-SLAM Replica BASE/MC ATE (cm), equal scene weights and three-seed sample SD. Native concurrent workers are preserved. Coverage $n/N$ remains explicit. Quality-run scene elapsed time is not complete-call tracking/mapping timing.','tab:ec_ate',quality_header,quality_panel([('Replica',h,r) for h,r in eca.items()],'MC'))
    record=dict(aggregation='equal scene weights; sample SD over three RTX/two MAXN seed-wise dataset averages; 15 W seed 0 only, SD unavailable; matched scene subsets',DP=list(dpa.values()),DP_scene_decisions=decision,DP_timing_scopes=dp_phases or [],MC=list(mca.values()),EC=list(eca.values()),ORIN_MAXN=(orin or {}).get('groups',[]),ORIN_15W=(orin15 or {}).get('groups',[]))
    return text,record


def ec_comparisons():
    queue=BASE/'results/ec_quality_v1'
    if not (queue/'job_manifest.json').is_file():
        return {},[]
    from run_ec_quality_round2 import rigid_rmse
    import numpy as np
    import statistics
    import yaml
    grouped={};sources=[]
    for job in json.loads((queue/'job_manifest.json').read_text())['jobs']:
        if not (queue/(job['id']+'.done.json')).is_file():continue
        directory=queue/job['id'];summary=json.loads((directory/'summary.json').read_text())
        assert summary['full_sequence'] and summary['finite_parameters_and_poses']
        assert summary['frames']==2000 and summary['seed']==job['seed'] and summary['condition']==job['condition']
        with np.load(directory/'trajectories.npz') as poses:
            for field,key in [('ate_rmse_m','estimated'),('corrected_ate_rmse_m','corrected')]:
                assert np.isfinite(poses[key]).all()
                assert abs(rigid_rmse(poses[key],poses['ground_truth'])-summary[field])<1e-6
        cfg=yaml.safe_load((directory/'config.yaml').read_text());cfg['data'].pop('output')
        summary['_cfg']=cfg
        summary['_source']=digest(directory/'source_manifest.json')
        assert summary['hash_size']==job['hash_size']
        key=(job['scene'],job['hash_size'])
        grouped.setdefault(key,{}).setdefault(job['condition'],{})[job['seed']]=summary
        sources += [directory/'summary.json',directory/'trajectories.npz',directory/'source_manifest.json',directory/'config.yaml']
    result={}
    for key,groups in grouped.items():
        if any(set(groups.get(c,{}))!={0,1,2} for c in ('BASE','MC')):continue
        for seed in (0,1,2):
            assert groups['BASE'][seed]['_cfg']==groups['MC'][seed]['_cfg']
            assert groups['BASE'][seed]['_source']==groups['MC'][seed]['_source']
        row={}
        for outfield,infield in [('ate_cm','ate_rmse_m'),('corrected_ate_cm','corrected_ate_rmse_m')]:
            for c in ('BASE','MC'):
                values=[100*groups[c][s][infield] for s in (0,1,2)]
                row[c+'_'+outfield]=statistics.mean(values)
                row[c+'_'+outfield+'_sd']=statistics.stdev(values)
            row[outfield+'_change_pct']=100*(row['MC_'+outfield]/row['BASE_'+outfield]-1)
        result[key]=row
    return result,sources


def update(audit_live=False):
    dp_root=BASE/'results/support_reference_v1_analysis'
    if (BASE/'experiments/manifests/rtx_dp_missing_scenes.json').exists():
        dp_root=BASE/'results/dp_all_scenes_analysis'
        with (BASE/'results/paper_dp_full_audit.log').open('w') as log:
            subprocess.run([sys.executable,str(BASE/'experiments/report_dp_all_scenes.py'),'--output',str(dp_root)],stdout=log,stderr=subprocess.STDOUT,check=True)
    mc_root=BASE/'results/morton_all_native_v1_analysis'
    if audit_live:
        queues=['support_reference_v1','morton_core_v1','replica_native_extension_v1','capacity_office0_h19_v1','replica_h19_extension_v1','all_nonreplica_extension_v1']
        argv=[sys.executable,str(BASE/'experiments/report_morton_reference.py')]
        for name in queues:
            if (BASE/'results'/name/'queue_identity.json').exists():argv += ['--queue',str(BASE/'results'/name)]
        mc_root=BASE/'results/morton_live_verified_analysis'
        argv += ['--output',str(mc_root)]
        with (BASE/'results/paper_morton_audit.log').open('w') as log:
            subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
    sources=[dp_root/'per_scene.csv',dp_root/'ANALYSIS_STATUS.json',mc_root/'per_scene.csv',mc_root/'ANALYSIS_STATUS.json']
    dp={(r['dataset'],r['scene']):r for r in rows(sources[0])}
    mc={(r['dataset'],r['scene'],int(r['hash_size'])):r for r in rows(sources[2])}
    ec,extra=ec_comparisons();sources+=extra
    now=datetime.now(ZoneInfo('Europe/Paris')).strftime('%d %B %Y, %H:%M Paris')
    sources += [dp_root/'per_run.csv',mc_root/'per_run.csv']
    dp_runs=raw_rows(dp_root/'per_run.csv');mc_runs=raw_rows(mc_root/'per_run.csv')
    ec_runs=[]
    for path in extra:
        if path.name!='summary.json':continue
        summary=json.loads(path.read_text())
        ec_runs.append(dict(dataset='Replica',scene=summary['scene'],hash_size=summary['hash_size'],condition=summary['condition'],seed=summary['seed'],ate_cm=100*summary['ate_rmse_m'],corrected_ate_cm=100*summary['corrected_ate_rmse_m']))
    from report_remaining_results import reconstruction,engineering,reconstruction_tex,engineering_tex,ec_timing,ec_timing_tex
    timed,timed_inputs=ec_timing(BASE);sources.extend(timed_inputs)
    with (BASE/'results/paper_dp_iteration_audit.log').open('w') as log:
        subprocess.run([sys.executable,str(BASE/'experiments/analyze_dp_phase_timing.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    iteration_root=BASE/'results/dp_phase_timing_analysis'
    iteration_status=iteration_root/'ANALYSIS_STATUS.json'
    dp_phases=json.loads(iteration_status.read_text())['comparisons']
    sources.extend([iteration_status,iteration_root/'per_run.csv',iteration_root/'dataset_phase_means.csv'])
    from paper_orin_results import load_orin,engineering_table,ec_table
    orin,orin_inputs=load_orin(BASE);sources.extend(orin_inputs)
    orin15,orin15_inputs=load_orin(BASE,'15W');sources.extend(orin15_inputs)
    text,aggregates=compact_tables(dp,mc,ec,dp_runs,mc_runs,ec_runs,now,timed,dp_phases,orin,orin15)
    recon,recon_inputs=reconstruction(BASE);eng,eng_inputs=engineering(BASE)
    sources.extend(recon_inputs+eng_inputs)
    from report_composition_results import report as composition_report,tex as composition_tex
    composition,composition_inputs=composition_report(BASE);sources.extend(composition_inputs)
    aggregates['RTX_COMPOSITION']=composition['groups']
    sources.append(BASE/'experiments/report_composition_results.py')
    sources.extend([BASE/'paper_round2_placeholder/abstracts/ABSTRACT_2026-10-20.txt', BASE/'experiments/prepare_round2_placeholder.py',BASE/'experiments/paper_results_template.tex',BASE/'experiments/paper_orin_results.py'])
    profile_protocol=BASE/'experiments/manifests/component_profile_protocol_v1.json'
    if profile_protocol.exists():sources.extend([profile_protocol,BASE/'docs/protocols/COMPONENT_PROFILING_PROTOCOL.md',BASE/'experiments/build_component_profile_protocol.py'])
    from report_component_profiles import report as component_report,table_tex as component_table_tex
    component_result,component_inputs=component_report(BASE);sources.extend(component_inputs)
    from report_ec_component_profiles import report as ec_component_report
    ec_component_result,ec_component_inputs=ec_component_report(BASE);sources.extend(ec_component_inputs)
    sources.append(BASE/'experiments/report_ec_component_profiles.py')
    if (BASE/'docs/protocols/EC_COMPONENT_PROFILING_PROTOCOL.md').exists():sources.append(BASE/'docs/protocols/EC_COMPONENT_PROFILING_PROTOCOL.md')
    sources.append(BASE/'experiments/report_component_profiles.py')
    aggregates['COMPONENT_PROFILES']=component_result['groups']
    aggregates['EC_COMPONENT_PROFILES']=ec_component_result['groups']
    PAPER.mkdir(exist_ok=True)
    (PAPER/'generated_verified_results.tex').write_text(text)
    (PAPER/'generated_reconstruction_results.tex').write_text(reconstruction_tex(recon))
    (PAPER/'generated_engineering_results.tex').write_text(engineering_table(eng,orin,val,gain,table,orin15) if orin else engineering_tex(eng))
    (PAPER/'generated_ec_timing_results.tex').write_text(ec_table(timed,orin,val,gain,table,orin15) if orin else ec_timing_tex(timed))
    from paper_orin_results import composition_table,composition_15w_table
    (PAPER/'generated_composition_results.tex').write_text(composition_tex(composition,table,val,gain)+composition_table(orin,val,gain,table)+composition_15w_table(orin15,val,gain,table))
    from paper_orin_results import load_component_profiles
    orin_profiles,orin_profile_inputs=load_component_profiles(BASE);sources.extend(orin_profile_inputs)
    orin15_profiles,orin15_profile_inputs=load_component_profiles(BASE,'15W');sources.extend(orin15_profile_inputs)
    co_for_table=dict(component_result,groups=component_result['groups']+orin_profiles.get('CO',{}).get('groups',[])+orin15_profiles.get('CO',{}).get('groups',[]))
    ec_for_table=dict(ec_component_result,groups=ec_component_result['groups']+orin_profiles.get('EC',{}).get('groups',[])+orin15_profiles.get('EC',{}).get('groups',[]))
    (PAPER/'generated_component_profile_table.tex').write_text(component_table_tex(co_for_table,ec_for_table))
    aggregates['ORIN_MAXN_COMPONENT_PROFILES']={k:r['groups'] for k,r in orin_profiles.items()}
    aggregates['ORIN_15W_COMPONENT_PROFILES']={k:r['groups'] for k,r in orin15_profiles.items()}
    (PAPER/'DATASET_AGGREGATES.json').write_text(json.dumps(aggregates,indent=2)+'\n')
    subprocess.run([sys.executable,str(BASE/'experiments/prepare_round2_placeholder.py')],check=True,stdout=subprocess.DEVNULL)
    env=dict(os.environ,TEXMFVAR=str(BASE/'build/tex_cache/var'),TEXMFCONFIG=str(BASE/'build/tex_cache/config'))
    with (PAPER/'build_live_update.log').open('w') as log:
        subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir=build_live','paper_round2_placeholder.tex'],cwd=PAPER,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    built=PAPER/'build_live/paper_round2_placeholder.pdf'
    temp=PAPER/'paper_round2_placeholder.pdf.tmp';shutil.copy2(built,temp);temp.replace(PAPER/'paper_round2_placeholder.pdf')
    info=subprocess.check_output(['pdfinfo',str(built)],text=True)
    pages=int(next(line.split(':',1)[1].strip() for line in info.splitlines() if line.startswith('Pages:')))
    progress=dict(updated_paris=now,dp_complete_scene_comparisons=len(dp),full_suite_scene_count=len(SCENES),co_morton_complete_scene_capacity_comparisons=len(mc),co_morton_expected_comparisons=len(SCENES)+len(REPLICA),ec_complete_scene_capacity_comparisons=len(ec),ec_expected_comparisons=16,seeds=[0,1,2],ate_conventions=['direct','relative-pose-corrected'],orins_results_imported=False,reconstruction_scores=f"{recon['scored']}/{recon['total']} checked meshes; {len(recon['failures'])} failures",engineering='three-seed Co ablation complete' if eng else 'pending full paired runs',composed_variant='pending',pdf_pages=pages,pdf_sha256=digest(PAPER/'paper_round2_placeholder.pdf'),input_hashes={str(p.relative_to(BASE)):digest(p) for p in sources})
    progress['ec_complete_call_profiles']=f"{timed['completed']}/{timed['total']} full profiles"
    progress['ec_component_profiles']=dict(completed=ec_component_result['completed'],total=ec_component_result['total'],status=ec_component_result.get('status',{}),orin_maxn_measurements_complete=bool(orin_profiles.get('EC')),orin_15w_measurements_complete=bool(orin15_profiles.get('EC')))
    progress['composed_variant']=f"{composition['completed_runs']}/72 full runs; {composition['complete_scenes']}/24 complete three-seed scenes"
    if profile_protocol.exists():
        planned=json.loads(profile_protocol.read_text())
        progress['component_profiling']=dict(status=planned['status'],workloads=[w['id'] for w in planned['workloads']],full_profile_passes=15,archived_initial_full_passes=planned['total_full_profile_passes'],orin_profile_seeds=[0],driver_status=planned['profiling_driver_status'],measurements_complete=False)
        if component_result['runtime_ready']:
            progress['component_profiling'].update(status=component_result['status'].get('status','PENDING'),driver_status='RTX and imported Orin fixed-state/online preflights passed',rtx_completed=component_result['completed'],rtx_total=component_result['total'],rtx_complete=component_result['completed']==component_result['total'],orin_maxn_complete=bool(orin_profiles.get('CO')),orin_15w_complete=bool(orin15_profiles.get('CO')))
            progress['component_profiling']['measurements_complete']=component_result['completed']==component_result['total'] and bool(orin_profiles.get('CO')) and bool(orin15_profiles.get('CO'))
    progress['orin_maxn_component_profiles']={k:dict(completed=r['completed'],total=r['total'],seeds=[0]) for k,r in orin_profiles.items()}
    progress['orin_15w_component_profiles']={k:dict(completed=r['completed'],total=r['total'],seeds=[0]) for k,r in orin15_profiles.items()}
    progress['orin_15w']=({k:orin15[k] for k in ('verified_runs','native_runs','engineering_runs','profile_runs','seeds','author_excluded','missing_runs','failed_runs','blocked_runs','dp_runs_in_this_snapshot','composition_runs_in_this_snapshot','clocks')} if orin15 else None)
    progress['orins_results_imported']=orin is not None
    progress['orin_maxn']=({k:orin[k] for k in ('verified_runs','native_runs','engineering_runs','reused_runs','seeds','pending_scannet_runs','dp_runs_in_this_snapshot','composition_runs_in_this_snapshot','ec_scope','clocks')} if orin else None)
    progress['dp_tables_complete']=len(dp)==len(SCENES)
    progress['dp_acceptance']='author decision after full dataset means; previous thresholds advisory'
    dp_reconstruction={r['dataset']:r for r in recon['groups'] if r['system']=='Co-SLAM' and r['variant']=='DP' and r['hash_size']==16 and not r['sensitivity']}
    progress['dp_reconstruction_coverage']={d:dict(complete=dp_reconstruction.get(d,{}).get('scene_count',0),expected=n) for d,n in [('Replica',8),('Synthetic',7)]}
    progress['dp_reconstruction_complete']=all(r['complete']==r['expected'] for r in progress['dp_reconstruction_coverage'].values())
    (PAPER/'RESULTS_PROGRESS.json').write_text(json.dumps(progress,indent=2)+'\n')
    if progress['dp_tables_complete']:
        (BASE/'results/TABLES_3_4_COMPLETE.json').write_text(json.dumps(dict(updated_paris=now,complete_scene_comparisons=len(dp),seeds=[0,1,2],pdf=str(PAPER/'paper_round2_placeholder.pdf'),pdf_sha256=progress['pdf_sha256'],acceptance='pending author review'),indent=2)+'\n')
    if progress['dp_reconstruction_complete']:
        (BASE/'results/DP_RECONSTRUCTION_COMPLETE.json').write_text(json.dumps(dict(updated_paris=now,coverage=progress['dp_reconstruction_coverage'],seeds=[0,1,2],combined_scored=recon['scored'],combined_declared=recon['total'],failures=len(recon['failures']),pdf=str(PAPER/'paper_round2_placeholder.pdf'),pdf_sha256=progress['pdf_sha256'],acceptance='pending author review'),indent=2)+'\n')
    (PAPER/'RESULTS_PROGRESS.md').write_text(f"# Paper results progress\n\nUpdated {now}.\n\n- Full configured coverage: Replica 8, Synthetic 7, TUM 3, ScanNet 6; timing and ATE for every Co scene, checked reconstruction additionally for Replica/Synthetic.\n- Conservative DP Tables 3 and 4: {len(dp)}/{len(SCENES)} complete three-seed scene comparisons. Previous thresholds are advisory; author acceptance follows review of full equal-scene dataset means.\n- Co Morton: {len(mc)}/{len(SCENES)+len(REPLICA)} three-seed scene/capacity comparisons; all completed outcomes retained.\n- EC Morton quality: {len(ec)}/16 three-seed comparisons.\n- Reconstruction: {progress['reconstruction_scores']}.\n- Co engineering: {progress['engineering']}.\n- EC complete-call timing: {progress['ec_complete_call_profiles']}; instrumented ATE is separate sensitivity evidence.\n- Orin imports, cross-machine timing scope reconciliation and composition: pending.\n\nThe working PDF has {pages} pages; the working draft has no page limit, and final submission sizing will be handled later. See RESULTS_PROGRESS.json for exact input hashes.\n")
    if orin:
        path=PAPER/'RESULTS_PROGRESS.md'
        note=f"- Orin MAXN imported: {orin['verified_runs']} audited full benchmark runs; {orin['dp_runs_in_this_snapshot']} DP and {orin['composition_runs_in_this_snapshot']} MC+DP runs, two paired seeds.\n"
        note+=f"- ScanNet MAXN pending: {orin['pending_scannet_runs']}; all native capacities are pooled with equal scene weights in dataset rows.\n"
        note+="- Tables 1, 2, 4, 6, 9, 10 and composition timing include available MAXN evidence. EC RTX active-worker and MAXN optimizer-method scopes remain explicitly distinct.\n"
        note+=f"- MAXN component profiles: {sum(r['completed'] for r in orin_profiles.values())}/5 audited, seed 0 only without across-seed SD. 15 W and manual kernel profiles remain pending."
        path.write_text(path.read_text().replace('- Orin imports, cross-machine timing scope reconciliation and composition: pending.',note))
    with (PAPER/'RESULTS_PROGRESS.md').open('a') as progress_file:
        progress_file.write(f"\nRTX MC+DP composition: {progress['composed_variant']}. The detached completion campaign includes 45 new Replica/Synthetic mesh scores; existing matched controls are reused.\n")
        if profile_protocol.exists():progress_file.write(f"\nComponent profiling: {component_result['completed']}/{component_result['total']} full RTX passes audited. Co-SLAM office0 H16/H19 and fr1_desk H16, tracking/BA, forward plus optional iteration/call decomposition. Table 1 includes only complete seed groups. Nine RTX and three per Orin mode profiles are separate end-of-benchmark blocks; MAXN profiles are imported when listed above; 15 W remains pending. Orin uses seed 0 only, without across-seed SD. See ../docs/protocols/COMPONENT_PROFILING_PROTOCOL.md and ../results/component_profile_analysis_v1/REPORT.md.\n")
        progress_file.write(f"\nEC component profiling: {ec_component_result['completed']}/{ec_component_result['total']} full RTX passes audited. Replica office0 H16/H19 only, native concurrent workers and eight tracking/ten mapping updates. Same temporal selection; six RTX/two per Orin mode profiles. Table 1 includes complete seed groups; optional iteration/call results retain explicit handshake waits. See ../docs/protocols/EC_COMPONENT_PROFILING_PROTOCOL.md.\n")
    if orin15:
        path=PAPER/'RESULTS_PROGRESS.md'
        content=path.read_text().replace('15 W and manual kernel profiles remain pending.','15 W results are imported; manual kernel profiles remain separate.').replace('15 W remains pending.','15 W profiles are also imported.')
        content+='\n15 W final import: **130/130 benchmarks +5/5 profiles**, seed 0 only, no across-seed SD. Stages 48/56/26/5; 162 author exclusions, no missing/failed/blocked slots. MC-only covers Replica H16/H19; BASE/DP and MC+DP cover all four datasets. Non-Replica MC-only comparisons are unavailable by design. Tables include complete-call host and iteration CUDA scopes separately. See ../results/orin_15w_analysis_v1/REPORT.md.\n'
        path.write_text(content)
    build_manifest=BASE/'metadata/DOCUMENT_BUILD_MANIFEST.json'
    document=json.loads(build_manifest.read_text()) if build_manifest.exists() else {}
    document['paper_round2_placeholder/paper_round2_placeholder.pdf']=dict(pages=pages,sha256=progress['pdf_sha256'],bytes=(PAPER/'paper_round2_placeholder.pdf').stat().st_size)
    build_manifest.write_text(json.dumps(document,indent=2)+'\n')
    print('PAPER_UPDATED',now,len(dp),len(mc),len(ec),pages,flush=True)


def completion_signature():
    # Only paired groups with all three seeds trigger a rebuild, not each job.
    queue=BASE/'results/replica_h19_extension_v1'
    groups={}
    for job in json.loads((BASE/'experiments/manifests/rtx_replica_h19_extension.json').read_text())['jobs']:
        if (queue/(job['id']+'.done.json')).exists():groups.setdefault(job['scene'],set()).add((job['condition'],job['seed']))
    count=sum(len(v)==6 for v in groups.values())
    other_queue=BASE/'results/all_nonreplica_extension_v1';other_groups={}
    for job in json.loads((BASE/'experiments/manifests/rtx_all_nonreplica_extension.json').read_text())['jobs']:
        if (other_queue/(job['id']+'.done.json')).exists():other_groups.setdefault((job['dataset'],job['scene']),set()).add((job['condition'],job['seed']))
    other_count=sum(len(v)==6 for v in other_groups.values())
    ec_queue=BASE/'results/ec_quality_v1';ec_groups={}
    manifest=ec_queue/'job_manifest.json'
    if manifest.exists():
        for job in json.loads(manifest.read_text())['jobs']:
            if (ec_queue/(job['id']+'.done.json')).exists():ec_groups.setdefault((job['scene'],job['hash_size']),set()).add((job['condition'],job['seed']))
    recon_queue=BASE/'results/reconstruction_v1';recon_groups={}
    recon_manifest=BASE/'experiments/manifests/rtx_reconstruction_all.json'
    recon_failures=0
    if recon_manifest.exists():
        for job in json.loads(recon_manifest.read_text())['jobs']:
            if (recon_queue/(job['id']+'.failed.json')).exists():recon_failures+=1
            if (recon_queue/(job['id']+'.done.json')).exists():
                key=(job['system'],job['dataset'],job['scene'],job['hash_size'])
                recon_groups.setdefault(key,{}).setdefault(job['condition'],set()).add(job['seed'])
    recon_count=sum(set(group.get('BASE',set()))=={0,1,2} and set(group.get(v,set()))=={0,1,2} for group in recon_groups.values() for v in ('MC','DP'))
    eng_queue=BASE/'results/engineering_v1';eng_count=len(list(eng_queue.glob('*.done.json'))) if eng_queue.exists() else 0
    timed_groups={};timed_queue=BASE/'results/ec_timing_v1'
    timed_manifest=BASE/'experiments/manifests/rtx_ec_complete_call_timing.json'
    if timed_manifest.exists():
        for job in json.loads(timed_manifest.read_text())['jobs']:
            if (timed_queue/(job['id']+'.done.json')).exists():timed_groups.setdefault((job['scene'],job['hash_size']),set()).add((job['condition'],job['seed']))
    timed_count=sum(len(v)==6 for v in timed_groups.values())
    priority=BASE/'results/dp_completion_priority_v1'
    priority_done=len(list(priority.glob('*.done.json'))) if priority.exists() else 0
    priority_failed=len(list(priority.glob('*.failed.json'))) if priority.exists() else 0
    extension=BASE/'results/reconstruction_dp_extension_v1'
    extension_done=len(list(extension.glob('*.done.json'))) if extension.exists() else 0
    extension_failed=len(list(extension.glob('*.failed.json'))) if extension.exists() else 0
    orin_active=BASE/'results/ORIN_MAXN_ACTIVE.json'
    orin_status=(Path(json.loads(orin_active.read_text())['analysis'])/'ANALYSIS_STATUS.json') if orin_active.exists() else BASE/'results/orin_maxn_analysis/ANALYSIS_STATUS.json'
    orin15_active=BASE/'results/ORIN_15W_ACTIVE.json'
    orin15_status=(Path(json.loads(orin15_active.read_text())['analysis'])/'ANALYSIS_STATUS.json') if orin15_active.exists() else None
    composition_queue=BASE/'results/composition_v1'
    composition_done=len(list(composition_queue.glob('*.done.json')))
    composition_recon=BASE/'results/reconstruction_composition_v1'
    composition_scored=len(list(composition_recon.glob('*.done.json')))
    composition_failed=len(list(composition_recon.glob('*.failed.json')))
    active_component=BASE/'results/COMPONENT_PROFILE_ACTIVE.json'
    component_done=0;component_failure=False
    if active_component.exists():
        component_queue=Path(json.loads(active_component.read_text())['queue'])
        component_done=len([p for p in component_queue.glob('*.done.json') if '_diagnostic.done' not in p.name])
        component_failure=(component_queue/'series_failure.json').exists()
    ec_component_done=0;ec_component_failure=False
    ec_component_active=BASE/'results/EC_COMPONENT_PROFILE_ACTIVE.json'
    if ec_component_active.exists():
        ec_component_queue=Path(json.loads(ec_component_active.read_text())['queue'])
        ec_component_done=len([p for p in ec_component_queue.glob('*.done.json') if '_diagnostic.done' not in p.name])
        ec_component_failure=(ec_component_queue/'series_failure.json').exists()
    return count,other_count,sum(len(v)==6 for v in ec_groups.values()),recon_count,recon_failures,eng_count//6,timed_count,priority_done//3,priority_failed,extension_done//3,extension_failed,(digest(orin_status) if orin_status.exists() else None),(digest(orin15_status) if orin15_status and orin15_status.exists() else None),composition_done//3,composition_scored//3,composition_failed,component_done//3,component_failure,ec_component_done//3,ec_component_failure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--audit-live',action='store_true')
    args=parser.parse_args()
    try:os.nice(19)
    except OSError:pass
    update(audit_live=args.audit_live)
    if args.watch:
        signature=completion_signature()
        while True:
            time.sleep(60)
            current=completion_signature()
            if current!=signature:
                try:
                    update(audit_live=True)
                    signature=current
                except Exception as error:
                    (PAPER/'UPDATE_FAILURE.json').write_text(json.dumps(dict(error=str(error),at=datetime.now(ZoneInfo('Europe/Paris')).isoformat()),indent=2)+'\n')
                    print('PAPER_UPDATE_FAILED',repr(error),flush=True)


if __name__=='__main__':main()
