"""Render audited Orin results with their actual two-seed timing scopes."""
import hashlib
import json
from pathlib import Path
from round2_paths import local_path


def load_orin(base, mode='MAXN'):
    active=base/f'results/ORIN_{mode}_ACTIVE.json'
    if mode=='15W' and not active.exists():return None,[]
    root=Path(json.loads(active.read_text())['analysis']) if active.exists() else base/'results/orin_maxn_analysis'
    path = root / 'ANALYSIS_STATUS.json'
    if not path.exists(): return None, []
    data = json.loads(path.read_text())
    assert data['mode'] == mode
    if mode=='15W':assert data['verified_runs']==130 and data['seeds']==[0] and data['profile_runs']==5
    else:assert data['verified_runs'] in (172,292) and data['seeds']==[0,1]
    for filename, expected in data['input_sha256'].items():
        assert hashlib.sha256(local_path(base,filename).read_bytes()).hexdigest() == expected, filename
    return data, [*([active] if active.exists() else []),path, root/'per_run.csv', root/'dataset_means.csv', root/'REPORT.md',
                  *[local_path(base,p) for p in data['input_sha256']]]


def find(orin, system, dataset, capacity, category='MC'):
    if dataset=='ScanNet' and any(r['category']==category and r['system']==system and r['dataset']==dataset and r['hash_size'] is None for r in (orin or {}).get('groups',[])):
        capacity=None  # Dataset-wide row includes all native H12/H14/H16 scenes.
    return next((r for r in (orin or {}).get('groups', []) if
                 (r['system'],r['dataset'],r['hash_size'],r['category']) ==
                 (system,dataset,capacity,category)), None)


def dp_phase_rows(orin):
    result=[]
    for dataset in ('Replica','Synthetic','Tum','ScanNet'):
        group=find(orin,'Co-SLAM',dataset,16,'DP')
        if group is None:continue
        for stage,prefix in [('tracking','tracking'),('bundle_adjustment','ba')]:
            row=dict(dataset=dataset,stage=stage,platform=orin['mode'])
            for condition in ('BASE','DP'):
                row[condition+'_iterations_per_call']=group[condition+'_'+prefix+'_updates_per_call']
                for field,source in [('summed_iteration_cuda_ms_per_call',prefix+'_iteration_cuda_ms'),('call_host_wall_ms',prefix+'_wall_ms')]:
                    row[condition+'_'+field]=group[condition+'_'+source]
                    row[condition+'_'+field+'_sd']=group[condition+'_'+source+'_sd']
                    row[field+'_reduction_pct']=group[source+'_reduction_pct']
            result.append(row)
    return result


def load_component_profiles(base, mode='MAXN'):
    """Only locally audited profile summaries; preserve RTX active pointers."""
    results={};inputs=[]
    for key in ('CO','EC'):
        active=base/'results'/f'ORIN_{mode}_{key}_COMPONENT_ACTIVE.json'
        if not active.exists():continue
        spec=json.loads(active.read_text())
        path=Path(spec['analysis_output'])/'SUMMARY.json'
        data=json.loads(path.read_text())
        assert data['completed']==data['total']==(3 if key=='CO' else 2)
        for filename,expected in data['input_sha256'].items():
            assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==expected,filename
        results[key]=data
        inputs.extend([active,path,*map(Path,data['input_sha256'])])
    return results,inputs


def composition_table(orin,val,gain,table):
    if not orin or orin.get('composition_runs_in_this_snapshot',0)!=48:return ''
    body=[]
    for dataset,total in [('Replica',8),('Synthetic',7),('Tum',3),('ScanNet',6)]:
        rows={c:find(orin,'Co-SLAM',dataset,16,c) for c in ('MC','DP','MC_DP')}
        assert all(r and r['scene_count']==total for r in rows.values())
        for condition in ('BASE','MC','DP','MC_DP'):
            row=rows['MC_DP' if condition=='BASE' else condition]
            values=[('TUM' if dataset=='Tum' else dataset)+f' ({total}/{total})' if condition=='BASE' else '',condition.replace('_','+')]
            for field in ('tracking_wall_ms','ba_wall_ms','ba_iteration_cuda_ms'):
                values.extend([val(row,condition,field),gain(0 if condition=='BASE' else row[field+'_reduction_pct'])])
            body.append(values)
    return table(r'MAXN composition timings, ms per invocation. Full native scenes and capacities, two paired seeds, equal scene weights; mean $\pm$ seed SD. TR/BA host wall includes preparation, every update and finalization. BA iteration total sums optimizer CUDA-event intervals, not loop wall time. Frames 0--19 excluded; BA every fifth frame, ten updates except TUM twenty. All conditions independently measured with matched sources/configurations; bold gains use BASE at the same scope. Negative gains are slower. No Orin quality measurements.',
                 'tab:composition_timing_maxn',['Dataset ($n/N$)','Condition','TR host',r'Gain (\%)','BA host',r'Gain (\%)','BA iter. CUDA',r'Gain (\%)'],body)


def co_morton_table(rtx_groups, orin, val, gain, table, orin15=None):
    body = []
    expected = {'Replica':8,'Synthetic':7,'Tum':3,'ScanNet':6}
    for dataset, capacity, rtx in rtx_groups:
        native_capacity = 16 if capacity is None else capacity
        platforms=[('RTX',rtx),('MAXN',find(orin,'Co-SLAM',dataset,native_capacity))]
        if orin15 and dataset=='Replica':platforms.append(('15 W',find(orin15,'Co-SLAM',dataset,native_capacity)))
        for platform, row in platforms:
            label = ('TUM' if dataset == 'Tum' else dataset) + (f' H{capacity}' if capacity else '')
            label += f" ({row['scene_count'] if row else 0}/{expected[dataset]})"
            fields = ('tracking_wall_ms','ba_wall_ms')
            values = []
            for field in fields:
                values += [val(row,'BASE',field),val(row,'MC',field),gain(row[field+'_reduction_pct']) if row else r'\pending']
            body.append([label,platform,*values])
    return table(r'Co BASE/MC, synchronized host wall ms per complete tracking/BA call: all updates, preparation, pose finalization and ordering. RTX: three paired seeds; MAXN: two; 15 W: seed 0 only, no seed SD. 15 W MC-only covers Replica H16/H19; other datasets are excluded from its scope. Equal scene weights; frames 0--19 excluded. TR once/frame; BA every fifth. Bold gains are time reductions; red denotes slower execution.',
                 'tab:morton_wall',['Dataset ($n/N$)','GPU','BASE TR','MC TR',r'\textbf{Gain (\%)}','BASE BA','MC BA',r'\textbf{Gain (\%)}'],body)


def engineering_table(rtx, orin, val, gain, table, orin15=None):
    body = []
    embedded = find(orin,'Co-SLAM','Replica',16,'ENG')
    platforms=[('RTX',rtx),('MAXN',embedded)]
    if orin15:platforms.append(('15 W',find(orin15,'Co-SLAM','Replica',16,'ENG')))
    for platform, row in platforms:
        for condition in ('BASE','ENG'):
            quality = [val(row,condition,f) if platform == 'RTX' else '---' for f in ('ate_cm','corrected_ate_cm')]
            body.append([platform,condition,val(row,condition,'ba_wall_ms'),*quality])
        changes = []
        for field in ('ba_wall_ms',):
            change = (-row[field+'_change_pct'] if platform == 'RTX' else row[field+'_reduction_pct']) if row else None
            changes.append(gain(change) if change is not None else r'\pending')
        body.append([platform,r'\textbf{Gain (\%)}',*changes,'---','---'])
    text = table(r'Co-SLAM engineering: BA comparison, office0 H16. Host wall ms per complete BA invocation, including all ten updates, preparation and pose finalization; BA every fifth frame. Three RTX/two MAXN paired seeds; 15 W seed 0 only, no seed SD. Frames 0--19 excluded. Dedicated matched BASE controls; no pruning or Morton. ATE (cm) was collected on RTX only (--- on Orin). Bold gains are BA time reductions.',
                 'tab:engineering',['GPU','Condition','BA ms/call','Direct ATE','Corr. ATE'],body)
    return text + 'Engineering comparisons concern BA; their gains remain separate from pruning and Morton.\n'


def ec_table(rtx, orin, val, gain, table, orin15=None):
    body = []
    for capacity in (16,19):
        desktop = next((r for r in (rtx or {}).get('groups',[]) if r['hash_size'] == capacity),None)
        embedded = find(orin,'EC-SLAM','Replica',capacity)
        platforms=[('RTX',desktop),('MAXN',embedded)]
        if orin15:platforms.append(('15 W',find(orin15,'EC-SLAM','Replica',capacity)))
        for platform, row in platforms:
            values = []
            for field in ('tracking_wall_ms', 'mapping_wall_ms' if platform == 'RTX' else 'ba_wall_ms'):
                reduction = (-row[field+'_change_pct'] if platform == 'RTX' else row[field+'_reduction_pct']) if row else None
                values += [val(row,'BASE',field),val(row,'MC',field),gain(reduction) if row else r'\pending']
            body.append([platform,f"H{capacity} ({row['scene_count'] if row else 0}/8)",*values])
    return table(r'EC-SLAM host wall ms at recorded call boundaries. RTX: active tracking/mapping worker bodies, handshake waits excluded. Both Orin modes: tracking/\texttt{optimize\_map} optimizer-method calls, excluding outer frame preparation/finalization. Compare BASE/MC within each GPU; differing RTX/Orin scopes preclude an absolute cross-GPU speed ratio. Native workers/streams overlap; no additive FPS. RTX three/MAXN two paired seeds; 15 W seed 0 only, no seed SD. Equal scene weights; frames 0--19 excluded.',
                 'tab:ec_calls',['GPU','Capacity ($n/N$)','BASE TR','MC TR',r'\textbf{Gain (\%)}','BASE map','MC map',r'\textbf{Gain (\%)}'],body)


def composition_15w_table(orin,val,gain,table):
    if not orin:return ''
    body=[]
    for dataset,total in [('Replica',8),('Synthetic',7),('Tum',3),('ScanNet',6)]:
        conditions=('BASE','MC','DP','MC_DP') if dataset=='Replica' else ('BASE','DP','MC_DP')
        for condition in conditions:
            row=find(orin,'Co-SLAM',dataset,16,'MC_DP' if condition=='BASE' else condition)
            assert row and row['scene_count']==total
            values=[('TUM' if dataset=='Tum' else dataset)+f' ({total}/{total})' if condition=='BASE' else '',condition.replace('_','+')]
            for field in ('tracking_wall_ms','ba_wall_ms','ba_iteration_cuda_ms'):
                values.extend([val(row,condition,field),gain(0 if condition=='BASE' else row[field+'_reduction_pct'])])
            body.append(values)
    return table(r'15 W composition timings, ms per invocation. Seed 0 only, no across-seed SD; equal scene weights and native capacities. TR/BA host wall includes preparation, every update and finalization; BA iteration total sums optimizer CUDA-event intervals, not loop wall time. Warmup frames 0--19 excluded; BA every fifth frame, ten updates except TUM twenty. Bold gains use matched BASE. MC-only was intentionally restricted to Replica; elsewhere MC+DP has BASE/DP comparisons but no MC-only comparison. No Orin quality measurements.',
                 'tab:composition_timing_15w',['Dataset ($n/N$)','Condition','TR host',r'Gain (\%)','BA host',r'Gain (\%)','BA iter. CUDA',r'Gain (\%)'],body)
