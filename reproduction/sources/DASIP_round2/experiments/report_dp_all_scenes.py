#!/usr/bin/env python3
"""Audit all native BASE/DP runs, including the author-priority extension."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo
from report_support_reference import audit_job,read_json,write_csv


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    base=Path(__file__).resolve().parents[1]
    roots=['support_reference_v1','replica_native_extension_v1','all_nonreplica_extension_v1','dp_completion_priority_v1']
    expected_hash=hashlib.sha256((base/'experiments/coslam_reference_runtime_v1/RUNTIME_MANIFEST.json').read_bytes()).hexdigest()
    rows=[];provenance=[];failures=[]
    for name in roots:
        queue=base/'results'/name
        if not (queue/'job_manifest.json').exists():continue
        identity=read_json(queue/'queue_identity.json');assert identity['runtime_manifest_sha256']==expected_hash
        provenance.append(dict(queue=str(queue),identity=identity))
        for job in read_json(queue/'job_manifest.json')['jobs']:
            if job['condition'] not in ('BASE','DP'):continue
            if (queue/(job['id']+'.failed.json')).exists():failures.append(read_json(queue/(job['id']+'.failed.json')))
            if not (queue/(job['id']+'.done.json')).exists():continue
            row=audit_job(queue,job,identity)
            row['queue']=str(queue);rows.append(row)
    grouped={}
    for row in rows:
        group=grouped.setdefault((row['dataset'],row['scene']),{}).setdefault(row['condition'],{})
        assert row['seed'] not in group,'Duplicate native scene/condition/seed'
        group[row['seed']]=row
    comparisons=[]
    fields=('ate_cm','corrected_ate_cm','tracking_wall_ms','ba_wall_ms','tracking_cuda_ms','ba_cuda_ms','tracking_pruned_pct','ba_pruned_pct')
    for (dataset,scene),groups in sorted(grouped.items()):
        if any(set(groups.get(c,{}))!={0,1,2} for c in ('BASE','DP')):continue
        for seed in (0,1,2):
            assert groups['BASE'][seed]['effective_config_sha256']==groups['DP'][seed]['effective_config_sha256']
            assert groups['BASE'][seed]['frames']==groups['DP'][seed]['frames']
        r=dict(dataset=dataset,scene=scene,paired_seeds=3)
        for field in fields:
            for condition in ('BASE','DP'):
                values=[groups[condition][seed][field] for seed in (0,1,2)]
                r[condition+'_'+field]=statistics.mean(values);r[condition+'_'+field+'_sd']=statistics.stdev(values)
            if field in ('ate_cm','corrected_ate_cm'):
                r[field+'_increase_pct']=100*(r['DP_'+field]/r['BASE_'+field]-1)
            elif field.endswith('_ms'):
                r[field+'_reduction_pct']=100*(1-r['DP_'+field]/r['BASE_'+field])
        r.update(tracking_tolerated_target_met=r['tracking_wall_ms_reduction_pct']>=5,ba_tolerated_target_met=r['ba_wall_ms_reduction_pct']>=25,direct_ate_target_met=r['ate_cm_increase_pct']<=15,corrected_ate_target_met=r['corrected_ate_cm_increase_pct']<=15)
        comparisons.append(r)
    expected={'Replica':8,'Synthetic':7,'Tum':3,'ScanNet':6}
    coverage={d:dict(complete=sum(r['dataset']==d for r in comparisons),expected=n) for d,n in expected.items()}
    complete=all(v['complete']==v['expected'] for v in coverage.values())
    status=dict(updated_paris=datetime.now(ZoneInfo('Europe/Paris')).isoformat(),audited_runs=len(rows),scenes_with_three_paired_seeds=len(comparisons),full_configured_tables_complete=complete,coverage=coverage,source_runtime_sha256=expected_hash,provenance=provenance,failures=failures,acceptance='Author review of full equal-scene dataset means; previous per-scene quality/speed thresholds are advisory diagnostics, not automatic exclusion',seed_policy='0/1/2 paired controls; no rerun or outcome selection',timing='complete calls after frames 0-19',reconstruction_scoring='separate; Tables 3 and 4 contain trajectory ATE and call timing')
    args.output.mkdir(exist_ok=True,parents=True);write_csv(args.output/'per_run.csv',rows);write_csv(args.output/'per_scene.csv',comparisons)
    (args.output/'ANALYSIS_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    report='# Conservative DP: full configured-scene evidence\n\n'+json.dumps(status,indent=2)+'\n\n'
    report+='The working PDF Tables 3 and 4 give equally weighted scene means and SD across three seed-wise dataset means. Incomplete scene groups are explicit and never silently treated as full coverage. Every completed outcome is retained. Existing matched BASE controls are reused after config, source, trajectory ATE and call-coverage audits. Prior threshold flags are reference diagnostics; the author decides acceptance after full coverage.\n\n'
    report+='| Dataset / scene | Direct ATE change | Corrected ATE change | Tracking time reduction | BA time reduction |\n|---|---:|---:|---:|---:|\n'
    for r in comparisons:report+='| '+r['dataset']+'/'+r['scene']+' | '+' | '.join(f"{r[k]:+.2f}%" for k in ('ate_cm_increase_pct','corrected_ate_cm_increase_pct','tracking_wall_ms_reduction_pct','ba_wall_ms_reduction_pct'))+' |\n'
    (args.output/'REPORT.md').write_text(report);print('DP_ALL_SCENES_AUDITED',len(rows),len(comparisons),'of 24',flush=True)


if __name__=='__main__':main()
