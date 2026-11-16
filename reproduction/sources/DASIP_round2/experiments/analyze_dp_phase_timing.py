#!/usr/bin/env python3
"""Explain DP tracking/BA timing from saved runs; no new GPU experiments."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import defaultdict


def read_rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def main():
    base=Path(__file__).resolve().parents[1];out=base/'results/dp_phase_timing_analysis';out.mkdir(exist_ok=True)
    expected={'Replica':8,'Synthetic':7,'Tum':3,'ScanNet':6}
    source=base/'results/dp_all_scenes_analysis/per_run.csv';inputs={str(source):hashlib.sha256(source.read_bytes()).hexdigest()};rows=[]
    for run in read_rows(source):
        if run['dataset'] not in expected:continue
        q=Path(run['queue'])/run['id'];cfg=json.loads((q/'config.json').read_text())
        calls=[r for r in read_rows(q/'calls.csv') if int(r['frame_id'])>=20]
        iters=defaultdict(list)
        for r in read_rows(q/'iterations.csv'):
            if int(r['frame_id'])>=20:iters[(r['iteration_type'],int(r['frame_id']))].append(r)
        queries=defaultdict(lambda:[0,0])
        for r in read_rows(q/'queries.csv'):
            if int(r['frame_id'])>=20:
                queries[r['stage']][0]+=int(r['dense_queries']);queries[r['stage']][1]+=int(r['actual_queries'])
        for name in ['calls.csv','iterations.csv','queries.csv','config.json','source_manifest.json']:
            p=q/name;inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        for stage in ('tracking','bundle_adjustment'):
            call_rows=[r for r in calls if r['stage']==stage];n=cfg['tracking']['iter'] if stage=='tracking' else cfg['mapping']['iters']
            all_iterations=[];iteration_sums=[];outside=[]
            for c in call_rows:
                values=iters[(stage,int(c['frame_id']))];assert len(values)==n
                all_iterations.extend(values)
                total=sum(float(v['full_iteration_cuda_ms']) for v in values)
                gap=float(c['cuda_ms'])-total;assert gap>=-.01
                iteration_sums.append(total);outside.append(gap)
            dense,actual=queries[stage];assert 0<actual<=dense
            row=dict(dataset=run['dataset'],scene=run['scene'],seed=int(run['seed']),condition=run['condition'],stage=stage,calls=len(call_rows),iterations_per_call=n,
                call_host_wall_ms=statistics.mean(float(c['wall_ms']) for c in call_rows),call_cuda_interval_ms=statistics.mean(float(c['cuda_ms']) for c in call_rows),
                summed_iteration_cuda_ms_per_call=statistics.mean(iteration_sums),outside_iterations_cuda_ms_per_call=statistics.mean(outside),
                forward_cuda_ms_per_iteration=statistics.mean(float(v['forward_cuda_ms']) for v in all_iterations),
                backward_cuda_ms_per_iteration=statistics.mean(float(v['backward_cuda_ms']) for v in all_iterations),
                full_iteration_cuda_ms=statistics.mean(float(v['full_iteration_cuda_ms']) for v in all_iterations),
                render_queries_pruned_pct=100*(1-actual/dense),
                unpruned_smoothness_feature_queries_per_iteration=(cfg['training']['smooth_pts']-1)**3 if stage=='bundle_adjustment' and cfg['training']['smooth_weight']>0 else 0)
            assert all(math.isfinite(v) for v in row.values() if isinstance(v,(float,int)))
            rows.append(row)
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],r['stage'],r['condition'])].append(r)
    for dataset in expected:
        for stage in ('tracking','bundle_adjustment'):
            identities=[]
            for condition in ('BASE','DP'):
                ids=[(r['scene'],r['seed']) for r in groups[(dataset,stage,condition)]]
                assert len(ids)==len(set(ids))
                by_scene=defaultdict(set)
                for scene,seed in ids:by_scene[scene].add(seed)
                assert len(by_scene)==expected[dataset]
                assert all(seeds=={0,1,2} for seeds in by_scene.values())
                identities.append(set(ids))
            assert identities[0]==identities[1]
    fields=[field for field in rows[0] if field not in ('dataset','scene','seed','condition','stage','calls')]
    means={};deviations={}
    for key,values in groups.items():
        means[key]={};deviations[key]={}
        for field in fields:
            seed_means=[statistics.mean(r[field] for r in values if r['seed']==seed) for seed in (0,1,2)]
            means[key][field]=statistics.mean(seed_means);deviations[key][field]=statistics.stdev(seed_means)
    comparisons=[]
    for dataset in expected:
        for stage in ('tracking','bundle_adjustment'):
            a=means[(dataset,stage,'BASE')];b=means[(dataset,stage,'DP')]
            row=dict(dataset=dataset,stage=stage)
            for key in a:
                row['BASE_'+key]=a[key];row['DP_'+key]=b[key]
                row['BASE_'+key+'_sd']=deviations[(dataset,stage,'BASE')][key]
                row['DP_'+key+'_sd']=deviations[(dataset,stage,'DP')][key]
                if '_ms' in key:
                    row[key+'_reduction_pct']=100*(1-b[key]/a[key]) if a[key] else None
            row['saved_host_wall_ms_per_call']=a['call_host_wall_ms']-b['call_host_wall_ms'];comparisons.append(row)
    for filename,records in [('per_run.csv',rows),('dataset_phase_means.csv',comparisons)]:
        with (out/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report='# Why DP percentage reductions differ between tracking and BA\n\n'
    report+='CPU-only analysis of the existing paired runs; no new SLAM runs, source changes or component timings were invented. Frames 0–19 are excluded. Means give all scenes equal weight through their three complete seeds.\n\n'
    report+='| Dataset | Stage | BASE ms/call | DP ms/call | Saved ms/call | Reduction | BASE/DP CUDA interval outside iteration ranges (ms/call) |\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in comparisons:
        report+=f"| {r['dataset']} | {r['stage']} | {r['BASE_call_host_wall_ms']:.3f} | {r['DP_call_host_wall_ms']:.3f} | {r['saved_host_wall_ms_per_call']:.3f} | {r['call_host_wall_ms_reduction_pct']:.2f}% | {r['BASE_outside_iterations_cuda_ms_per_call']:.3f} / {r['DP_outside_iterations_cuda_ms_per_call']:.3f} |\n"
    report+='\nBA saves more absolute milliseconds per invocation, even though its percentage reduction is smaller. More batch elements or iterations do not imply a larger percentage reduction: both removable and unchanged work contribute to the larger denominator.\n\n'
    report+='## Two BA timing scopes\n\nIteration totals sum all recorded complete optimizer-iteration CUDA intervals within each BA invocation; these are not one-iteration averages or a continuously measured loop wall time. Complete invocation times are synchronized host wall latencies and include preparation, all iterations and finalization. Each percentage uses BASE and DP measured with the same scope and clock. Do not subtract the CUDA iteration total from host call time to attribute CPU overhead. SD in the CSV is over the three seed-wise equal-scene dataset means.\n\n'
    report+='| Dataset | Updates/call | BASE iteration total (CUDA ms/call) | DP iteration total | Iteration reduction | BASE complete call (host ms/call) | DP complete call | Complete-call reduction |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for r in comparisons:
        if r['stage']!='bundle_adjustment':continue
        report+=f"| {r['dataset']} | {r['BASE_iterations_per_call']:.0f} | {r['BASE_summed_iteration_cuda_ms_per_call']:.3f} | {r['DP_summed_iteration_cuda_ms_per_call']:.3f} | {r['summed_iteration_cuda_ms_per_call_reduction_pct']:+.2f}% | {r['BASE_call_host_wall_ms']:.3f} | {r['DP_call_host_wall_ms']:.3f} | {r['call_host_wall_ms_reduction_pct']:+.2f}% |\n"
    report+='\nCombining ENG and DP requires its own matched BASE/DP/ENG/ENG+DP measurements; percentages from separate ablations must not be added. The current engineering variant also changes ray handling, so its effect is not guaranteed to be exclusively outside the iterations.\n\n'
    report+='| Dataset | Stage | BASE/DP forward CUDA ms/iteration | BASE/DP backward CUDA ms/iteration | Forward reduction | Backward reduction |\n|---|---|---:|---:|---:|---:|\n'
    for r in comparisons:
        report+=f"| {r['dataset']} | {r['stage']} | {r['BASE_forward_cuda_ms_per_iteration']:.3f} / {r['DP_forward_cuda_ms_per_iteration']:.3f} | {r['BASE_backward_cuda_ms_per_iteration']:.3f} / {r['DP_backward_cuda_ms_per_iteration']:.3f} | {r['forward_cuda_ms_per_iteration_reduction_pct']:.2f}% | {r['backward_cuda_ms_per_iteration_reduction_pct']:.2f}% |\n"
    report+='''
## Confirmed code mechanisms and limits of attribution

- TUM/ScanNet BA evaluates dense feature smoothness with `smooth_pts=64`: `coordinates(63)` creates **250,047 feature-grid queries per iteration**, plus feature differences and their backward pass. Replica/Synthetic use `smooth_pts=32`, giving **29,791** feature-grid queries. This path is unchanged by DP and absent from tracking's loss. A small positive smoothness weight still executes it. These are feature queries, not extra RGB/SDF decoder evaluations.
- BA resamples keyframe/current-frame rays every iteration, reconstructs multi-pose tensors, steps the map optimizer and commits keyframe poses. The DP support mask must be rebuilt for each new BA ray batch. Tracking reuses its ray batch and cached support membership within a call; it steps one pose optimizer. The saved point counts cover ray rendering and exclude dense smoothness.
- Both stages retain dense anchor iterations. With the frozen period 3, 10 iterations contain 4 dense/6 pruned updates; 20 contain 8 dense/12 pruned updates. Doubling the iteration count does not increase the fraction of pruned updates.
- BA's recorded backward range includes gradients from dense smoothness. Its recorded model-forward range excludes construction of that smoothness loss, which occurs between the model-forward and backward ranges. Forward reductions alone therefore do not establish whole-iteration or whole-call reductions.
- The pinned public `freeze_model` helper assigns `require_grad`, rather than `requires_grad`. This analysis does **not** assume field-parameter gradients are absent during tracking; no baseline behavior was changed.

The outside-iteration quantity subtracts **CUDA intervals from CUDA intervals**, matched by frame and stage. It is not a subtraction of host wall time and GPU time. It contains setup/finalization, instrumentation gaps and possible GPU stream idle time while the host submits work; it is not a measurement of a specific CPU block or pure GPU compute. The existing data establish the large, almost unchanged BA remainder. Exact causal shares of smoothness, pose work, mask construction and optimizer cost still require disjoint component profiling. No causal ablation was run.

## Timing vocabulary used by the paper

| Unit | Meaning |
|---|---|
| Forward evaluation | One neural model/rendering forward evaluation inside an optimizer iteration; no backward or optimizer step. A BA forward range does not include the separately constructed smoothness loss. |
| Optimizer iteration | One complete loop update, including forward, loss assembly, backward and scheduled optimizer steps. CUDA intervals are retained separately from host wall latencies. |
| Co tracking call | One `tracking_render` invocation including all 10 iterations, preparation and finalization; once per input frame after initialization. |
| Co BA call | One `global_BA` invocation including all 10 iterations (Replica/Synthetic/ScanNet) or 20 (TUM), preparation and pose finalization; invoked every fifth frame. |
| Frame work | One complete post-loading frame-loop body: tracking, BA when scheduled, and keyframe work. `frames.csv` excludes DataLoader waiting and is distinct from a call latency. |
| Scene SLAM-work duration | `slam_work_wall_seconds`, from before the DataLoader loop through its end, including first-frame mapping, data loading and keyframe work; excludes model construction, post-run ATE and mesh extraction. It is not complete process execution time. |
| EC active worker call | One tracking/mapping worker invocation, all iterations and preparation/finalization, excluding explicitly recorded handshake waits. Worker intervals overlap and cannot be added into frame latency or FPS. |

Co `calls.csv` host wall values use synchronized call boundaries. `iterations.csv` values use CUDA-event intervals, not per-frame or whole-scene elapsed time. Dataset means are means of scene/seed call means; they are not a concatenation weighted by scene frame counts. The scene durations in native EC quality runs are a separate measurement scope and are never substituted for call timings.
'''
    (out/'REPORT.md').write_text(report)
    (out/'ANALYSIS_STATUS.json').write_text(json.dumps(dict(source_runs=len(rows)//2,phase_records=len(rows),new_gpu_experiments=False,input_sha256=inputs,comparisons=comparisons),indent=2)+'\n')
    print('DP_PHASE_TIMING_ANALYZED',len(rows)//2,'runs',len(comparisons),'dataset/stage comparisons')


if __name__=='__main__':main()
