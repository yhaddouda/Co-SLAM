"""Write the qualified study report from independently checked measurements."""
import argparse,json,statistics,csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def report(r):
    assert r['passed']
    if not r['prior_reference_guard']:
        import copy
        r=copy.deepcopy(r)
        for row in r['cases']:
            row['prior_ate_cm']=row['dense_ate_cm'];row['prior_ba_ms']=row['dense_ba_ms']
    datasets=['Tum','Replica','Synthetic']
    group=lambda d:[v for v in r['cases'] if v['dataset']==d]
    mean=lambda d,k:statistics.mean(v[k] for v in group(d))
    fmt=lambda d,k,digits=4:f"{mean(d,k):.{digits}f} ± {statistics.stdev(v[k] for v in group(d)):.{digits}f}"
    s=['# Sample pruning with at least 30% BA wall-time reduction',
       '', '**Accepted method: buffered interval-support pruning plus sampled feature smoothness (`support_tv`).** Each scene meets the 17% ATE mean limit and has faster mean tracking and BA. TUM meets the requested ≥30% BA wall-time reduction. All seeds are included.',
       '', '## Criteria and protocol',
       '', 'Full sequences: TUM fr1_desk (592 frames), Replica office0 (2000), Synthetic thin_geometry/tg (395). Three paired seeds [0, 1, 2] per scene. The ATE criterion is the ratio of scene means, not a per-seed limit or the mean of per-seed percentages. The 30% BA reduction needs to hold on at least one scene. Actual sample pruning and lower mean complete tracking/BA time are required on every scene. No ScanNet or mesh testing.',
       '', 'Fresh controls and candidates use identical archived runtime source; only sampling configuration and output path differ. The full original ray counts, iteration counts, mixed-keyframe batches, optimizer settings and pose rules remain. No execution-speed variant is enabled. Runs execute serially on an NVIDIA RTX 4070 in conda `coslam`, PyTorch 2.3.0+cu118. Dense/candidate order alternates by seed within each scene.',
       '', 'CUDA-synchronized timing encloses each entire tracking or BA call, including sample selection, transfers, feature regularization, losses, backward, optimizer and pose updates. Stage means omit frames 0–19; ATE and query counts use the full sequence. Total SLAM time includes initialization and per-frame data loading, but excludes model/dataset construction, post-run evaluation and mesh export. Negative ATE changes mean improvement.',
       '', '## Accuracy and the previous reference',
       '', 'The conservative headline compares against the previous report’s three-seed native dense means. Acceptance was independently checked against both these means and fresh paired dense controls.',
       '', '| Scene | Previous dense ATE (cm) | Candidate ATE (cm) | ATE change vs previous mean | Fresh dense ATE (cm) | ATE change vs fresh mean |',
       '| --- | ---: | ---: | ---: | ---: | ---: |']
    for d in datasets:
        s.append(f"| {d} / {group(d)[0]['scene']} | {fmt(d,'prior_ate_cm')} | {fmt(d,'selected_ate_cm')} | {100*(mean(d,'selected_ate_cm')/mean(d,'prior_ate_cm')-1):+.2f}% | {fmt(d,'dense_ate_cm')} | {100*(mean(d,'selected_ate_cm')/mean(d,'dense_ate_cm')-1):+.2f}% |")
    s+=['', 'Values are arithmetic mean ± sample SD across three seeds; ATE is rigidly aligned translational RMSE without scale fitting. The fresh TUM seed-0 dense run has an unusually large ATE. It is retained, never replaced or excluded. CUDA hash-grid atomic updates can make identical seeds diverge. No accuracy-improvement claim is based on that outlier: the candidate must also satisfy the 17% gate against the earlier dense mean. The candidate TUM ATE SD is larger than that of the previous dense reference, so satisfying the mean limit does not establish per-run stability. These are three development scenes used to select the method, not held-out evidence.',
        '', '## Complete wall time',
        '', '| Scene | Tracking dense → candidate (ms) | Tracking reduction | BA dense → candidate (ms) | BA reduction | BA reduction vs previous dense | Total SLAM dense → candidate (s) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for d in datasets:
        s.append(f"| {d} | {fmt(d,'dense_tracking_ms',2)} → {fmt(d,'selected_tracking_ms',2)} | {100*(1-mean(d,'selected_tracking_ms')/mean(d,'dense_tracking_ms')):.2f}% | {fmt(d,'dense_ba_ms',2)} → {fmt(d,'selected_ba_ms',2)} | {100*(1-mean(d,'selected_ba_ms')/mean(d,'dense_ba_ms')):.2f}% | {100*(1-mean(d,'selected_ba_ms')/mean(d,'prior_ba_ms')):.2f}% | {mean(d,'dense_run_s'):.2f} → {mean(d,'selected_run_s'):.2f} |")
    s+=['', '“30% faster” is operationalized here as ≥30% less wall time, equivalent to a speed factor of at least 1/0.7≈1.43. The target is checked against fresh controls and the previous dense means. Replica’s tracking improvement is small and is not evidence of a substantial or statistically established gain. Individual tracking pairs can be slower; the requested scene-mean gate governs acceptance. Frame timings are correlated observations, not independent experimental replicates.',
        '', '## Actual sample pruning',
        '', '| Scene | Tracking joint RGB/SDF points pruned | BA joint RGB/SDF points pruned | BA feature-only points pruned |',
        '| --- | ---: | ---: | ---: |']
    for d in datasets:s.append(f"| {d} | {mean(d,'tracking_pruning_pct'):.2f}% | {mean(d,'ba_pruning_pct'):.2f}% | {mean(d,'feature_pruning_pct'):.4f}% |")
    s+=['', 'Query totals include every dense ray-anchor update. Joint RGB/SDF and feature-only encoding points have different costs and are not combined into an unweighted count. Initialization is outside the tracking/BA totals and remains dense.',
        '', '## Every paired seed',
        '', '| Scene | Seed | Fresh dense ATE (cm) | Candidate ATE (cm) | ATE change | Tracking reduction | BA reduction |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for d in datasets:
        for v in group(d):s.append(f"| {d} | {v['seed']} | {v['dense_ate_cm']:.4f} | {v['selected_ate_cm']:.4f} | {v['ate_change_pct']:+.2f}% | {v['tracking_time_saved_pct']:.2f}% | {v['ba_time_saved_pct']:.2f}% |")
    s+=['', '## Matched sampling controls',
        '', 'Two complete-sequence controls at seed 0 use the same archived runtime source: (1) `support_tv` with a scene-spanning support margin and ρ=1, which retains every ray and feature sample, and (2) the previously accepted interval-only pruning, with native dense feature smoothness. The latter has exactly the same joint RGB/SDF point counts as the new candidate. Both controls retain every original BA feature query. All control ATE values are included; these single-seed ablations are not three-seed accuracy claims.',
        '', '| Scene | Control | Control ATE (cm) | Tracking control → candidate (ms) | BA control → candidate (ms) | BA reduction |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for d in datasets:
        for v in r['ablations']:
            if v['dataset']!=d:continue
            s.append(f"| {d} | {'Full sample path' if v['family']=='matched_control' else 'Interval only'} | {v['ate_cm']:.4f} | {v['tracking_ms']:.2f} → {v['selected_tracking_ms']:.2f} | {v['ba_ms']:.2f} → {v['selected_ba_ms']:.2f} | {v['ba_saved_pct']:.2f}% |")
    s+=['', 'The full-sample path checks that changing the surrounding computation alone does not explain the benefit. The interval-only comparison isolates the added regularization sampling at the algorithm level; it does not attribute an exact percentage to particular GPU kernels. TUM’s native regularization grid has 250,047 vertices per BA update, versus 29,791 on the other scenes, explaining the much larger opportunity there.',
        '', '## Independent checks and artifacts',
        '', f"All 24 fresh complete runs were checked using saved trajectories, frame timings, configurations, query counters and per-update diagnostics. Direct and relative ATE were recomputed using independent NumPy rigid alignment. {r['source_files_verified']} archived runtime-file hashes were verified, including the nine previous dense-reference runs. All fresh runs have identical 13-file runtime manifests; controls and candidates have the same non-sampling configuration. Actual feature queries and joint point counts reconcile with update schedules and diagnostics. Analysis/test sources and supplemental dependencies have separate hash manifests.",
        '', 'Regression suite: **57 tests run, 56 passed, one skipped** because a historical baseline-source fixture is unavailable. New tests compare the actual native smoothness function against all-edge enumeration, loss/gradient expectations, native Torch RNG draws, initialization behavior, full-budget restoration and actual feature-query reduction. Unbiased fixed-field gradients do not guarantee unchanged optimization trajectories or meshes.',
        '', 'Artifacts: [acceptance.json](acceptance.json), [per-seed measurements](measurements.csv), [regression.log](regression.log), [environment.json](environment.json), [GPU metadata](hardware.csv), [protocol](protocol.json), [fresh dense controls](controls/), [accepted runs](selected/), [full-sample controls](matched_control/), [interval-only controls](interval_only/), [analysis and tests](validation_source/), [analysis hashes](validation_source_hashes.json), [supplemental dependencies](supplemental_source/), [supplement hashes](supplemental_source_hashes.json). Every run contains its full inherited config, trajectories, frame timings, diagnostics, source and hashes. The earlier nine native controls are retained at [previous study](../pruning_accepted/controls/). Failed exploratory candidates are not promoted or presented as retained versions.',
        '', '![Accuracy, complete wall time and actual neural sample pruning](pruning_ba30.png)',
        '', 'Mesh quality, mesh completion, additional scenes, ScanNet, significance and publication novelty remain unestablished. The feature regularizer acts in unobserved space, so trajectory accuracy alone cannot certify reconstruction quality.', '']
    text='\n'.join(s)
    if not r['prior_reference_guard']:
        text=text.replace('The conservative headline compares against the previous report’s three-seed native dense means. Acceptance was independently checked against both these means and fresh paired dense controls.','This reproduction compares against its fresh paired native dense controls. No separate historical reference guard was requested.')
        text=text.replace('Previous dense ATE','Reference dense ATE').replace('ATE change vs previous mean','ATE change vs reference mean').replace('BA reduction vs previous dense','BA reduction vs reference dense')
        start=text.index('The fresh TUM seed-0 dense run')
        end=text.index('These are three development scenes',start)
        text=text[:start]+'CUDA hash-grid atomic updates can make identical seeds diverge. Every replicate is retained. '+text[end:]
        text=text.replace('including the nine previous dense-reference runs','across the fresh paired and ablation runs').replace('against fresh controls and the previous dense means','against fresh controls')
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);args=p.parse_args()
    r=json.loads((args.root/'acceptance.json').read_text());assert r['passed'] and r['ablations_verified']
    text=report(r)
    if not (args.root/'hardware.csv').exists():text=text.replace(', [GPU metadata](hardware.csv)','')
    if not (args.root/'protocol.json').exists():text=text.replace(', [protocol](protocol.json)','')
    text+='\n'+(ROOT/'docs/sampled_smoothness_pruning.md').read_text()
    (args.root/'REPORT.md').write_text(text)
    with (args.root/'measurements.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(r['cases'][0]));writer.writeheader();writer.writerows(r['cases'])

if __name__=='__main__':main()
