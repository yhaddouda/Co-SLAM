"""Build the research report from saved, auditable full-scene measurements."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from summarize_pruning_research import collect


def metric(values, key, precision=3):
    x=[v[key] for v in values]
    return f'{statistics.mean(x):.{precision}f} ± {statistics.stdev(x) if len(x)>1 else 0.:.{precision}f}'


def verification_evidence(root):
    integrity = json.loads((root/'integrity_checks.json').read_text())
    tests = json.loads((root/'regression_checks.json').read_text())
    lines = ['## Completed verification', '',
        f"[Independent integrity checks](integrity_checks.json) passed for **{integrity['full_scene_runs']} full-scene runs** and {integrity['completed_runs']-integrity['full_scene_runs']} short smoke runs. Both direct and relative ATE were recomputed from saved trajectories with an independent NumPy SVD alignment. Actual network-query totals were cross-checked against sampling diagnostics; {integrity['source_files_verified']} archived source files passed SHA-256 verification. No ScanNet runs were present.", '',
        f"The [CPU regression suite](regression_checks.json) ran {tests['tests_run']} tests: {tests['passed']} passed and {tests['skipped']} historical comparison was skipped because its saved BA source snapshot is unavailable. Separate trained CUDA audits ran on all three datasets. Each uses four frozen-map conditions, including a 0.2 m pose perturbation, and compares outputs plus map, rotation and translation gradients. Tail is compared with the original dense objective; refine is compared with its matched refined-dense objective. These checks do not assert that refinement preserves the original unrefined objective.", '',
        '| Dataset | Matched objective | Cases | Maximum absolute output difference | Maximum relative gradient norm difference | Passed |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for dataset in ('Tum','Replica','Synthetic'):
        path=root/'audits'/dataset/'audit.json'
        audit=json.loads(path.read_text())
        for label,check in [('tail / dense',audit),('refine / refine_dense',audit['refined_objective_check'])]:
            output=max(e for r in check['results'] for e in r['maximum_absolute_output_errors'].values())
            gradient=max(e['relative_norm_error'] for r in check['results'] for e in r['gradient_errors'].values())
            lines.append(f"| [{dataset}](audits/{dataset}/audit.json) | {label} | {len(check['results'])} | {output:.3e} | {gradient:.3e} | {check['passed']} |")
    lines += ['', 'The audit tolerances were output rtol 2e-4 / atol 2e-6 and relative gradient norm error 0.003. The small measured differences support numerical consistency of the guarded objectives on these trained maps, without implying identical subsequent CUDA optimisation trajectories.']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--selected',required=True)
    args=p.parse_args()
    all_rows=collect(args.root)
    full={'Tum':592,'Replica':2000,'Synthetic':395}
    rows=[r for r in all_rows if r['frames']==full.get(r['dataset'])]
    groups=defaultdict(list)
    for r in rows:
        groups[(r['dataset'],r['preset'])].append(r)
    selected=args.selected
    complete=all(len(groups[(d,v)])>=3 for d in full for v in ('dense','legacy',selected))
    status='Confirmation complete.' if complete else '**DRAFT — experiments and multi-seed confirmation are still running; the candidate below is provisional.**'
    lines=[f'''# Scientific investigation of sample pruning

{status}

This report supersedes the earlier dense-sampling recommendation in `results/ba_accuracy/REPORT.md`. The task here is genuine sample pruning in tracking and BA. Dense sampling is only an accuracy control; execution optimizations are excluded from the proposed contribution.

Selected candidate: **`{selected}`**. Every reported reduction is measured from actual joint RGB/SDF network input points, including recovery queries. Initialization stays dense. See the tables for the remaining ATE differences; successful sample reduction does not imply universal preservation of quality.

## Experimental scope and comparators

Full sequences: **TUM fr1_desk, 592 frames; Replica office0, 2000 frames; Synthetic thin_geometry (`tg`), 395 frames**. No ScanNet experiments were run for this investigation. No mesh, depth-L1, reconstruction accuracy or completion metrics are reported.

The dense control and new policies share pose-storage and RGB-weight bug fixes, initialization, hash representation, learning rates, ray counts, optimization iterations, keyframe storage, original mixed-ray BA batches, and original pose commit. `ba_bucket.mode=stable`, preloaded BA batches, batched pose commit, tracking pose/cache execution variants, and Morton sorting are disabled in these experiments. This comparison attributes changes to sampling policies rather than those execution changes.

The legacy comparator uses the original shallow-depth tracking selection and depth-sorted BA buckets, with the same shared bug fixes. It is a controlled re-run, not a reproduction of historical DASIP table numbers. Pixel selection and ray-batch composition are part of its algorithm and differ from the new policies.

The first seed is exploratory. Additional seeds confirm controls and promising actual-pruning candidates. Seed 0 remains included. ATE uses rigid Horn alignment without scale fitting, in centimetres. Standard deviation is the sample SD; `n=1` rows are exploratory and their printed zero SD is not evidence of zero variance. CUDA hash-gradient accumulation is nondeterministic, so identical seeds do not guarantee bitwise-identical trajectories.

## What caused degradation in the original pruning design

1. The tracking ceiling was chosen using depth: grid coverage could be satisfied while selecting mostly shallow geometry. Spatial coverage did not preserve the original depth distribution of pose constraints. The new experiments retain the ordinary fixed tracking ray batch and prune points along its rays instead of excluding distant rays.
2. Depth-sorted BA buckets altered the composition of individual optimizer updates. Shuffling bucket order did not restore mixed-depth constraints within an update. New policies preserve the original mixed-ray BA draws and change only point evaluation/support.
3. A shortened near-to-depth linspace resampled coordinates and jitter intervals rather than just omitting unused points. It changed both what was supervised and how rendering sampled the field.
4. The original geometry MSE averaged masked residuals over every sample, including zeros behind the observed surface. If only those zero terms are removed but the denominator shrinks from N to K, the same retained residual sum is multiplied by N/K. Resampling also changes the mask-balancing counts. Thus a nominal query optimization can silently change the effective geometry loss. The proposed method keeps the reference denominator and mask measure; refinement splits its existing mass.
5. Missing-depth rays have different information from valid RGB-D rays. They retain dense sampling in the new policies rather than being forced into low-count/no-geometry updates.

The preceding investigation also fixed aliased pose storage and boolean conversion of fractional missing-depth RGB weights. Those are shared implementation corrections, not proposed scientific contributions. This study's dense and legacy comparators include them, so their benefit is not being attributed to the new pruning method. The controlled full-scene comparisons test the complete sampling designs; they do not isolate every listed cause individually.

## Main full-scene results

ATE: mean ± sample SD in **cm**. The selected column and dense control use the same seeds when calculating paired differences.

| Scene | Dense control | Legacy pruning | Selected pruning | Paired ATE change versus dense (mm) | n selected |
| --- | ---: | ---: | ---: | ---: | ---: |''']
    for d in full:
        a,b,c=groups[(d,'dense')],groups[(d,'legacy')],groups[(d,selected)]
        paired={r['seed']:r for r in a}
        delta=[10*(r['ate_cm']-paired[r['seed']]['ate_cm']) for r in c if r['seed'] in paired]
        change=f'{statistics.mean(delta):+.3f} ± {statistics.stdev(delta) if len(delta)>1 else 0.:.3f}' if delta else 'pending'
        lines.append(f"| {d} / {c[0]['scene'] if c else ''} | {metric(a,'ate_cm') if a else 'pending'} | {metric(b,'ate_cm') if b else 'pending'} | {metric(c,'ate_cm') if c else 'pending'} | {change} | {len(c)} |")
    lines += ['','## All tested policies','',
        'Full-scene ATE in cm, mean ± sample SD; number of seeds in parentheses. Use the per-seed measurements for paired comparisons. Exploratory single-seed policies have not received the same confirmation as the selected method.', '',
        '| Policy | TUM fr1_desk | Replica office0 | Synthetic thin_geometry |',
        '| --- | ---: | ---: | ---: |']
    for preset in ('dense','legacy','tail','margin','ht2','adaptive','features','refine','refine_dense','tail_ba','tail_tracking'):
        if not any(groups[(d,preset)] for d in full):
            continue
        cells=[]
        for d in full:
            values=groups[(d,preset)]
            cells.append(f"{metric(values,'ate_cm')} ({len(values)})" if values else 'pending')
        lines.append('| '+preset+' | '+' | '.join(cells)+' |')
    selection=args.root/'SELECTION.md'
    if selection.exists():
        lines += ['',selection.read_text()]
    lines += ['''
## Actual sample reduction and prototype cost

Percent reductions below compare each run's actual joint network queries with the full configured lattice for its ray batches. They include all fallback/recovery queries. All-stage reduction includes unchanged dense initialization. The separate SDF smoothness regularizer is unchanged and is not included in these joint RGB/SDF query counts. Missing-depth rays stay dense.

| Dataset | Tracking queries saved (%) | BA queries saved (%) | All-stage queries saved (%) | Recovery triggers (% rays) | Dense run time (s) | Selected run time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |''']
    for d in full:
        c=groups[(d,selected)];b=groups[(d,'dense')]
        if c and b:
            avg=lambda key:statistics.mean(r[key] for r in c)
            lines.append(f"| {d} | {avg('tracking_reduction_pct'):.2f} | {avg('ba_reduction_pct'):.2f} | {avg('all_stages_reduction_pct'):.2f} | {avg('recovered_rays_pct'):.2f} | {statistics.mean(r['run_seconds'] for r in b):.2f} | {avg('run_seconds'):.2f} |")
    lines += [r'''
Recovery triggers count the guard decision, including rays whose initial selection was already dense. Actual additional queries are accounted for separately in `pruning_diagnostics.csv` and included in the queried-sample totals above. These are research prototypes using ordinary PyTorch packing, selection, and diagnostics. Wall times include that overhead. Query reduction is the computational result; do not describe these figures as an FPS improvement unless the wall-time measurements support it. No engineering speedup is being substituted for sample pruning.

![ATE versus actual sample reduction](ate_vs_pruning.png)

## Algorithm and scientific rationale

### 1. Preserve coordinates and supervision, prune queries

Construct the same dense near-to-far and depth-centred lattice as the reference, including its original jitter. For ray r, retain every sample with z <= d_r + tau + m and the first two near samples. tau is the scaled TSDF truncation width; m is an optional margin. Query only the retained points. The zero-filled storage for omitted points is not a network evaluation.

This differs from reducing a uniform sample count and resampling a new interval: retained coordinates, jitter and the supervision measure stay tied to the dense reference. Keep the original dense denominator and global free-space/TSDF balancing. Samples behind d_r + tau have zero geometry masks, so dropping their network queries removes no supervised geometry term.

For rendering, detect the first sign change among retained samples. Recover a full ray when no sign change exists or its rendered support intersects the omitted tail. Reuse already evaluated points; recovery queries each missing point only once. With tail-only pruning, the retained prefix and recovered support preserve the dense objective in the analytic tests. Floating-point network/gradient reductions can still differ, and full trajectories must be tested.

The support argument is structural: a retained prefix contains every possible earlier sign change. If a crossing is found and all samples with z < z_first + tau are retained, all nonzero rendering weights and every geometry-supervised point have been evaluated. The omitted terms are zero in the reference objective. Otherwise evaluate the complete ray. This argument assumes identical pointwise field values and arithmetic; it is not a claim of bitwise-equivalent CUDA training, nor does it apply to free-space thinning that creates gaps before the first crossing.

### 2. Compensate for free-space thinning

The `ht2` candidate additionally keeps one sample out of every two in distant free space (z < d_r - tau - m), with a uniform random phase per ray. Keep all near-surface samples and the first two near samples. `ht4` uses one in four. A private RNG avoids changing the perturbation draws. The selected subset stays fixed within tracking of a frame; BA subsets vary by iteration.

For an original lattice of R rays and N points per ray, let K_ri indicate retention and pi_ri its inclusion probability. pi=1/stride for thinned free space and 1 for mandatory support. The compensated additive geometry estimator is

    L_fs_hat = w_fs / (R N) * sum_ri [K_ri / pi_ri * I_front(ri) * (s_ri - 1)^2].

The TSDF residual uses the same dense denominator and masks. For a fixed thinning policy, averaging over sampling phases recovers the dense additive geometry loss and its gradient. Unit tests verify this by enumerating all phases, including incomplete blocks. This is standard [Horvitz–Thompson inverse-probability estimation](https://www.tandfonline.com/doi/abs/10.1080/01621459.1952.10483446), not a newly invented estimator.

Rendering multiplies retained sigmoid-SDF weights by 1/pi before normalization. This compensates the sampling density but is a self-normalized ratio estimator: **the rendered RGB/depth estimate is not claimed unbiased**. A pruned gap can also hide a sign change. Retaining the full surface band and measuring full-scene ATE are therefore necessary; a geometry-loss proof alone is insufficient.

### 3. Recover unreliable rays

The `adaptive` candidate adds two empirical triggers to crossing/support recovery: normalized SDF magnitude at the retained sample nearest the measured depth > 0.35, or rendered/measured depth disagreement > 2 tau. Such a ray is evaluated densely for the current update. The triggers use sensor depth and map predictions, never evaluation ground-truth poses.

This trades some pruning for protection of model/sensor disagreement. It is a heuristic residual test, not calibrated predictive uncertainty and not a VolSDF opacity-error bound. Because recovery depends on evaluated values, the fixed-selection unbiasedness statement does not apply to the whole adaptive estimator.

### 4. Spend saved budget on surface resolution

The follow-up `refine` candidate keeps free-space queries and tail pruning, then inserts at most four midpoints in original lattice intervals nearest sensor depth, requiring both endpoints to lie in the TSDF band. For each selected interval [a,b], transfer 0.25 mass from each endpoint to a midpoint of mass 0.5. This preserves total quadrature mass and its first depth moment. Use those masses in rendering and geometry, while retaining the original reference denominator and mask-balancing weights. Missing-depth rays receive no new points.

The resulting nonlinear objective changes intentionally; matching zeroth/first moments is not a guarantee of lower error. The hypothesis is that finer surface resolution compensates for pruning better than thinning free space. Extra points and any recovery overhead are included in query counts, including cases exceeding the original dense count on a ray. This candidate was added after the initial Synthetic `ht2` degradation; it is a data-informed follow-up rather than a prespecified initial hypothesis.

The `refine_dense` ablation evaluates this same refined quadrature without pruning its tail. It is a scientific control, not an optimized candidate, and its query savings versus the original dense lattice are negative because of the added midpoints. Comparing it with `refine` separates refinement from pruning. Frozen loss/gradient tests compare the pruned method against this matched refined reference, rather than incorrectly asserting equivalence to the original unrefined dense objective.

## Candidate definitions and ablations

| Preset | Point-pruning rule | Quality compensation | Role |
| --- | --- | --- | --- |
| dense | None | Original dense renderer | Accuracy control only |
| legacy | Reduced/resampled depth intervals and depth-sorted BA pools | Original pruning behaviour, shared bugs fixed | Comparator |
| tail | Original lattice, cutoff d + tau | Dense geometry measure and crossing/support recovery | Proposed-method candidate |
| margin | tail, extra 0.15 m | Conservative surface support | Proposed-method candidate |
| ht2 | margin + one-in-two distant free-space thinning | Inverse-inclusion geometry, corrected rendering weights | Proposed-method candidate |
| adaptive | ht2 | SDF/depth-residual recovery | Proposed-method candidate |
| refine | tail + up to four surface interval midpoints | Split existing quadrature mass; keep reference geometry normalization | Data-informed follow-up candidate |
| refine_dense | Same surface refinement, no tail pruning | Same quadrature as refine | Matched objective ablation; not eligible as optimized method |
| features | adaptive | 25% Shi–Tomasi corners + grid-uniform tracking pixels | EC-inspired comparison; ineligible as contribution |
| tail_ba | tail in BA only | Dense tracking | Stage ablation, if measured |
| tail_tracking | tail in tracking only | Dense BA | Stage ablation, if measured |

All hyperparameters are shared across datasets; the methods do not use per-scene pruning thresholds. Detailed results, seeds, direct and relative ATE, query counts and timings are in [MEASUREMENTS.md](MEASUREMENTS.md) and [measurements.csv](measurements.csv). Every completed run is included there, including smoke checks. Any failed process is recorded rather than silently replaced.

## Relation to prior work and publication positioning

- [EC-SLAM paper](https://arxiv.org/html/2404.13346v2) uses depth-guided ray/surface sampling and describes feature/uniform pixel mixing. Its [pinned official Model.py](https://github.com/Lightingooo/EC-SLAM/blob/c269de61d7ff4623f3d5584aa47c6261c89dbad6/src/Model.py) constructs a free-space interval ending at 1.2 d and surface samples within +/-1.5 truncation widths. This work instead removes queries from an existing perturbed lattice and controls the resulting change in its objective. EC code is not imported or copied into the implementation.
- The [pinned Scene.py](https://github.com/Lightingooo/EC-SLAM/blob/c269de61d7ff4623f3d5584aa47c6261c89dbad6/src/Scene.py) disables feature/grid branches. Our `features` comparison implements the paper-level idea with standard corners/grid coverage. It is not an EC-SLAM reproduction and cannot establish superiority over EC-SLAM.
- [ESLAM (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Johari_ESLAM_Efficient_Dense_SLAM_System_Based_on_Hybrid_Representation_of_CVPR_2023_paper.pdf) and [GO-SLAM (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_GO-SLAM_Global_Optimization_for_Consistent_3D_Instant_Reconstruction_ICCV_2023_paper.pdf) motivate separating free-space exploration from surface sampling. That separation is established prior work.
- [VolSDF (NeurIPS 2021)](https://lioryariv.github.io/volsdf/) derives sampling control for its own opacity model. Our recovery heuristic has no corresponding certified error bound.
- [Uni-SLAM (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/html/Wang_Uni-SLAM_Uncertainty-Aware_Neural_Implicit_SLAM_for_Real-Time_Dense_Indoor_Scene_WACV_2025_paper.html) and [UncLe-SLAM (ICCV workshops 2023)](https://openaccess.thecvf.com/content/ICCV2023W/UnCV/html/Sandstrom_UncLe-SLAM_Uncertainty_Learning_for_Dense_Neural_SLAM_ICCVW_2023_paper.html) establish uncertainty-based reweighting and adaptive SLAM optimisation. Residual-dependent recovery alone should not be claimed novel.

The candidate scientific question is whether pruning relative to a fixed jittered lattice, preserving/estimating its supervision measure, and protecting pose optimisation reduces the quality penalty of depth-based pruning. That is different from copying EC-SLAM's sampler. The current experiments assess that question within Co-SLAM. They do **not** establish that the entire combination is novel across the literature, or that it outperforms complete external systems. Do not claim novelty for inverse-probability weighting, feature/grid sampling, packing, or depth-near samples individually.

## Validation and practical limits

- CPU regression tests compare losses and map/pose gradients on planes, curved fields, multiple crossings, missing depths, scaled truncation and late/no-crossing surfaces. Tests check RNG preservation, sample uniqueness and the fixed-selection geometry estimator.
- CUDA smoke and full-scene trajectories establish that actual queries are pruned in both stages and that recovered points are included in counts. Frozen-map equivalence is supporting evidence, not a substitute for ATE.
- One scene per dataset is a development set. Additional seeds test numerical variation, not generalisation to new scenes. Candidate selection on these scenes must be disclosed in a paper. No statistical-significance claim is made from these small samples.
- Mesh quality remains unmeasured. In particular, free-space thinning and even small trajectory changes may affect completion, thin geometry and surface artefacts.
- The full-trajectory error figure below aligns final saved poses once, including later BA refinements. It is not online tracking error. The selected method and dense control use identical evaluation conventions.

![Final trajectory errors, seed 0](trajectory_errors_seed0.png)

## Reproduction and artifacts

Use conda `coslam` on the CUDA host. [environment.json](environment.json) records Python, Torch, CUDA runtime, GPU driver and package versions. Every ATE-run directory stores `config.json`, `source_hashes.json`, exact executable source snapshots, `summary.json`, `trajectories.npz`, `frames.csv`, and pruning diagnostics when applicable. [PROTOCOL.md](PROTOCOL.md) records the initial hypotheses and subsequent data-informed extensions.

```bash
conda activate coslam
python scripts/Validation/run_pruning_research.py \
  --output results/pruning_research_reproduction/explore \
  --presets dense legacy tail margin ht2 adaptive features refine \
  --datasets Tum Replica Synthetic --seeds 0 --frames 0
python scripts/Validation/run_pruning_research.py \
  --output results/pruning_research_reproduction/confirm \
  --presets dense legacy tail refine \
  --datasets Tum Replica Synthetic --seeds 1 2 --frames 0
python scripts/Validation/run_pruning_research.py \
  --output results/pruning_research_reproduction/ablation \
  --presets refine_dense tail_ba tail_tracking \
  --datasets Tum Replica Synthetic --seeds 0 --frames 0
python scripts/Validation/summarize_pruning_research.py results/pruning_research_reproduction
conda run -n base python scripts/Validation/plot_pruning_research.py results/pruning_research_reproduction
```

Use a fresh output root after source changes. Plotting uses the base environment's Matplotlib; the local `coslam` environment lacks its `packaging` dependency. Training, CUDA audits and regression tests use `coslam`. No ScanNet dataset is available in the research runner's choices. The main implementation is `optimization/sample_pruning.py`, dispatched by `JointEncoding.forward` only for the configured tracking/BA stages. The experiment-specific depth/spatial and bucket flags are disabled for the new policies. The selected settings are documented with the final selection rationale and checked against the measured configurations in [selected_config_checks.json](selected_config_checks.json).

To repeat the separate trained-map objective audits and regression checks:

```bash
python -m unittest discover -s tests
python scripts/Validation/audit_pruning_objective.py --dataset Tum --scene fr1_desk \
  --frames 30 --output results/pruning_research_reproduction/audits/Tum
python scripts/Validation/audit_pruning_objective.py --dataset Replica --scene office0 \
  --frames 30 --output results/pruning_research_reproduction/audits/Replica
python scripts/Validation/audit_pruning_objective.py --dataset Synthetic --scene tg \
  --frames 30 --output results/pruning_research_reproduction/audits/Synthetic
python scripts/Validation/verify_pruning_results.py results/pruning_research_reproduction --final
```

For the user's subsequent mesh evaluation, the benchmark has an optional `--export-mesh` flag. It saves the final mesh after the ATE/timing summary, with mesh extraction excluded from training timings. It retains the paired RNG protocol, avoiding a separate training protocol for reconstruction evaluation. No mesh export was run during this investigation. For the selected `refine` candidate:

```bash
python scripts/Validation/benchmark_ba_accuracy.py \
  --dataset Replica --scene office0 --tracking baseline --variant baseline \
  --pruning refine --pruning-margin 0 --seed 0 --frames 0 \
  --output output/pruning_mesh/Replica_office0_seed0 --export-mesh
```

Equivalent scene arguments are `--dataset Tum --scene fr1_desk` and `--dataset Synthetic --scene tg`, with distinct output directories. This command requires a new training run because the ATE-only experiments do not save large map checkpoints. The export option is a reproduction convenience, not part of the proposed algorithm or a speedup.
''']
    report='\n'.join(lines)+'\n'
    if (args.root/'integrity_checks.json').exists() and (args.root/'regression_checks.json').exists():
        report=report.replace('## Validation and practical limits',verification_evidence(args.root)+'\n\n## Validation and practical limits')
    (args.root/'REPORT.md').write_text(report)
    print('Wrote',args.root/'REPORT.md')


if __name__=='__main__':
    main()
