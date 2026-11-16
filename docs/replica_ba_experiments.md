# Replica BA-only sampling experiments

The subsequent tracking and EC-SLAM mapping experiments are described in
[execution_experiments.md](execution_experiments.md). They test execution
changes with dense sampling, separately from these BA sampling variants.

The original `ba_bucket` implementation remains available. The new
`ba_experiment.variant` switch affects only global bundle adjustment. It does
not enable or disable tracking features; the benchmark explicitly disables
all three tracking samplers and the legacy pruning flag.

| Variant | `ba_bucket.enabled` | Behavior |
|---|---|---|
| `baseline` | False | Original random-ray BA and dense sampling. When the new switch is absent/default, setting `ba_bucket.enabled=True` still runs the original buckets. |
| `bucket` | True | Explicit name for the existing bucket implementation. |
| `bucket_dense` | True | Existing depth-sorted iteration batches, with dense baseline samples. Isolates batch segregation/order. |
| `bucket_reference` | True | Existing adaptive buckets, scaling geometry losses by reduced/dense sample count. Isolates the loss denominator; changed sample positions and mask balancing remain. |
| `lattice` | False | Baseline mixed ray batches, baseline lattice and perturbations, original geometry normalization, packed network queries up to measured depth + truncation + margin. Rendering support beyond that prefix is approximated. |
| `lattice_guarded` | False | Same lattice packing, with dense completion for rays whose first crossing or complete rendering support cannot be established. |
| `lattice_guarded_preload` | False | Guarded lattice, with the exact baseline per-iteration ray draws gathered ahead of time and transferred together. No depth sorting or change to optimizer update boundaries. |
| `baseline_fast` | False | Dense baseline sampling and optimizer schedule, with exact ray-batch preloading and batched pose finalization. Isolates BA data/finalization overhead from sampling. |
| `lattice_guarded_fast` | False | Guarded lattice sampling plus exact ray-batch preloading and batched pose finalization. |

The lattice variants currently require `training.n_importance=0`. The margin
must be finite and nonnegative and is in scene-coordinate distance units.
Replica includes a default configuration for this experimental switch; its
default `baseline` value preserves existing behavior.

The fast finalization reuses `pose_optim`, already recomputed after each pose
optimizer step. It detaches/clones the batch once and assigns its camera views
to the pose dictionary. The original path reconstructs, transfers an identity
matrix, and clones each camera separately after every BA invocation. This cost
grows with the number of keyframes and is outside the original iteration timers.
The fast variants preserve the map/pose update schedule and optimizer settings.

## Mechanisms in the original buckets

The original sampler changes three parts of the optimization at once:

1. It sorts the pooled rays by required sample count, so individual Adam map
   updates receive depth-homogeneous batches. The baseline mixes depths in each
   iteration. Shuffling bucket order does not restore that mixture.
2. It redistributes the uniform samples across `near -> measured_depth` rather
   than selecting points from the original `near -> far` lattice. The existing
   renderer does not compensate for that changed spacing with quadrature weights.
3. Free-space and SDF losses apply masks before an all-elements MSE. Removing
   zero-error back samples changes the denominator and the effective geometry
   weight relative to RGB/depth losses. For example, keeping the same supervised
   residuals but reducing 43 total samples to 25 multiplies that geometry loss by
   43/25 = 1.72. Changed positions also change the batch-wide mask-balancing counts.

`bucket_dense` and `bucket_reference` separate parts of these effects. The
guarded lattice keeps the baseline sampled objective, while the fast dense
variant removes data/finalization overhead without removing depth samples.

## Why the guarded variant preserves the sampled objective

Generate the complete baseline depth lattice and apply the original random
perturbations before selecting any points. Retain every point at or before
`target_depth + truncation + margin`, retaining full sampling for missing depths.
This keeps all points with nonzero free-space or surface masks. Reconstruct a
dense raw tensor from the packed predictions; omitted points have zero geometry
masks. The existing SDF loss therefore retains the same numerator, denominator,
and batch-wide balancing counts as dense BA.

For rendering, find the first adjacent SDF sign crossing in the retained prefix.
If there is no crossing, evaluate all omitted points for that ray. If a crossing
exists but an omitted baseline point lies before `z_first + truncation`, also
evaluate all omitted points for that ray. Otherwise every omitted point has
zero rendering weight under the existing renderer. Mask omitted predictions
out of rendering; do not invent SDF predictions for them.

This preserves the dense sampled objective and its gradients in exact arithmetic.
Changing CUDA batch sizes and atomic gradient accumulation can still introduce
floating-point differences, so full-trajectory ATE must be measured.

## Validation and benchmarking

Use the existing `coslam` conda environment, with its executables on PATH:

```bash
conda activate coslam
python -m unittest discover -s tests -p 'test_*.py'
python scripts/Validation/rank_replica_ba_scenes.py \
  --output results/ba_experiments/paper_scene_ranking.csv
python scripts/Validation/check_ba_lattice_gpu.py \
  --scene office1 \
  --checkpoint output/Replica/office1/run_1/baseline_Size19_Run1/checkpoint1999.pt \
  --output results/ba_experiments/gpu_equivalence.json
python scripts/Validation/run_replica_ba_suite.py \
  --scenes office1 room1 office0 \
  --variants baseline bucket lattice_guarded \
  --frames 0 --seeds 0 --output results/ba_experiments/full
python scripts/Validation/run_replica_ba_suite.py \
  --scenes office1 room1 office0 \
  --variants baseline_fast lattice_guarded_fast \
  --frames 0 --seeds 0 --output results/ba_experiments/full
python scripts/Validation/summarize_replica_ba.py \
  results/ba_experiments/full --reevaluate --plot
python scripts/Validation/run_replica_ba_suite.py \
  --scenes office1 room1 office0 \
  --variants baseline bucket bucket_dense bucket_reference lattice \
  --frames 200 --seeds 0 --output results/ba_experiments/screen200
python scripts/Validation/run_replica_ba_suite.py \
  --scenes office1 room1 office0 \
  --variants lattice_guarded lattice_guarded_preload \
  --frames 200 --seeds 0 --output results/ba_experiments/screen200
python scripts/Validation/summarize_replica_ba.py \
  results/ba_experiments/screen200 --reevaluate --plot
```

Each condition runs in a fresh process on the same GPU, sequentially.
Frame/phase seeds pair tracking ray selection, tracking perturbations, and
keyframe database sampling across conditions. The tracked algorithm and iteration
counts remain unchanged. ATE uses the repository's Horn rigid alignment without
scale fitting, calculated over the final trajectory; both raw and relative-pose
trajectories are retained. `trajectories.npz` allows independent re-evaluation.
The `--reevaluate` option recalculates every saved ATE using float64 coordinates
and checks the alignment against the repository's `evaluate_ate` function.

`frames.csv` records synchronized BA invocation wall time, including all
preprocessing and finalization, and synchronized complete frame wall time.
Frame zero and the first 20 frames are excluded from normal timing summaries.
Short smoke runs use a one-frame warmup. Evaluation and final validation occur
after timing. Diagnostic runs check every component and total loss, and their
timings must not be compared with normal benchmark timings. Each new run saves
its exact configuration, source copies, and source hashes.
The 200-frame runs screen mechanisms on all three scenes. Their ATE is for
that prefix and must not be compared directly with the full 2,000-frame ATE.
Network-sample totals include all BA invocations, including warmup; wall-time
summaries exclude the configured warmup frames.

The paper's seven paired BA-only ablations rank office1, room1, and office0 as
the three largest relative ATE degradation cases (47.01%, 35.59%, and 33.25%).
The source is `Quality/depth sampling/replica_tracking_ba_ate_results.csv`,
baseline A versus BA-only C. These ranks select the stress scenes; existing
paper measurements are not reused as timing or ATE baselines for new variants.

## Full-scene dense-fast results

`baseline_fast` is the strongest tested general candidate. On fresh full
2,000-frame runs in the `coslam` environment, with baseline tracking throughout:

| Scene | Baseline ATE (mm) | Dense-fast ATE (mm) | ATE change | BA-call time reduction | Complete BA-frame time reduction |
|---|---:|---:|---:|---:|---:|
| office1 | 5.009 | 4.958 | -1.01% | 49.44% | 36.56% |
| room1 | 7.243 | 7.278 | +0.48% | 45.62% | 31.33% |
| office0 | 5.260 | 5.411 | +2.89% | 48.13% | 35.01% |

Equal-weight means are +0.78% ATE change, 47.73% less BA-call time, and 34.30%
less complete BA-frame time. The original buckets measured +34.64% ATE change
with only 7.78% less complete BA-frame time under these same wall-time boundaries.
These runs use seed 0 on an NVIDIA Orin in its existing MAXN power mode, with
one GPU benchmark at a time. They establish the observed tradeoff on these three
scenes; they do not establish exact zero degradation or multi-seed uncertainty.

The dense-fast variant retains all 43 baseline depth samples, ray draws, and
optimizer update boundaries. Its gain comes from gathering/transferring the
same ray batches together and committing the already computed pose batch.
The 400-camera CUDA check measured approximately 480 ms for original singleton
finalization versus 2.8 ms for cached batch finalization, with maximum pose-matrix
difference 5.96e-8. Office1's first/last 50 BA calls after frame 100 changed
from 466/976 ms in baseline to 356/362 ms with dense-fast finalization.

The [12-page implementation report](baseline_fast_explained/baseline_fast_explained.pdf)
explains pose reuse and batched transfers, their relationship to the three bucket
mechanisms, the limits of ATE equivalence, and the engineering/research claims.
Its tables and runtime curves are generated from the saved full-scene results;
the [source and build instructions](baseline_fast_explained/README.md) are retained.

To select this variant while isolating BA, use:

```yaml
tracking_bucket:
  enabled: false
tracking_naive_pruning:
  enabled: false
tracking_depth_spatial:
  enabled: false
ba_bucket:
  enabled: false
ba_experiment:
  variant: baseline_fast
  margin: 0.0
```

Repository defaults still select `baseline`, and every original bucket path is
retained. Guarded packing should remain experimental: the full guarded-fast
room1 run degraded ATE by 66.92%, despite matching the dense sampled objective
in frozen-map tests. Dense objective equivalence alone is insufficient for
whole-trajectory validation.

Detailed measurements, both raw and relative ATE, exact configurations, source
snapshots, per-frame timings, and plots are under
`results/ba_experiments/full/`; the 200-frame mechanism ablations are under
`results/ba_experiments/screen200/`. `REPORT.md` and `comparison.csv` in each
directory distinguish those trajectory lengths.
The first office1 baseline preceded source-snapshot capture; its configuration,
trajectory, and timings are retained. The other 14 full runs include intact
source snapshots/hashes, and all six fast runs used the same core source.

## Mechanism-screening observations

On the paired 200-frame prefixes, `bucket_dense` still increased ATE on all
three scenes (+27.91% office0, +102.15% office1, +28.17% room1). Keeping dense
samples while retaining the bucket ray schedule did not recover the baseline.
`bucket_reference` improved room1 from 11.440 to 8.794 mm and office0 from
12.233 to 10.388 mm, but office1 remained approximately 6.45 mm versus its
3.897 mm baseline. Denominator correction therefore helps some cases but is
insufficient by itself.

The mixed-batch fixed-lattice prefix variant measured ATE changes of -0.25%,
+1.06%, and -8.63% for office0, office1, and room1, respectively. Its mean
complete BA-frame time reduction was only 4.11%. These are screening results,
not full-scene validation of the unguarded renderer. The full-tested dense-fast
variant has the stronger measured speed/accuracy tradeoff.
