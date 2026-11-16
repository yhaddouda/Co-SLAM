# Execution optimizations across tracking and mapping

This experiment tests whether the execution improvements behind the BA
`baseline_fast` result extend to other stages and a second SLAM implementation.
It preserves the dense objectives, ray counts and optimizer schedules. It does
not use depth buckets or Morton sorting. All measurements currently come from
the Jetson AGX Orin; an RTX 4070 is not available.

The implementations remain optional. Co-SLAM's default tracking variant is
`baseline`. EC-SLAM runs from an isolated copy of the user's current repository
at `experiments/ec_slam_execution`; the adjacent original repository and its
uncommitted edits were preserved. The copy's `UPSTREAM_SNAPSHOT.json` identifies
the starting sources. Its dataset symlink references the original data.
The [exported patch](../experiments/ec_slam_execution/execution_variants.patch)
contains the optional EC changes and baseline-default configuration. It was
dry-run checked against the recorded original sources. Apply it to another copy
only after `git apply --check` succeeds; its context targets the user's current
EC implementation, rather than a generic upstream EC-SLAM release.

## Results and evidence

The generated reports are the numerical sources of truth:

| System | Screening / mechanism ablations | Full trajectories |
|---|---|---|
| Co-SLAM | [200 frames](../results/execution_experiments/coslam/screen200/REPORT.md) | [2,000 frames](../results/execution_experiments/coslam/full/REPORT.md) |
| EC-SLAM | [200 frames](../results/execution_experiments/ec_slam/screen200/REPORT.md) | [2,000 frames](../results/execution_experiments/ec_slam/full/REPORT.md) |

Each report has an adjacent `comparison.csv`, `summaries.json` and tradeoff
plot. Individual run directories contain synchronized per-frame timings,
estimated/relative/ground-truth trajectories, resolved configuration, source
snapshots, source hashes and a device/software summary. A report exists only
after its comparisons have been generated; `SUITE_COMPLETE` records completed
conditions. [STATE.md](../results/execution_experiments/STATE.md) records the
running sequence while work is in progress.

All 21 primary full runs and both baseline repeats are complete. The
[consolidated report](../results/execution_experiments/REPORT.md) contains the
three-scene means and worst ATE changes. Co `cached_fast` is the strongest tested
tracking candidate: 13.72% less complete-call time and 13.22% less pure-frame
time, with mean ATE change −3.65% and worst change +2.08% (+0.116 mm on office0).
Its additional cache preserves fresh sampling jitter and all field queries.

EC mapping-only saves 7.24% of mapping-frame time across the scenes, with mean
ATE change −0.65% and worst change +19.22%. Combined EC tracking/mapping saves
4.41% of tracking-frame time and 7.72% of mapping-frame time, with mean ATE
change −1.52% and worst change +8.07%. Tracking-only saves 3.22% of frame time
on average, but barely changes time on room1/office0 and has a +52.83% worst
ATE change. Thus mapping's time benefit transfers more consistently than
tracking's; these results do not validate tracking-only quality equivalence.

The same-source office1 baseline repeats themselves change ATE by −7.41%
(Co) and +8.08% (EC). EC baseline tracking time changes by −3.81%, whereas its
mapping-frame time changes by about +0.09%. This limited repeat evidence makes
small tracking gains less secure than the larger, consistent mapping gains.
It does not explain away the +52.83% tracking-only ATE result or prove zero
expected degradation. The [repeat/telemetry report](../results/execution_experiments/diagnostics/REPORT.md)
and window plots retain these observations separately from treatment results.

CUDA loss/gradient checks and short traces live in
`results/execution_experiments/gpu_checks`; see the
[trace/validation report](../results/execution_experiments/gpu_checks/TRACE_REPORT.md).
Traced runs are diagnostic evidence;
their timings are excluded from the performance tables. Continuous read-only
telemetry is in `results/execution_experiments/hardware_telemetry.csv`. Telemetry
started partway through EC-SLAM screening and does not cover earlier runs.
The repeat telemetry file extends coverage through both baseline repeats.
Across 20,835 samples, CPU0 remains at 2,201.6 MHz and GPU at 1,300.5 MHz;
GPU temperature spans 44.88–55.59 °C. Dependency versions and actual installed
tiny-cuda-nn source/binding hashes are in `gpu_checks/environments`.

## What carries over from BA

The transferable idea is to identify data that remain constant over an
optimization invocation, keep them on the device, and avoid reconstructing or
transferring an already available result. The actual opportunity depends on the
stage's existing code.

| Stage | Existing behavior | Opportunity tested here |
|---|---|---|
| Earlier Co-SLAM BA | CPU sample draws transferred in every iteration; redundant pose finalization | Bulk transfer of already drawn samples and reuse of the final updated pose batch; see the earlier BA report |
| Co-SLAM tracking | One fixed pixel batch already transferred once per frame; dense depth lattice rebuilt every iteration | Reuse an unchanged pose matrix, allocate constants on the GPU, defer CPU best-loss decisions when early stopping cannot fire, cache the fixed lattice |
| EC-SLAM tracking | One fixed pixel batch; repeated scalar diagnostics and CPU loss decisions | Device pose allocation, reuse the selected pose matrix, defer selection and skip unused diagnostics |
| EC-SLAM mapping | Sample batches already preloaded; CPU trajectory copied to GPU in each iteration; repeated inverse of one reference pose | Cache fixed reference poses on GPU, reuse one final pose transfer and one reference inverse |

There is no new transfer batching benefit to obtain from tracking's pixels:
both tracking implementations already select one batch and reuse it. EC-SLAM
also already preloads mapping batches. Repeating the BA batching change there
would not explain a gain.

## Co-SLAM tracking variants

Source: `optimization/tracking_experiments.py` and `CoSLAM.tracking_render` in
`coslam.py`. Configuration: `tracking_experiment.variant`.

| Variant | Change |
|---|---|
| `baseline` | Original tracking calculations and CPU best-loss logic |
| `reuse_pose` | Reuse the matrix from the current forward pass when saving its pre-update best-pose candidate |
| `device_pose` | Construct identity constants directly on the reference device/dtype once per frame |
| `defer_best` | Save detached losses and pre-update pose matrices on GPU, then select the first strict minimum at finalization |
| `baseline_fast` | Combine all three changes, subject to the selection eligibility rule |
| `cache_depth` | Cache the fixed dense depth lattice and perturbation-bin boundaries once per tracking frame |
| `cached_fast` | Combine `baseline_fast` and `cache_depth` |

`reuse_pose` removes the second rotation/translation-to-matrix conversion in an
iteration. The original conversion for the forward pass and the second
conversion occur before the optimizer step, with identical rotation and
translation parameters. Detaching the first matrix therefore preserves the
pose associated with the measured loss. Reusing the matrix after an optimizer
step without this association would change the algorithm; that is not done.

`device_pose` retains the existing Rodrigues/quaternion arithmetic. It changes
where the identity tensors are allocated. The axis-angle implementation also
retains the baseline's behavior at a zero-angle singularity; this experiment
does not introduce a different rotation formula under an execution flag.

`defer_best` removes the per-iteration CPU loss copy and host comparison. It
stores the pre-update pose paired with every loss, then performs a device
minimum and one final host index read. The first minimum wins ties. The
baseline's first-iteration NaN behavior and later NaN comparisons are
preserved. Loss storage adopts the actual loss dtype rather than silently
rounding losses to the pose dtype. With `tracking.best=false`, the existing
last pre-step pose behavior remains in effect.

Deferral is allowed only when `wait_iters >= tracking.iter`, so the baseline's
early-stop rule cannot shorten that invocation. Otherwise the original CPU
decision and stopping behavior remain active. The experiment's Replica
configuration uses 10 tracking iterations and `wait_iters=100`.

The trace also revealed a separate invariant: with one fixed tracking pixel
batch, measured depths and the dense unperturbed sampling lattice do not change
over pose iterations. `cache_depth` builds that lattice and its lower/upper
perturbation-bin bounds once. Ray origins/directions and world points are still
recomputed from each current pose. Every iteration still draws its original
CPU `torch.rand` perturbations, at the original position in the RNG sequence.
Thus it does not freeze the random jitter or change sample counts, support,
spacing, sort order or geometry-loss normalization. The cache is local to one
tracking call, requires fixed target depths without gradients, and is disabled
with tracking pruning/buckets or the legacy adaptive-depth flag. BA receives no
cache. The CUDA check confirms exact samples, forward values and final CPU/CUDA
RNG states across three changing poses on each of three real frames.

For `I` pose iterations, `R` rays and `S` dense samples, the unchanged lattice
sort moves from roughly `O(I * R * S log S)` to `O(R * S log S)`. Perturbation
draws, world-point generation, field evaluation and differentiation still run
each iteration. This is reuse of phase-invariant sampling state; it is not a
reduction in the number of field queries. A sampler whose bounds/PDF depend on
the current pose or field cannot cache those changing quantities under the
same argument.

This additional Co-SLAM variant was introduced after the initial screening and
four traces. The full comparison uses a single source version with all three
conditions. The already completed original-source office1 baseline is retained
separately in `coslam/initial_full_source`; it is not silently substituted into
the matched full-suite table. EC-SLAM's tracking tests currently cover the pose
and host-selection variants, without a new depth-lattice cache.

All tracking sampling/pruning flags are disabled in the harness. The map runs
the dense baseline BA with `ba_bucket.enabled=false` and
`ba_experiment.variant=baseline`. Its body and update schedule are preserved;
the semantic test compares the BA function's AST with the earlier saved source.
BA has no phase timer and BA-frame times are not performance comparisons.

## EC-SLAM tracking variants

Source: `experiments/ec_slam_execution/src/execution_experiments.py` and
`src/Tracker.py`. The same five tracking names are available, although the
underlying opportunities differ from Co-SLAM.

`device_pose` implements the same quaternion formula as EC-SLAM's existing
`get_camera_from_tensor`, creating the 3x3 zero tensor directly on the GPU. It
preserves the original default-dtype allocation behavior. It does not replace
the quaternion conversion with a different library formula.

`reuse_pose` saves the already computed 3x4 matrix associated with the winning
pre-step camera tensor. Finalization uses that matrix instead of converting
the selected tensor again. EC-SLAM already makes only one matrix per tracking
iteration; this alone removes one conversion per frame, unlike Co-SLAM's
per-iteration redundant conversion.

`defer_best` stores losses and camera tensors on the GPU and selects after all
iterations. It preserves the original strict comparison against the initial
`1e10` ceiling, earliest tie, and `last_best` behavior. It also skips the
initial/per-iteration/final ground-truth-camera error diagnostics when those
values are not printed or used by optimization. Those diagnostics otherwise
perform reductions, device-to-host scalar reads and some ground-truth copies.
Deferral is disabled when verbose/debug tracker-loss output needs those values.
The Replica configuration performs eight pose iterations per frame.

Thus the combined EC-SLAM tracking variant partly removes unconsumed
instrumentation. A resulting gain would demonstrate an implementation
opportunity, not a new tracking objective. A lack of gain is equally useful:
device history writes, Python dispatch and the network's runtime can outweigh
the saved work. The measurements determine the result.

## EC-SLAM mapping variants

Source: `experiments/ec_slam_execution/src/Mapper.py`. Configuration:
`mapping_experiment.variant`.

| Variant | Change |
|---|---|
| `baseline` | Existing mapping calculations |
| `cache_poses` | Transfer the reference trajectory once per mapping invocation; update the selected entries on GPU; transfer the finalized optimized pose batch once |
| `reuse_inverse` | Compute the fixed current-pose inverse once for sliding-window selection |
| `baseline_fast` | Combine both changes |

With mapping BA active, the original loop clones the complete CPU trajectory,
transfers it to the GPU and writes the current optimized pose matrices into it
every iteration. The fixed entries do not change while tracking waits for the
mapping invocation. The variant transfers that reference once, stores selected
IDs on GPU, and uses an out-of-place `index_copy` to construct the current
trajectory. Gradient flow to the optimized entries remains intact. It still
constructs a per-iteration trajectory tensor; this experiment does not claim to
eliminate all trajectory allocation or bytes moved on the GPU.

At finalization EC-SLAM must reconstruct the optimized matrices after the last
pose optimizer step. The variant retains that reconstruction. It then performs
one CPU transfer and assigns the same values into the two separate destination
trajectory arrays. The earlier Co-SLAM BA final-matrix reuse is not applied
across this different update boundary.

Sliding-window selection compares the current pose to every keyframe. The
reference pose is constant across that loop, so the inverse is computed once.
Quaternion conversion, translation/angle scores, sorting and selected IDs keep
the original formulas. This opportunity grows with the number of keyframes;
a 200-frame prefix can understate its contribution on a 2,000-frame sequence.

## Benchmark protocol

The three scenes are `office1`, `room1` and `office0`, selected by the previous
saved BA stress-case ranking. Screening runs use 200 frames and all mechanism
ablations. Full runs use the complete 2,000-frame sequences:

- Co-SLAM: baseline, `baseline_fast` and `cached_fast`; BA baseline in all conditions.
- EC-SLAM: baseline, tracking only, mapping only, and combined tracking/mapping.

Every condition starts a fresh process in its matching conda environment
(`coslam` or `ec_slam`). Seed 0 is paired by frame and phase, including keyframe
sampling. This prevents one variant's random-number consumption from changing
the later phase's random stream. The initialization and sampling protocol is
common to every comparison, but differs from an ordinary unmodified runner's
single global stream. Conditions rotate order across scenes. Depth buckets,
Morton sorting, active profilers, mesh extraction, checkpoints and online ATE
evaluation are disabled for the headline timing runs. Both systems use
CoherentPrime hashing with size 19.

ATE is computed after all final mapping/BA updates. The primary value is the
raw estimated trajectory's rigid-aligned translation RMSE, using the
repository's Horn alignment accumulated in float64, with no fitted scale.
Configured dataset scale is undone. Relative-pose ATE is retained separately.
No per-frame ATE values are averaged into this final metric.

Timings use `perf_counter` with CUDA synchronization before and after complete
invocations. The first 20 frames are excluded from timing statistics, while
all frames remain in ATE. Mean, median, p95, standard deviation and counts are
saved. Reduction is `100 * (1 - variant_time / baseline_time)`; positive is
faster. ATE change is `100 * (variant_ATE / baseline_ATE - 1)`; positive means
degradation. Aggregate percentages average paired scene changes; they do not
pool all frames or compare different trajectory lengths.
EC's isolated conditions also retain timing differences in the untouched stage.
Those differences are contextual observations, not evidence that an execution
transformation was applied there: evolving poses/field states and run
variability can influence the other stage's cost.

| Timing | Included | Excluded |
|---|---|---|
| Co tracking call | Ray selection/transfer, complete pose loop, best-pose selection and final storage | BA, frame loading, evaluation |
| Co pure tracking frame | Loading and all tracking/frame work between completed frames | Frames that also run BA |
| EC tracking call | Entire `optimize_cam_in_batch`, including its setup and selection | Frame loading, mapping wait, pose initialization and final frame storage |
| EC tracking frame work | Frame transfers, pose initialization, optimization and storage | Loader and mapping wait |
| EC mapping call | Entire `optimize_map`, including window selection/setup/finalization | Outer loader and keyframe insertion |
| EC mapping frame work | Loader, keyframe insertion and full mapping invocation | Waiting for tracking |

EC-SLAM waits for mapping before the next frame's transfers in **every**
condition, including baseline. This prevents the next worker's transfer or
kernels contaminating the measured stage. End timers run before publishing
worker-state flags. Native optimizer update boundaries remain unchanged.
This common scheduling choice differs from the ordinary EC-SLAM runner and
must be stated when publishing these results. Tracking-frame time is not an
end-to-end frame latency including time spent waiting for mapping.

Final checks require the expected number of frames, completion of the last
mapping frame, finite model parameters and finite trajectories. Errors in EC
worker threads are propagated to the driver; incomplete runs cannot produce a
successful comparison. For the full EC runs, frames 20–1999 provide 1,980
tracking samples and 397 mapping samples (multiples of five plus frame 1999).
Co-SLAM provides 1,980 tracking calls and 1,584 pure tracking frames.

`audit_execution_results.py` independently reloads trajectories and recomputes
both ATE values, checks timing counts/frame ranges, compares paired resolved
configurations and verifies snapshot hashes. Each suite's `AUDIT.json` lists
the checked runs and any source-version differences. The first four office1 EC
screening runs precede an additional post-run finite-pose assertion in the
harness; their optimization/timing code is identical. The audit independently
checks those earlier saved trajectories for finiteness as well.

## Numerical validation and interpretation

CPU tests check original-versus-device pose values/gradients, deferred selection
semantics, early-stop/logging fallbacks, mapping pose replacement gradients,
ATE evaluator agreement, cached depth samples/losses/gradients/RNG, and
preservation of the Co BA body. All 25 pass. Separate short
CUDA diagnostics check every tracking/mapping loss for finiteness. Frozen CUDA
checks compare real field losses and gradients through map and pose parameters:
a trained Co-SLAM checkpoint on frames 0/300/900, and a real initialized
EC-SLAM model in both tracking and mapping modes. The latter is not a trained
EC-SLAM checkpoint. Pose-only microbenchmarks isolate conversion overhead and
include constant setup where relevant; they are not whole-stage speedups.

Final frozen checks also report rotation/translation or camera gradients
separately from map gradients. A small error normalized over millions of map
parameters could otherwise hide a larger discrepancy in the few pose
parameters. Those final JSON files are in `gpu_checks/final_validation`.
The final EC check additionally compares its actual eight-iteration tracking
optimizer loop on identical fixed rays and three perturbed initial poses. Its
real field is first fitted for 32 updates to one frame, then held fixed between
paired invocations. It records every loss, pre-step camera and pose gradient,
checks selected-iteration/camera/matrix association and final RNG states, and
reports numeric differences. This short-fitted field is not a full EC SLAM
checkpoint. The check was added in response to office1's observed full-run ATE
increase; it is a numerical diagnostic outside the timing suite.
The completed check found exact losses, pose gradients, all iterates, selected
poses/matrices and RNG states for all three starts. An additional run verifies
nonzero pose gradients and actual parameter movement; see
`gpu_checks/final_validation/ec_slam_nontrivial.json`. Co's final checks also
find exact rotation/translation gradients in the tested trained-field cases,
including the depth cache. Small aggregate field-gradient differences remain
recorded rather than being interpreted as a pose mismatch. These local checks
found no optimizer/selection inconsistency in the tested invocations, while
full-trajectory ATE still varies.

Same formulas and sampling do not guarantee bitwise identical trajectories.
Parallel reductions and atomic gradient accumulation can differ across runs;
small numerical differences enter later pose/map updates in a nonlinear SLAM
system. PyTorch describes both floating-point accuracy limitations and limits
to reproducibility. This is why final ATE is measured instead of inferred from
local equality checks. [Numerical accuracy](https://docs.pytorch.org/docs/2.3/notes/numerical_accuracy.html),
[reproducibility](https://docs.pytorch.org/docs/2.3/notes/randomness.html).

A single seed does not establish statistical equivalence or zero expected ATE
degradation. Same-source/same-seed office1 baseline repeats are also run in
separate `coslam/repeat_full` and `ec_slam/repeat_full` roots; they provide limited
repeat checks rather than statistical equivalence tests. The EC repeat was
added after its full tracking-only office1 condition showed a substantial ATE
increase, to avoid attributing that change to variability without checking EC
itself. A useful follow-up would pair multiple
seeds and more baseline repeats,
define a practical absolute ATE tolerance in advance, and report confidence
intervals over independent runs. Frames within one trajectory are correlated
and are not independent accuracy replicates. These runs measure ATE and time;
depth and 3D reconstruction equivalence are not established by this task.

## Orin and the RTX 4070 question

The measured board is a Jetson AGX Orin Developer Kit. CUDA reports compute
capability 8.7, 16 SMs and about 62,840 MiB addressable memory, with PyTorch 2.3
and CUDA 12.4. A read-only `nvpmodel -q` query reports MAXN. No power or clock
settings are changed by these experiments. The hardware JSON and telemetry
record actual queries; reported total memory is not free memory or memory
dedicated exclusively to GPU work.

The CUDA queries also report a 4 MiB L2 cache, two asynchronous engines, managed
memory support, and no direct pageable-memory or concurrent-managed access.
The driver reports virtual-memory-management support, while tiny-cuda-nn emits
its arena fallback warning in these runs. The local arena source also falls
back if address reservation fails, so that warning alone does not establish
that Orin lacks virtual-memory capability. The allocator path is a software
and platform observation to record in a future hardware comparison; no
controlled allocator ablation was performed. [Local arena source](../tiny-cuda-nn/include/tiny-cuda-nn/gpu_memory.h),
[upstream arena implementation](https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/gpu_memory.h).
The retained Co tracking trace ranges contain no `cudaMalloc`/`cudaFree`
calls. Each EC tracking trace records one `cudaMalloc` across 26 calls; each
mapping trace records one allocation and one free across six calls, with the
same counts in baseline and combined variants. Those short traces do not
support frequent allocator calls as the main explanation for the observed
steady-stage gain, and they do not establish allocator behavior over a full run.

Orin uses an Ampere GPU integrated with Arm CPUs. NVIDIA's AGX Orin 64GB
specification lists a 256-bit LPDDR5 interface and 204.8 GB/s nominal memory
bandwidth. The non-Super RTX 4070 uses Ada Lovelace, 5,888 CUDA cores and 12GB
GDDR6/GDDR6X on a 192-bit interface. These are architecture/specification facts,
not benchmark predictions. [Orin specification](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/),
[RTX 4070 specification](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070-family/).

On Tegra, CPU and integrated GPU allocations share physical SoC DRAM, but host,
device and unified memory retain different allocation/access/cache behavior.
An ordinary PyTorch CPU-to-CUDA tensor conversion is not automatically a free
alias. Orin does not have the full CPU/GPU system-memory coherency NVIDIA
documents for Thor. Therefore the Orin result should not be described as
removing a PCIe bottleneck, and shared DRAM alone cannot predict zero copy cost.
[CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html).

Several competing hypotheses need measurements:

- A faster RTX field evaluation could make unchanged host dispatch and scalar
  synchronization a larger fraction of stage time, increasing relative savings.
- A faster desktop CPU could reduce the very host work being removed, reducing
  relative savings instead.
- Discrete GPU transfers, pinned/pageable allocation, cache behavior and driver
  latency differ from Orin. Transfer counts/bytes and allocation behavior must
  be measured, rather than extrapolated from peak bandwidth.
- Field kernel duration depends on rays, sample counts, hash-grid working set,
  SM resources and clocks. Those costs do not scale uniformly with CUDA cores
  or advertised floating-point throughput.

A useful approximate model is
`T_stage = T_field + T_pose + T_host_dispatch + T_transfer + T_wait`.
The components can overlap, so summed CUDA API durations are not an additive
decomposition of wall time. In particular, a synchronization API often spends
its duration waiting for field work; it is not all removable CPU overhead.
The paired short traces use explicit `ExperimentTrackingCall` and
`ExperimentMappingCall` ranges with an end fence. Attribute API/copy/kernel
counts and GPU busy intervals to these complete calls, excluding the initial
first-frame mapping. Profile timings themselves are not the unprofiled result.

No RTX speedup is claimed. Future comparison should run the same scenes,
configuration and seeds on a native desktop installation, record the host CPU,
GPU, power/clocks and software versions, and use the same synchronized timing
boundaries. The existing Morton extension wrappers default their build target
to `8.7` even when sorting is disabled; set `TORCH_CUDA_ARCH_LIST=8.9` for the
RTX 4070 and build CUDA dependencies for that machine. An Arm conda environment
cannot simply be copied to an x86 host.

## What could support a paper

These variants are currently execution/engineering optimizations. Caching an
inverse, eliminating unused diagnostics, moving allocations and batching
transfers are established techniques; applying them to two repositories does
not by itself establish a novel SLAM estimator or a generally faster hash-grid
method. Their value is measured quality at unchanged work and a clear account
of where the runtime was spent.

A research contribution could be a validated execution model and transformation
strategy for iterative neural SLAM: identify phase-invariant state, preserve
pre/post-step semantics and sample/objective distributions, predict which
transformations help under different CPU/GPU/memory regimes, and demonstrate
those predictions across systems and hardware. Mixed or negative stage results
should be retained because they test the model's boundary conditions. Two
systems on Orin support an observation about those implementations; they do
not establish all hash-grid SLAMs or RTX behavior. Multi-seed full trajectories,
full quality metrics, additional hardware and comparison with existing
optimization techniques would strengthen a submission.

One further opportunity found in the code is outside the tested variants:
tracking computes field-parameter gradients even though it steps only the pose
optimizer. Co BA and EC mapping clear field gradients at their invocation setup,
so those tracking field gradients are discarded. A future variant could freeze
only field parameters during tracking, retain differentiation through the
field to ray/pose inputs, and restore each original parameter flag afterward.
The local tiny-cuda-nn binding checks `params.requires_grad()` before allocating
the parameter-gradient output. This could reduce backward memory/atomic work,
but needs dedicated pose-gradient, state-restoration, timing and full-ATE tests.
It has not been implemented or included in the present speedup tables.

Another possible follow-up is batching the remaining CPU perturbation-noise
transfers in Co tracking. The cache currently retains all ten original CPU
draws and per-iteration transfers. Preloading distinct noise tensors could
remove more transfer/host fences while keeping fresh jitter in every iteration.
It would require proving that draw shapes and order are fixed and that no
interleaved CPU RNG consumer is displaced; adaptive sampling can invalidate
that argument. Identical final RNG state alone would not prove identical
per-iteration samples. This has not been implemented or benchmarked here.

## Reproduce or extend

Run GPU suites serially; simultaneous jobs invalidate this timing comparison.
Choose a new output root rather than overwriting previous observations.

The suite configures isolation automatically. To use `cached_fast` directly in
Co-SLAM's usual runner, select it in the resolved configuration and disable the
tracking samplers that invalidate the dense cache:

```yaml
tracking_experiment:
  variant: cached_fast
tracking_bucket:
  enabled: false
tracking_depth_spatial:
  enabled: false
tracking_naive_pruning:
  enabled: false
training:
  uniform_samples_until_depth: false
ba_bucket:
  enabled: false
ba_experiment:
  variant: baseline
```

Keep the remaining baseline settings, including sample counts and optimizer
iterations. These are key overrides to merge into a complete config, not a
standalone scene configuration. The cache is deliberately ineligible while an
incompatible sampler is active; setting only the variant name in such a config
will not activate depth caching. The ordinary runner also retains its own RNG
stream and outputs, rather than adopting the experiment harness's phase seeds.

For EC-SLAM, the isolated workspace accepts `tracking_experiment.variant:
baseline_fast` and/or `mapping_experiment.variant: baseline_fast`. Set the
other stage to `baseline` for isolation. Deferred tracking selection requires
`verbose: false` and no active tracker-loss debug display; otherwise the
existing diagnostic/selection path remains active. Mapping pose caching applies
when its existing BA option is active. The exported patch adds these optional
choices to the recorded source snapshot.

```bash
conda run --no-capture-output -n coslam python -u \
  scripts/Validation/run_replica_execution_suite.py \
  --system coslam --frames 0 --seeds 0 \
  --scenes office1 room1 office0 --variants baseline baseline_fast cached_fast \
  --output results/execution_experiments/coslam/full_repeat

conda run --no-capture-output -n ec_slam python -u \
  scripts/Validation/run_replica_execution_suite.py \
  --system ec_slam --frames 0 --seeds 0 \
  --scenes office1 room1 office0 \
  --variants baseline tracking_fast mapping_fast both_fast \
  --output results/execution_experiments/ec_slam/full_repeat

python scripts/Validation/summarize_execution_experiments.py \
  --root results/execution_experiments/coslam/full_repeat --plot
python scripts/Validation/summarize_execution_experiments.py \
  --root results/execution_experiments/ec_slam/full_repeat --plot

conda run --no-capture-output -n coslam python \
  scripts/Validation/audit_execution_results.py --require-complete \
  --roots results/execution_experiments/coslam/full_repeat \
          results/execution_experiments/ec_slam/full_repeat
```

Use `--frames 200` for screening. Co-SLAM's individual mechanisms are
`reuse_pose device_pose defer_best`; EC-SLAM's are
`tracking_reuse_pose tracking_device_pose tracking_defer_best
mapping_cache_poses mapping_reuse_inverse`. Use `--seeds 0 1 2` for paired
independent runs; this is a proposed extension, not a claim that those runs have
already been completed. The harness copies the exact sources used by each run.

On a future RTX 4070 machine, install the native dependencies and stage the same
datasets, then prefix the above commands with `TORCH_CUDA_ARCH_LIST=8.9`.
Prepare the EC isolated workspace from that machine's current repository using
`prepare_ec_execution_workspace.py`, and apply the optional experimental patch.
The present symlink points to this Orin's dataset directory and must be recreated.
