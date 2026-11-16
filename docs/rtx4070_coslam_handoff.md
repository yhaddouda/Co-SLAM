# Co-SLAM experiment handoff to the RTX 4070

Prepared on 2026-10-04. The Orin experiments are complete. The user now has an
RTX 4070 machine on the same intranet, a Co-SLAM checkout, the `coslam` conda
environment, and Codex available there. No RTX benchmark has been run by this
Orin session. This file supplies the context needed by a new RTX Codex chat.

The immediate comparison is the completed **tracking-only execution suite**.
The earlier BA suite is a separate experiment, with separate output roots and
timing boundaries. A variant called `baseline_fast` exists in both suites;
those names refer to different stage-specific changes.

Read [the execution protocol](execution_experiments.md),
[the Orin tracking report](../results/execution_experiments/coslam/full/REPORT.md),
and, for the optional BA comparison,
[the BA protocol](replica_ba_experiments.md) and
[the Orin BA report](../results/ba_experiments/full/REPORT.md).

## Transfer and verify the working tree

The prepared package is `transfer/rtx4070/coslam_experiments.tar.gz` with its
`.sha256` checksum. Transfer those two files instead of selecting individual
files. [Packaging/extraction instructions](../scripts/Experiments/README.md)
explain staging, integrity verification, contents and exclusions. The extracted
`RTX_TRANSFER_MANIFEST.json` records the exact working-tree files and the
tiny-cuda-nn source commit used on Orin. The package also includes the optional
isolated EC source/patch and EC reference results; they do not change the
tracking-only scope below.

Use the latest Orin working tree, including untracked experimental files.
Matching repository names or Git commits alone is insufficient: the experiment
helpers and benchmark scripts are currently untracked, and runtime files have
uncommitted changes. Review differences before applying them to the RTX
checkout; preserve any independent RTX changes. Transfer source/configuration,
these documents, and the Orin reference results, using Git, an archive, or
SSH-based file transfer. Keep the RTX machine's native environment and compiled
dependencies.

Important experiment additions are `optimization/ba_experiments.py`,
`optimization/tracking_experiments.py`, and the relevant files under
`scripts/Validation/`. Runtime changes include `coslam.py`, `model/scene_rep.py`,
and `model/utils.py`. Include the dataset loader, model, optimization, encoding
and CUDA-extension sources used by the existing checkout, and the Replica
configs. Do not copy an Arm conda environment or Arm `.so` files to x86.

The saved tracking runs record the runtime source hashes in
`results/execution_experiments/coslam/full/office1/baseline_seed0/source_hashes.json`.
Compare those paths against the RTX working tree before timing. Compare the
resolved configuration with the corresponding Orin `config.json`; only local
data/output paths should need adjustment. Record any other differences rather
than silently treating the runs as identical.

Check that `../data/Replica/{office1,room1,office0}` resolves to the same Replica
data as on Orin, or adjust the scene data paths. These are the three scenes
selected from the saved BA degradation ranking, rather than the earlier
fallback scene list. Confirm all 2,000 RGB/depth frames and the trajectories
are available on each scene.

## Prepare the RTX execution context

Run from the Co-SLAM repository root in `coslam`. Confirm CUDA identifies the
RTX 4070 and that PyTorch, tiny-cuda-nn and PyTorch3D import correctly. Use the
native dependency builds already installed there, matching source versions to
the Orin environment where possible; record differences. The RTX 4070 compute
capability is 8.9. Prefix commands with `TORCH_CUDA_ARCH_LIST=8.9` because some
custom extensions may compile on import even when their optimization is off.
This variable does not rebuild an incompatible preinstalled tiny-cuda-nn binding.

Capture metadata before running:

```bash
TORCH_CUDA_ARCH_LIST=8.9 conda run --no-capture-output -n coslam python \
  scripts/Validation/capture_execution_environment.py \
  --output results/execution_experiments/rtx4070_metadata/environment.json

TORCH_CUDA_ARCH_LIST=8.9 conda run --no-capture-output -n coslam python \
  scripts/Validation/capture_execution_hardware.py \
  --output results/execution_experiments/rtx4070_metadata/hardware.json
```

Also record the desktop CPU, NVIDIA driver, GPU power limit and clock behavior.
Monitor GPU utilization, clocks, temperature and power with `nvidia-smi` during
the benchmark. The existing Orin sysfs monitor does not provide desktop NVIDIA
GPU telemetry. Run one benchmark process at a time, with no competing GPU work.
Separate import/compilation smoke runs from measured full runs. Keep the machine
awake and use a persistent terminal or process supervisor for long suites.

## Matched full tracking suite: nine runs

```bash
TORCH_CUDA_ARCH_LIST=8.9 conda run --no-capture-output -n coslam python -u \
  scripts/Validation/run_replica_execution_suite.py \
  --system coslam --frames 0 --seeds 0 \
  --scenes office1 room1 office0 \
  --variants baseline baseline_fast cached_fast \
  --output results/execution_experiments/coslam/rtx4070_full

conda run --no-capture-output -n coslam python \
  scripts/Validation/summarize_execution_experiments.py \
  --root results/execution_experiments/coslam/rtx4070_full --plot

conda run --no-capture-output -n coslam python \
  scripts/Validation/audit_execution_results.py --require-complete \
  --roots results/execution_experiments/coslam/rtx4070_full
```

`--frames 0` means the full 2,000-frame trajectory. Seed 0 and the 20-frame
timing warmup match the primary Orin runs. The harness keeps dense baseline BA,
disables the experimental tracking samplers and BA buckets, preserves ray/sample
counts and optimization schedules, and uses the same paired frame/phase seeds.
It times synchronized complete tracking calls and complete pure tracking frames
(excluding frames that run BA). Evaluation/mesh/profiler work is disabled by the
same harness. Do not enable `--diagnostic` for reported timings.

`baseline_fast` combines pose/setup/selection execution changes. `cached_fast`
also caches the fixed dense depth lattice while generating fresh perturbations
at each iteration. It keeps current-pose world points and neural-field queries.
See the protocol for the detailed semantic contracts and eligibility guards.

Use a new output directory for smoke tests (`--frames 20`), screening
(`--frames 200`), or reruns. Never compare their ATE with the full-scene ATE.
The full suite writes a `SUITE_COMPLETE` marker; each run saves `frames.csv`,
`trajectories.npz`, `summary.json`, resolved config and source snapshots.

The Orin equal-weight mean reductions were 6.14% in tracking-call time and
6.04% in pure tracking-frame time for `baseline_fast`, and 13.72% / 13.22%
for `cached_fast`. These are observed seed-0 results, not expected RTX gains.
ATE changes varied across scenes; a same-source/same-seed office1 baseline
repeat changed Orin ATE by about 7.4%. Do not interpret one matched seed as proof
of zero degradation or architecture causality.

For the corresponding baseline repeat, run office1 / baseline / seed 0 into a
separate root, as on Orin. For a stronger paper study, subsequently run seeds
0/1/2 on both machines and more baseline repeats. Preserve the first matched
seed-0 comparison and label the extension separately.

## Optional earlier BA suite: fifteen runs

Run this after the tracking suite, if repeating the earlier BA comparison is
also desired. It uses baseline tracking throughout. Its `baseline_fast`
preloads the same BA ray batches and reuses the final optimized pose batch.

```bash
TORCH_CUDA_ARCH_LIST=8.9 conda run --no-capture-output -n coslam python -u \
  scripts/Validation/run_replica_ba_suite.py \
  --scenes office1 room1 office0 --frames 0 --seeds 0 \
  --variants baseline bucket lattice_guarded baseline_fast lattice_guarded_fast \
  --output results/ba_experiments/rtx4070_full

conda run --no-capture-output -n coslam python \
  scripts/Validation/summarize_replica_ba.py \
  results/ba_experiments/rtx4070_full --reevaluate --plot
```

The BA summary reports complete synchronized BA invocations, including setup
and finalization, and complete BA frames, including baseline tracking. Check
that all 15 summaries and full trajectories exist; this older suite prints
`SUITE_COMPLETE` in its log but does not write the tracking suite's marker file.
The original Orin BA suite launched the first three variants and the last two
in separate batches; this combined RTX command has a different run order.
Record that difference when assessing timing/order effects, or use the two
original batches shown in the BA protocol to reproduce that order.

## Deliver the comparison

Return the new RTX reports, raw per-frame times, trajectories, source/config
snapshots and hardware/environment metadata. Compare each variant with its
own GPU's baseline, paired by scene, seed, trajectory length and timing boundary.
Report both absolute times and percentage reductions, plus ATE in millimetres
and relative changes. Then compare the paired speed reductions between Orin and
RTX. Different host CPUs, memory systems and dependency builds remain part of
the platform comparison; two machines alone cannot isolate GPU architecture.

Keep all Orin observations under their existing roots and RTX observations
under new roots. No RTX result should be inferred from the Orin measurements.

## Prompt for Codex on the RTX machine

> Read `docs/rtx4070_coslam_handoff.md` and `docs/execution_experiments.md`.
> Reproduce the full Co-SLAM tracking-only suite on this RTX 4070, using the
> `coslam` conda environment, office1/room1/office0, seed 0, and
> baseline/baseline_fast/cached_fast. Verify the experimental source and resolved
> configuration against the transferred Orin reference first. Keep BA at
> baseline. Run serially, record environment and hardware metadata, audit the
> completed results, and report final ATE and synchronized call/frame times
> against both the RTX baseline and the saved Orin results. Preserve existing
> code and results. Follow the handoff file for dataset paths and native CUDA
> dependency handling.

Append an explicit request for the optional BA suite if desired; the prompt
above requests only the tracking suite.
