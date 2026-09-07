# Co-SLAM Layer Timing Profiling Guide

This patch adds **selectable profiling layers** for Co-SLAM so you can measure either:

- **`scene_rep` mode**: deep forward-side timings inside `scene_rep.py`
- **`coslam` mode**: shallow timings in `coslam.py` for forward / loss / backward / optimizer
- **`frame_total` mode**: one compact row per frame, summing only `TR_ITER_PROFILE_TOTAL` and `BA_ITER_PROFILE_TOTAL`
- **`iteration_breakdown` mode**: one row per tracking or bundle-adjustment iteration with forward, backward, and total timing
- **`none`**: disables extra timing

Only **one layer is active at a time** to limit overhead.

## Files
- `coslam.py`: timing context + shallow layer timers
- `model/scene_rep.py`: deep forward-side timers
- `model/layer_timing.py`: shared CSV timing logger
- `configs/Replica/office0_layer_modes.yaml`: config for mode, warmup, frame limit

## Outputs
- `scene_rep` mode -> `scene_rep_timing.csv`
- `coslam` mode -> `coslam_layer_timing.csv`
- `frame_total` mode -> `frame_total_output_csv`
- `iteration_breakdown` mode -> `iteration_output_csv`

## Tools used
- **CUDA events** for GPU timing
- **NVTX ranges** for visual timeline inspection in Nsight
- **Nsight Systems / Nsight Compute** remain optional for deeper analysis

## Main config flags
```yaml
timing:
  mode: scene_rep   # or coslam, frame_total, iteration_breakdown, none
  warmup_frames: 5
  max_frames: 40
  disable_eval: True
```

For a full per-frame iteration total file:
```yaml
timing:
  mode: frame_total
  warmup_frames: 0
  max_frames: null  # process every frame in the sequence
  frame_total_output_csv: ./office0_frame_total_timing.csv
  frame_total_write_header: False
```

With `frame_total_write_header: False`, the file contains one data line per
processed frame. Set `max_frames` to an integer only when a frame cap is needed.
Columns are:
`frame_id,total_iter_profile_ms,tr_iter_profile_total_ms,ba_iter_profile_total_ms`.

## Run
```bash
sudo /bin/bash /home/nvidia/work/Co-SLAM/Orin\ tutos/profile_coslam.sh \
  --config configs/Replica/office0_layer_modes.yaml
```
It is preferred to run with sudo to avoid some errors. Use this config for profiling, because it disables mesh generation (overhead)

## Switch profiling layer
Edit:
```yaml
timing:
  mode: scene_rep
```
Change `scene_rep` to `coslam`, `frame_total`, `iteration_breakdown`, or `none`
as needed.

## Multi-dataset frame totals

Run all Replica, Synthetic, and TUM scenes in baseline and pruning modes:
```bash
python scripts/run/auto_run_frame_total_modes.py \
  --datasets Replica Synthetic Tum
```

Select scenes with bare names or dataset-qualified names:
```bash
python scripts/run/auto_run_frame_total_modes.py \
  --datasets Replica Synthetic Tum \
  --scenes office0 Synthetic:br Tum:fr1_desk
```

If `--scenes` is omitted, every scene in each selected dataset is included.
All available frames are processed by default. Use `--frame-count 2000` (or any
positive integer) to apply a limit, and `--frame-count all` for the explicit
unlimited form.
Timing CSV files are grouped under `results/frame_total/<Dataset>/`.
  example :

  python scripts/run/auto_run_frame_total_three_modes.py --datasets Replica --runs 5 --sizes 19 --mode baseline pruning pruningD

## Replica A/D per-iteration timings

There is a dedicated runner for the paper's plain-Co-SLAM comparison between
A (`CoherentPrime`, no Morton sort) and D (`Morton`, Morton sort):

```bash
python scripts/run/auto_run_replica_iteration_timings.py \
  --setups A D \
  --sizes 19 \
  --runs 5 \
  --frame-count all
```

With no `--scenes` argument, this processes all eight Replica scenes. Each
scene/setup/hash-size/run combination gets a unique CSV under
`results/iteration_breakdown/Replica/` and a row in
`output/Replica_iteration_timing_runs.csv`.

The timing CSV columns are:

```text
frame_id,iteration_type,iteration_id,forward_cuda_ms,backward_cuda_ms,full_iteration_cuda_ms
```

`iteration_id` is zero-based within each frame and iteration type. The runner
uses the focused `iteration_breakdown` timing mode, validates every CSV before
publishing it, and records only complete tracking and bundle-adjustment rows.
It explicitly disables the separate depth-pruning/bucketing paths so their
local Morton overrides cannot invalidate the A/D comparison. A and D remain
adjacent for each scene/hash size, while their execution order alternates
across scenes and repetitions to balance order and thermal effects.

`--warmup-frames` is a frame-ID threshold, not an iteration count. Frame 0 only
performs first-frame mapping, so values 0 and 1 retain the same tracking/BA
rows. With the default 2,000-frame Replica sequences, each complete output file
contains 19,990 tracking rows and 3,990 bundle-adjustment rows.
