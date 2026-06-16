# Co-SLAM Layer Timing Profiling Guide

This patch adds **selectable profiling layers** for Co-SLAM so you can measure either:

- **`scene_rep` mode**: deep forward-side timings inside `scene_rep.py`
- **`coslam` mode**: shallow timings in `coslam.py` for forward / loss / backward / optimizer
- **`frame_total` mode**: one compact row per frame, summing only `TR_ITER_PROFILE_TOTAL` and `BA_ITER_PROFILE_TOTAL`
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

## Tools used
- **CUDA events** for GPU timing
- **NVTX ranges** for visual timeline inspection in Nsight
- **Nsight Systems / Nsight Compute** remain optional for deeper analysis

## Main config flags
```yaml
timing:
  mode: scene_rep   # or coslam, frame_total, none
  warmup_frames: 5
  max_frames: 40
  disable_eval: True
```

For a full per-frame iteration total file:
```yaml
timing:
  mode: frame_total
  warmup_frames: 0
  max_frames: 2000
  frame_total_output_csv: ./office0_frame_total_timing.csv
  frame_total_write_header: False
```

With `frame_total_write_header: False`, a 2000-frame run writes exactly 2000 data lines.
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
Change `scene_rep` to `coslam`, `frame_total`, or `none` as needed.
