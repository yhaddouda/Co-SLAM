# Co-SLAM Layer Timing Profiling Guide

This patch adds **selectable profiling layers** for Co-SLAM so you can measure either:

- **`scene_rep` mode**: deep forward-side timings inside `scene_rep.py`
- **`coslam` mode**: shallow timings in `coslam.py` for forward / loss / backward / optimizer
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

## Tools used
- **CUDA events** for GPU timing
- **NVTX ranges** for visual timeline inspection in Nsight
- **Nsight Systems / Nsight Compute** remain optional for deeper analysis

## Main config flags
```yaml
timing:
  mode: scene_rep   # or coslam or none
  warmup_frames: 5
  max_frames: 40
  disable_eval: True
```

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
Change `scene_rep` to `coslam` or `none` as needed.
