#!/usr/bin/env bash
set -euo pipefail

# ---- Fixed ground-truth mesh path ----
GT_MESH="$HOME/work/Co-SLAM/neural_slam_eval/data/Replica/Replica/office0/gt_mesh_cull_virt_cams.ply"

# ---- Parent directory with runs ----
OFFICE_DIR="../output/Replica/office0"

# Loop through all subfolders (13, 13_half, 14, 14_half, 16_base, 16_half, ..., 20, 20-half)
for run_dir in "$OFFICE_DIR"/*/; do
    run_name="$(basename "$run_dir")"
    rec_mesh="$run_dir/mesh_track1999_cull_virt_cams.ply"

    if [[ -f "$rec_mesh" ]]; then
        echo "=== Running eval for $run_name ==="
        xvfb-run -a python eval_recon_headless.py \
            --rec_mesh "$rec_mesh" \
            --gt_mesh "$GT_MESH" \
            --dataset_type Replica -2d -3d 
    else
        echo ">>> Skipping $run_name (no mesh found)"
    fi
done
