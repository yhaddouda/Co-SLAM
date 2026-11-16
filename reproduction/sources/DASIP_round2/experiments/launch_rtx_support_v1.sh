#!/usr/bin/env bash
set -euo pipefail
exec /home/yh279050/miniforge3/bin/conda run --no-capture-output -n coslam \
  python /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/run_manifest_queue.py \
  --manifest /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/manifests/rtx_support_core.json \
  --runtime /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/coslam_reference_runtime_v1 \
  --output /home/yh279050/work/Co-SLAM/DASIP_round2/results/support_reference_v1 \
  > /home/yh279050/work/Co-SLAM/DASIP_round2/results/support_reference_v1_queue/queue.log 2>&1
