#!/usr/bin/env bash
set -euo pipefail
source /home/yh279050/miniforge3/etc/profile.d/conda.sh
conda activate coslam
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
cd /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1
exec python -u /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/run_manifest_queue.py --runtime /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/runtime --manifest /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/manifests/orin_coslam_diagnostic.json --output /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/results/diagnostic_maxn --power-mode MAXN > /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/run_records/diagnostic_maxn_queue.log 2>&1
