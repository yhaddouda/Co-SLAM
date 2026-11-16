#!/usr/bin/env bash
set -euo pipefail
source /home/yh279050/miniforge3/etc/profile.d/conda.sh
conda activate coslam
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
cd /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1
set +e
python -u /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/run_manifest_queue.py --runtime /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/runtime --manifest /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/manifests/orin_coslam_first_full.json --output /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/results/full_maxn --power-mode MAXN > /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/run_records/full_maxn_queue.log 2>&1
queue_status=$?
set -e
python -u /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/run_records/validate_and_package_maxn.py --queue-exit-code "$queue_status" > /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_coslam_runtime_v1/run_records/full_maxn_finalization.log 2>&1
