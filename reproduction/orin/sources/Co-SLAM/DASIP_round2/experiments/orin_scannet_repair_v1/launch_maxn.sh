#!/bin/bash
set -euo pipefail
cd /home/yh279050/work/Co-SLAM
source /home/yh279050/miniforge3/etc/profile.d/conda.sh
conda activate coslam
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
exec python -u DASIP_round2/experiments/orin_scannet_repair_v1/repair_and_run.py --resume-preparation >> DASIP_round2/experiments/orin_scannet_repair_v1/parent.log 2>&1
