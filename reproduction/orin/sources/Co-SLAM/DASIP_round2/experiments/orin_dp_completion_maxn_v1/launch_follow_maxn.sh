#!/bin/bash
set -euo pipefail
cd /home/yh279050/work/Co-SLAM
source /home/yh279050/miniforge3/etc/profile.d/conda.sh
conda activate coslam
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
exec python -B -u DASIP_round2/experiments/orin_dp_completion_maxn_v1/follower.py >> DASIP_round2/experiments/orin_dp_completion_maxn_v1/follower.log 2>&1
