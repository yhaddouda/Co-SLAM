#!/bin/bash
set -euo pipefail
source /home/yh279050/miniforge3/etc/profile.d/conda.sh
conda activate coslam
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_after_v4_maxn_v1
exec python -B -u post_v4_follower.py >> follower.log 2>&1
