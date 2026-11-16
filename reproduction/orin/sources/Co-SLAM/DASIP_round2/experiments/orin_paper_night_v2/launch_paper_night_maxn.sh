#!/usr/bin/env bash
set -euo pipefail
source /home/yh279050/miniforge3/etc/profile.d/conda.sh
conda activate coslam
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_paper_night_v2
exec python -u paper_night_parent.py > paper_night_parent.log 2>&1
