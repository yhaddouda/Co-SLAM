#!/bin/bash
set -euo pipefail
source /home/yh279050/miniforge3/etc/profile.d/conda.sh
conda activate coslam
export PYTHONDONTWRITEBYTECODE=1
cd /home/yh279050/work/Co-SLAM
exec python -B -u /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_ec_component_profile_runtime_v2_single_seed/continuation.py >> /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_ec_component_profile_runtime_v2_single_seed/follower.log 2>&1
