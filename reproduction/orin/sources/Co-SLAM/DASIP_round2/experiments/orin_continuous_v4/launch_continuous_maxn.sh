#!/bin/bash
set -euo pipefail
source /home/yh279050/miniforge3/etc/profile.d/conda.sh
conda activate coslam
cd /home/yh279050/work/Co-SLAM/DASIP_round2/experiments/orin_continuous_v4
exec python -B -u continuous_parent.py >> continuous_parent.log 2>&1
