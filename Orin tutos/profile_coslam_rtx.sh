#!/usr/bin/env bash
# profile_coslam.sh  (RTX 4070 / Ada)
set -euo pipefail

# --- edit if your paths differ ---
CONDA_ENV="/home/yh279050/miniforge3/envs/coslam"
PROJECT="/home/yh279050/work/Co-SLAM"
# ---------------------------------

PYTHON="$CONDA_ENV/bin/python"

# Use your conda env and project code, target Ada (8.9). Keep a local build cache.
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9+PTX"
export TORCH_EXTENSIONS_DIR="$PROJECT/.torch_extensions_ada"
# If root doesn't have the Python 'ninja' package, set to 0 (matches your Orin workaround)
export USE_NINJA="${USE_NINJA:-1}"

mkdir -p "$TORCH_EXTENSIONS_DIR"

# Forward all args to coslam.py (e.g., --config configs/Replica/office0.yaml)
exec "$PYTHON" "$PROJECT/coslam.py" "$@"
