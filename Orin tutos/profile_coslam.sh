#!/usr/bin/env bash
set -euo pipefail

CONDA_PREFIX="/home/yh279050/miniforge3/envs/coslam"
PROJECT_ROOT="/home/yh279050/work/Co-SLAM"

# Make sure we run from the project root
cd "$PROJECT_ROOT"

# Use the same extension cache for user + sudo
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/.torch_extensions"

# Make env binaries (including ninja) visible
export PATH="$CONDA_PREFIX/bin:${PATH-}"

# Libraries & stdc++
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

exec "$CONDA_PREFIX/bin/python" -W ignore coslam.py "$@"


