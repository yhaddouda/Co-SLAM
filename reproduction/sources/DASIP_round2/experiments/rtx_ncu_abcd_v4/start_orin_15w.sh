#!/usr/bin/env bash
set -euo pipefail
ncu_kit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(id -u)" != 0 ]]; then
  echo "This Jetson's Nsight Compute requires root. Run this script with sudo in your own terminal; do not send passwords to chat." >&2
  exit 2
fi
exec /home/yh279050/miniforge3/envs/coslam/bin/python -B "$ncu_kit_dir/capture_queue.py" --machine "$ncu_kit_dir/machine_orin.json" --mode 15W --start "$@"
