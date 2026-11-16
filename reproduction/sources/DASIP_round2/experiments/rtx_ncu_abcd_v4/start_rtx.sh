#!/usr/bin/env bash
set -euo pipefail
ncu_kit_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/yh279050/miniforge3/envs/coslam/bin/python -B \
  "$ncu_kit_root/capture_queue.py" \
  --machine "$ncu_kit_root/machine_rtx.json" --start
