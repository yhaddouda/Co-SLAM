#!/usr/bin/env bash
set -euo pipefail
ncu_python="$1"
ncu_extension_cache="$2"
ncu_preload_std="$3"
shift 3
ncu_prefix="$(dirname -- "$(dirname -- "$ncu_python")")"
export PATH="$ncu_prefix/bin:${PATH-}"
export LD_LIBRARY_PATH="$ncu_prefix/lib:${LD_LIBRARY_PATH-}"
export TORCH_EXTENSIONS_DIR="$ncu_extension_cache"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
if [[ "$ncu_preload_std" == yes && -f "$ncu_prefix/lib/libstdc++.so.6" ]]; then
  # Preserve any Nsight injection preload supplied when this target starts.
  export LD_PRELOAD="$ncu_prefix/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
exec "$ncu_python" "$@"
