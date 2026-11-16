"""Read hardware/software context without changing power, clocks or CUDA flags."""
import argparse
import ctypes
import datetime
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from scripts.Validation.execution_benchmark_common import device_metadata


def read(path):
    try:
        return Path(path).read_text().strip().replace('\x00', '')
    except (OSError, UnicodeError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  platform=platform.platform(), architecture=platform.machine(),
                  board_model=read('/proc/device-tree/model'), **device_metadata())
    result['cpu'] = {str(p): read(p) for p in Path('/sys/devices/system/cpu').glob('cpu*/cpufreq/scaling_*freq')}
    result['devfreq'] = {str(p): read(p) for name in ('cur_freq', 'min_freq', 'max_freq', 'governor')
                         for p in Path('/sys/class/devfreq').glob('*/' + name)}
    result['thermal'] = {str(p.parent): dict(type=read(p.parent / 'type'), millidegrees_c=read(p))
                         for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
    executable = shutil.which('nvpmodel') or '/usr/sbin/nvpmodel'
    try:
        completed = subprocess.run([executable, '-q'], text=True, capture_output=True, timeout=5)
        result['power_mode_query'] = dict(returncode=completed.returncode,
                                          output=completed.stdout + completed.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        result['power_mode_query'] = str(error)
    try:
        cuda = ctypes.CDLL('libcudart.so')
        attributes = dict(integrated=18, clock_rate_khz=13, multiprocessors=16,
                          memory_clock_rate_khz=36, memory_bus_width_bits=37,
                          l2_cache_bytes=38, async_engines=40, managed_memory=83,
                          pageable_memory_access=88, concurrent_managed_access=89)
        result['cuda_attributes'] = {}
        for name, attribute in attributes.items():
            value = ctypes.c_int()
            status = cuda.cudaDeviceGetAttribute(ctypes.byref(value), attribute, 0)
            result['cuda_attributes'][name] = dict(value=value.value, status=status)
        flags, version = ctypes.c_uint(), ctypes.c_int()
        result['cuda_device_flags_status'] = cuda.cudaGetDeviceFlags(ctypes.byref(flags))
        result['cuda_device_flags'] = flags.value
        cuda.cudaRuntimeGetVersion(ctypes.byref(version))
        result['queried_runtime_version'] = version.value
    except OSError as error:
        result['cuda_attribute_query_error'] = str(error)
    try:
        driver = ctypes.CDLL('libcuda.so.1')
        value = ctypes.c_int()
        # CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
        # as defined by this installation's CUDA driver header.
        status = driver.cuDeviceGetAttribute(ctypes.byref(value), 102, 0)
        result['driver_virtual_memory_management'] = dict(value=value.value, status=status)
    except OSError as error:
        result['driver_attribute_query_error'] = str(error)
    result['scope'] = 'One metadata snapshot; not a sustained-clock, power or bandwidth measurement.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
