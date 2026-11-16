"""Validate and trace the optional tracking depth cache before full benchmarks."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/execution_experiments/gpu_checks'


def main():
    if not (OUTPUT/'CHECKS_COMPLETE').exists():
        raise RuntimeError('Earlier CUDA/profiler jobs must finish before this GPU job')
    with (OUTPUT/'coslam_depth_cache.log').open('w') as stream:
        subprocess.run(['conda','run','--no-capture-output','-n','coslam','python','-u',
                        str(ROOT/'scripts/Validation/check_tracking_depth_cache_gpu.py'),
                        '--output',str(OUTPUT/'coslam_depth_cache.json')],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    nsys = shutil.which('nsys') or '/opt/nvidia/nsight-systems/2024.5.4/target-linux-tegra-armv8/nsys'
    label = 'coslam_cached_fast_trace'
    prefix = OUTPUT/label
    status = dict(label=label)
    try:
        with (OUTPUT/(label+'.log')).open('w') as stream:
            subprocess.run([nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none',
                            '--force-overwrite=false','-o',str(prefix),
                            'conda','run','--no-capture-output','-n','coslam','python','-u',
                            str(ROOT/'scripts/Validation/trace_replica_execution.py'),
                            '--system','coslam','--scene','office1','--variant','cached_fast',
                            '--frames','30','--warmup','1','--output',str(OUTPUT/'trace_runs/coslam/cached_fast')],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        status['success'] = True
        with (OUTPUT/(label+'_analysis.log')).open('w') as stream:
            subprocess.run([nsys,'export','--type=sqlite','--force-overwrite=false','--output',str(prefix)+'.sqlite',str(prefix)+'.nsys-rep'],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,str(ROOT/'scripts/Validation/analyze_execution_trace.py'),
                            '--sqlite',str(prefix)+'.sqlite','--output',str(prefix)+'_analysis'],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
        status['analysis_success'] = True
    except subprocess.CalledProcessError as error:
        status['error'] = str(error)
    (OUTPUT/'cache_trace_status.json').write_text(json.dumps(status,indent=2)+'\n')
    (OUTPUT/'CACHE_CHECKS_COMPLETE').write_text('Dense depth-cache CUDA equivalence passed; trace availability in cache_trace_status.json.\n')
    print('CACHE_CHECKS_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
