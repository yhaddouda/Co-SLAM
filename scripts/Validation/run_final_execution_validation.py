"""Finish numerical audits after all timed jobs and the baseline repeat finish."""
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT/'results/execution_experiments/gpu_checks/final_validation'


def main():
    marker = ROOT/'results/execution_experiments/coslam/repeat_full/SUITE_COMPLETE'
    while not marker.exists():
        print('WAIT '+str(marker),flush=True)
        time.sleep(30)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for system in ('coslam','ec_slam'):
        with (OUTPUT/(system+'.log')).open('w') as stream:
            subprocess.run(['conda','run','--no-capture-output','-n',system,'python','-u',
                            str(ROOT/'scripts/Validation/check_execution_gpu.py'),
                            '--system',system,'--output',str(OUTPUT/(system+'.json'))],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)
    with (OUTPUT/'depth_cache.log').open('w') as stream:
        subprocess.run(['conda','run','--no-capture-output','-n','coslam','python','-u',
                        str(ROOT/'scripts/Validation/check_tracking_depth_cache_gpu.py'),
                        '--output',str(OUTPUT/'depth_cache.json')],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    (OUTPUT/'VALIDATION_COMPLETE').write_text('Frozen CUDA pose gradients checked separately from map gradients.\n')
    print('FINAL_VALIDATION_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
