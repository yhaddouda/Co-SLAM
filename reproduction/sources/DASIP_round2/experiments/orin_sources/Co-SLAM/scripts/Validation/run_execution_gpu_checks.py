"""Serialize CUDA equivalence checks, hardware capture and optional short traces."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for', type=Path)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.wait_for:
        while not args.wait_for.exists():
            print(f'WAIT {args.wait_for}', flush=True)
            time.sleep(30)
    def run(label, environment, script, arguments, prefix=()):
        command = [*prefix, 'conda', 'run', '--no-capture-output', '-n', environment,
                   'python', '-u', str(ROOT / 'scripts/Validation' / script), *arguments]
        print('START ' + label, flush=True)
        with (args.output / (label + '.log')).open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print('DONE ' + label, flush=True)
    run('hardware', 'coslam', 'capture_execution_hardware.py', ['--output', str(args.output / 'hardware.json')])
    for system in ('coslam', 'ec_slam'):
        run(system + '_equivalence', system, 'check_execution_gpu.py',
            ['--system', system, '--output', str(args.output / (system + '_equivalence.json'))])
    trace_status = []
    if args.trace:
        nsys = shutil.which('nsys') or '/opt/nvidia/nsight-systems/2024.5.4/target-linux-tegra-armv8/nsys'
        for system, variants, script in (
            ('coslam', ('baseline', 'baseline_fast'), 'trace_replica_execution.py'),
            ('ec_slam', ('baseline', 'both_fast'), 'trace_replica_execution.py'),
        ):
            for variant in variants:
                label = system + '_' + variant + '_trace'
                trace_path = args.output / label
                prefix = [nsys, 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                          '--force-overwrite=false', '-o', str(trace_path)]
                try:
                    run(label, system, script,
                        ['--system',system,'--scene','office1','--variant',variant,'--frames','30','--warmup','1',
                         '--output',str(args.output / 'trace_runs' / system / variant)], prefix)
                    status = dict(label=label, success=True)
                    trace_status.append(status)
                    # Export/analyze while the GPU suite is still serialized,
                    # before full timing runs start on the shared Orin CPU.
                    sqlite_path = trace_path.with_suffix('.sqlite')
                    try:
                        with (args.output / (label + '_analysis.log')).open('w') as stream:
                            subprocess.run([nsys, 'export', '--type=sqlite', '--force-overwrite=false',
                                            '--output', str(sqlite_path), str(trace_path) + '.nsys-rep'],
                                           stdout=stream, stderr=subprocess.STDOUT, check=True)
                            subprocess.run([sys.executable, str(ROOT / 'scripts/Validation/analyze_execution_trace.py'),
                                            '--sqlite', str(sqlite_path), '--output', str(args.output / (label + '_analysis'))],
                                           stdout=stream, stderr=subprocess.STDOUT, check=True)
                        status['analysis_success'] = True
                    except subprocess.CalledProcessError as error:
                        status.update(analysis_success=False, analysis_error=str(error))
                except subprocess.CalledProcessError as error:
                    # A trace-tool failure must not prevent the unprofiled full
                    # quality/time experiments; retain its exact diagnostic log.
                    trace_status.append(dict(label=label, success=False, error=str(error)))
    (args.output / 'trace_status.json').write_text(json.dumps(trace_status, indent=2))
    (args.output / 'CHECKS_COMPLETE').write_text('CUDA checks passed; see trace_status.json for trace availability.\n')
    print('CHECKS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
