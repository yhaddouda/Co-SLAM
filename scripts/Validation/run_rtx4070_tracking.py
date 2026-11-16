"""Supervise the serial RTX tracking suite, its repeat, and desktop telemetry."""
import datetime
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / 'results/execution_experiments/rtx4070_metadata'
LOGS = ROOT / 'results/execution_experiments/logs/rtx4070'
FULL = ROOT / 'results/execution_experiments/coslam/rtx4070_full'
REPEAT = ROOT / 'results/execution_experiments/coslam/rtx4070_repeat_full'
QUERY = ('timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,'
         'temperature.gpu,power.draw,power.limit,clocks.current.graphics,'
         'clocks.current.sm,clocks.current.memory,clocks_event_reasons.active')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(script, args, logfile):
    command = [sys.executable, '-u', str(ROOT / 'scripts/Validation' / script), *args]
    print('START', utc(), logfile, flush=True)
    with (LOGS / logfile).open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('DONE', utc(), logfile, flush=True)


def main():
    os.chdir(ROOT)
    assert os.environ['TORCH_CUDA_ARCH_LIST'] == '8.9'
    assert Path(sys.prefix).name == 'coslam', sys.prefix
    assert (META / 'PREFLIGHT.json').exists()
    assert (ROOT / 'results/execution_experiments/coslam/rtx4070_smoke20/SUITE_COMPLETE').exists()
    assert not FULL.exists() and not REPEAT.exists(), 'Use new roots; never overwrite observations.'
    files = [ROOT / 'coslam.py', ROOT / 'config.py', ROOT / 'utils.py']
    for directory in ('model', 'datasets', 'optimization', 'configs/Replica', 'scripts/Validation'):
        files += [p for p in (ROOT / directory).rglob('*') if p.suffix in ('.py', '.yaml', '.cu')]
    hashes = {}
    for source in sorted(set(files)):
        relative = source.relative_to(ROOT)
        target = META / 'runtime_source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data)
        hashes[str(relative)] = hashlib.sha256(data).hexdigest()
    (META / 'runtime_source_hashes.json').write_text(json.dumps(hashes, indent=2))
    session = dict(started_utc=utc(), pid=os.getpid(), python=sys.executable,
                   cuda_arch=os.environ['TORCH_CUDA_ARCH_LIST'],
                   extension_directory=os.environ.get('TORCH_EXTENSIONS_DIR'),
                   gpu_telemetry_query=QUERY, telemetry_interval_seconds=1,
                   scope='Co-SLAM tracking only, dense baseline BA, no EC-SLAM runs',
                   full_root=str(FULL), repeat_root=str(REPEAT),
                   order=['office1/baseline', 'office1/baseline_fast', 'office1/cached_fast',
                          'room1/baseline_fast', 'room1/cached_fast', 'room1/baseline',
                          'office0/cached_fast', 'office0/baseline', 'office0/baseline_fast',
                          'repeat/office1/baseline'], status='running', stages=[])
    path = META / 'SESSION.json'
    path.write_text(json.dumps(session, indent=2))
    with (META / 'gpu_telemetry.csv').open('x') as telemetry, (LOGS / 'gpu_telemetry.log').open('x') as errors:
        monitor = subprocess.Popen(['nvidia-smi', f'--query-gpu={QUERY}', '--format=csv', '--loop=1'],
                                   stdout=telemetry, stderr=errors)
        try:
            for output, scenes, variants, logname in (
                (FULL, ['office1', 'room1', 'office0'], ['baseline', 'baseline_fast', 'cached_fast'], 'coslam_full.log'),
                (REPEAT, ['office1'], ['baseline'], 'coslam_repeat_full.log')):
                assert monitor.poll() is None, 'Telemetry process exited.'
                stage = dict(log=logname, started_utc=utc())
                session['stages'].append(stage)
                path.write_text(json.dumps(session, indent=2))
                run('run_replica_execution_suite.py',
                    ['--system', 'coslam', '--frames', '0', '--seeds', '0',
                     '--scenes', *scenes, '--variants', *variants, '--output', str(output)], logname)
                stage['finished_utc'] = utc()
                path.write_text(json.dumps(session, indent=2))
            assert monitor.poll() is None, 'Telemetry process exited.'
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
            session['telemetry_finished_utc'] = utc()
            path.write_text(json.dumps(session, indent=2))
    for output, logname in ((FULL, 'summarize_full.log'), (REPEAT, 'summarize_repeat.log')):
        run('summarize_execution_experiments.py', ['--root', str(output), '--plot'], logname)
    run('audit_execution_results.py', ['--require-complete', '--roots', str(FULL), str(REPEAT)], 'audit.log')
    for relative, expected in hashes.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
    with (META / 'nvidia_smi_after.txt').open('x') as stream:
        subprocess.run(['nvidia-smi', '-q'], stdout=stream, check=True)
    session.update(finished_utc=utc(), status='complete_and_audited', runtime_source_unchanged=True)
    path.write_text(json.dumps(session, indent=2))
    print('RTX_TRACKING_COMPLETE_AND_AUDITED', utc(), flush=True)


if __name__ == '__main__':
    main()
