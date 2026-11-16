"""Supervise six serial dense BA runs with fresh baselines and desktop telemetry."""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / 'results/ba_experiments/rtx4070_metadata'
OUT = ROOT / 'results/ba_experiments/rtx4070_dense_fast_full'
LOGS = ROOT / 'results/execution_experiments/logs/rtx4070/ba_dense_fast'
SCENES = ['office1', 'room1', 'office0']
VARIANTS = ['baseline', 'baseline_fast']
QUERY = ('timestamp,index,name,pstate,utilization.gpu,utilization.memory,memory.used,'
         'temperature.gpu,power.draw,power.limit,clocks.current.graphics,'
         'clocks.current.sm,clocks.current.memory,clocks_event_reasons.active')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    os.chdir(ROOT)
    assert Path(sys.prefix).name == 'coslam'
    assert os.environ['TORCH_CUDA_ARCH_LIST'] == '8.9'
    assert (META / 'PREFLIGHT.json').exists()
    assert not OUT.exists(), 'Never overwrite prior runs; select a new root.'
    files = [ROOT / 'coslam.py', ROOT / 'config.py', ROOT / 'utils.py']
    for directory in ('model', 'datasets', 'optimization', 'configs/Replica', 'scripts/Validation'):
        files += [p for p in (ROOT / directory).rglob('*') if p.suffix in ('.py', '.yaml', '.cu')]
    hashes = {}
    for p in sorted(set(files)):
        rel = p.relative_to(ROOT)
        dest = META / 'runtime_source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = p.read_bytes()
        dest.write_bytes(data)
        hashes[str(rel)] = hashlib.sha256(data).hexdigest()
    (META / 'runtime_source_hashes.json').write_text(json.dumps(hashes, indent=2))
    session = dict(started_utc=utc(), pid=os.getpid(), python=sys.executable,
                   output=str(OUT), frames=2000, seed=0, warmup_frames=20,
                   tracking='baseline', ba_variants=VARIANTS, stages=[], status='running',
                   order=[f'{scene}/{variant}' for variant in VARIANTS for scene in SCENES],
                   gpu_telemetry_query=QUERY, telemetry_interval_seconds=1,
                   order_difference='Baseline batch then dense-fast batch, like Orin; other Orin variants omitted.')
    path = META / 'SESSION.json'
    path.write_text(json.dumps(session, indent=2))
    try:
        with (META / 'gpu_telemetry.csv').open('x') as data, (LOGS / 'telemetry.log').open('x') as errors:
            monitor = subprocess.Popen(['nvidia-smi', f'--query-gpu={QUERY}', '--format=csv', '--loop=1'],
                                       stdout=data, stderr=errors)
            try:
                for variant in VARIANTS:
                    assert monitor.poll() is None, 'Telemetry stopped.'
                    stage = dict(variant=variant, started_utc=utc())
                    session['stages'].append(stage)
                    path.write_text(json.dumps(session, indent=2))
                    command = [sys.executable, '-u', 'scripts/Validation/run_replica_ba_suite.py',
                               '--scenes', *SCENES, '--variants', variant, '--frames', '0',
                               '--seeds', '0', '--output', str(OUT)]
                    print('START', utc(), variant, flush=True)
                    with (LOGS / f'{variant}_full.log').open('x') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    stage['finished_utc'] = utc()
                    path.write_text(json.dumps(session, indent=2))
                    print('DONE', utc(), variant, flush=True)
                assert monitor.poll() is None, 'Telemetry stopped.'
            finally:
                monitor.terminate()
                monitor.wait(timeout=10)
                session['telemetry_finished_utc'] = utc()
        for scene in SCENES:
            for variant in VARIANTS:
                record = json.loads((OUT / scene / f'{variant}_seed0/summary.json').read_text())
                assert record['frames'] == 2000 and record['warmup_frames'] == 20
                assert not record['diagnostic'] and record['tracking'] == 'baseline'
                assert record['finite_parameters_and_poses']
        for rel, value in hashes.items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == value, rel
        (OUT / 'SUITE_COMPLETE').write_text(json.dumps(dict(scenes=SCENES, variants=VARIANTS,
                                                          seeds=[0], frames=2000), indent=2))
        with (META / 'nvidia_smi_after.txt').open('x') as stream:
            subprocess.run(['nvidia-smi', '-q'], stdout=stream, check=True)
        session.update(status='benchmarks_complete', finished_utc=utc(), runtime_source_unchanged=True)
        print('BA_BENCHMARKS_COMPLETE', utc(), flush=True)
    except BaseException as error:
        session.update(status='failed', error=repr(error), failed_utc=utc())
        raise
    finally:
        path.write_text(json.dumps(session, indent=2))


if __name__ == '__main__':
    main()
