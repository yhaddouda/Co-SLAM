#!/usr/bin/env python3
"""Separate serial profiling block, optionally guarded by explicit local gates."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def gates_ready(spec):
    for failure in spec.get('failure_files', []):
        if Path(failure).exists():
            raise RuntimeError('Upstream failure preserved: '+failure)
    for condition in spec['conditions']:
        path = Path(condition['file'])
        if not path.exists():
            return False
        try:
            record = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        for check in condition['checks']:
            value = record
            for key in check['pointer'].strip('/').split('/'):
                value = value[int(key)] if isinstance(value, list) else value[key]
            if value != check['equals']:
                return False
    for process in spec.get('wait_processes', []):
        path = Path(f"/proc/{process['pid']}/stat")
        if path.exists():
            try:
                fields = path.read_text().rsplit(')', 1)[1].split()
            except FileNotFoundError:
                continue
            if fields[0] != 'Z' and int(fields[19]) == process['start_ticks']:
                return False
    return True


def gpu_idle(command=None):
    if command:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)['idle'] is True
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return not result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', type=Path, help='Explicit locally bound normal-series/export JSON checks, not an ETA')
    args = parser.parse_args()
    runtime, output, manifest_path = args.runtime.resolve(), args.output.resolve(), args.manifest.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['jobs'] and all(j['condition']=='BASE' for j in manifest['jobs'])
    if manifest['platform'].startswith('ORIN'):
        assert manifest.get('power_mode'), 'Resolve exact nvpmodel name locally; never guess/change it'
        assert args.gate, 'Orin continuation requires explicit upstream normal-completion/export gates'
        assert manifest.get('idle_check_command'), 'Bind the existing validated Orin idle checker; do not assume nvidia-smi exists'
    identity = dict(manifest_sha256=sha(manifest_path), runtime_manifest_sha256=sha(runtime/'RUNTIME_MANIFEST.json'))
    save(output/'identity.json', identity)
    save(output/'manifest.json', manifest)
    gate = json.loads(args.gate.read_text()) if args.gate else None
    if gate:
        assert gate['conditions'] and all(c['checks'] for c in gate['conditions'])
        save(output/'gate.json', gate)
    state = dict(status='WAITING_FOR_NORMAL_UPSTREAM_COMPLETION' if gate else 'WAITING_FOR_EXCLUSIVE_GPU',
                 platform=manifest['platform'], current=None, diagnostic_complete=0, completed=0,
                 total=len(manifest['jobs']), failed=0, automatic_email=False)
    def snapshot():
        state['updated_paris'] = datetime.now(ZoneInfo('Europe/Paris')).isoformat()
        save(output/'status.json', state)
        if manifest.get('local_live_status'):
            Path(manifest['local_live_status']).write_text('# Separate Co-SLAM component profiling\n\n'+json.dumps(state, indent=2)+'\n\nFull online BASE passes; office0 H16/H19 and fr1_desk H16. Both decompositions are collected together on predetermined calls across three post-warmup thirds. Benchmark quality/reconstruction remains complete; these are new profiling measurements. No new pruning method or mode change.\n')
    def run(job, diagnostic):
        name = job['id'] + ('_diagnostic' if diagnostic else '')
        directory = output/name
        argv = [sys.executable, str(runtime/'run_component_profile.py'), '--workload', job['workload'],
                '--seed', str(job['seed']), '--datadir', job['datadir'], '--output', str(directory)]
        if manifest.get('power_mode'):
            argv += ['--power-mode', manifest['power_mode']]
        if diagnostic:
            argv += ['--frames', '50', '--validate-state']
        assert sha(manifest_path)==identity['manifest_sha256']
        assert sha(runtime/'RUNTIME_MANIFEST.json')==identity['runtime_manifest_sha256']
        state.update(current=name, status='DIAGNOSTIC_FIXED_STATE_AND_50_FRAMES' if diagnostic else 'RUNNING_FULL_PROFILE')
        save(output/(name+'.launch.json'), argv)
        snapshot()
        start = time.perf_counter()
        with (output/(name+'.log')).open('w') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            state['child_pid'] = child.pid
            while child.poll() is None:
                snapshot()
                time.sleep(5)
        state.pop('child_pid', None)
        if child.returncode:
            raise RuntimeError(f'{name} failed ({child.returncode}); see its preserved log')
        summary = json.loads((directory/'summary.json').read_text())
        assert summary['finite_parameters_and_poses'] and summary['diagnostic']==diagnostic
        if diagnostic:
            assert json.loads((directory/'fixed_state_validation/PASSED.json').read_text())['status']=='PASSED'
        else:
            assert summary['full_sequence']
            assert summary['selected_calls']=={'tracking':300, 'bundle_adjustment':60}
        save(output/(name+'.done.json'), dict(id=name, elapsed_seconds=time.perf_counter()-start, summary_sha256=sha(directory/'summary.json')))
        state['diagnostic_complete' if diagnostic else 'completed'] += 1
        snapshot()
    try:
        snapshot()
        while gate and not gates_ready(gate):
            time.sleep(30); snapshot()
        with open('/tmp/dasip_round2_gpu0.lock', 'a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    state['status']='WAITING_FOR_EXCLUSIVE_GPU'; snapshot(); time.sleep(15)
            # Fresh checks after acquiring ownership; an idle job gap is not an upstream gate.
            for _ in range(3):
                assert gpu_idle(manifest.get('idle_check_command')), 'GPU device has a competing user'
                time.sleep(2)
            seen = set()
            for job in manifest['jobs']:
                if job['workload'] in seen:
                    continue
                run(job, True)
                seen.add(job['workload'])
            state['status']='ALL_WORKLOAD_PREFLIGHTS_PASSED'; snapshot()
            for job in manifest['jobs']:
                run(job, False)
            state.update(status='COMPLETE', current=None)
            snapshot()
            save(output/'series_complete.json', dict(**state, gpu_idle=gpu_idle(manifest.get('idle_check_command'))))
        print('COMPONENT_PROFILE_SERIES_COMPLETE', flush=True)
    except BaseException as error:
        state.update(status='STOPPED', error=str(error), failed=1)
        snapshot()
        save(output/'series_failure.json', state)
        raise


if __name__ == '__main__':
    main()
