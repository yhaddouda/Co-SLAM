#!/usr/bin/env python3
"""Native-threaded EC complete-call timing continuation, preflight gated."""
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


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for key in ('after', 'runtime', 'manifest', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    base = Path(__file__).resolve().parents[1]
    spec = read(args.manifest)
    frozen = read(args.runtime/'RUNTIME_MANIFEST.json')
    identity = dict(runtime_manifest_sha256=hashlib.sha256((args.runtime/'RUNTIME_MANIFEST.json').read_bytes()).hexdigest(), manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    write(args.output/'campaign_identity.json', identity)
    write(args.output/'job_manifest.json', spec)
    status = dict(status='WAITING_FOR_CO_CAMPAIGN', completed=0, total=len(spec['jobs']), current=None, preflight='PENDING', after=str(args.after), actual_runs=[])
    def update():
        status['updated_paris'] = datetime.now(ZoneInfo('Europe/Paris')).isoformat()
        write(args.output/'campaign_status.json', status)
        text = '# EC-SLAM complete-call timing continuation\n\n'+f"Updated {status['updated_paris']}. **{status['status']}: {status['completed']}/{status['total']} full jobs.** Preflight: {status['preflight']}. Current: {status['current'] or '—'}.\n\n"
        text += 'Active worker call timing includes input preparation and pose finalization, excludes separately recorded handshake waits, preserves native threads and streams, and retains legacy CUDA iteration CSVs. Synchronized overlapping worker intervals are not additive GPU time or scene FPS. Instrumented-run ATE is retained separately from primary native-quality results. All eight Replica scenes, H16/H19, BASE/MC, three seeds.\n'
        (args.output/'LIVE_STATUS.md').write_text(text)
        # Do not contend with the still-running Co writer.
        if (args.after/'campaign_complete.json').is_file():
            prior=read(args.after/'campaign_complete.json')['completed']
            (base/'results/LIVE_STATUS.md').write_text('# Round 2 overnight timing stage\n\nPrimary training has finished; the engineering/reconstruction stages have exited (check their completion/failure records). Native EC complete-call timing is the final prepared stage.\n\n'+text.split('\n\n', 1)[1]+'\nSee `DASIP_round2/results/ec_timing_v1/` for logs and status, and DASIP_round2/RTX_REMAINING_NIGHT.md for the full agenda. No email sent automatically.\n')
    def validate_files():
        for name, expected in frozen['runtime_hashes'].items():
            if hashlib.sha256((args.runtime/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Frozen EC runtime changed: '+name)
        if identity['manifest_sha256'] != hashlib.sha256(args.manifest.read_bytes()).hexdigest():
            raise RuntimeError('EC experiment manifest changed')
    def run(job, diagnostic=False):
        validate_files()
        directory = args.output/job['id']
        command = [sys.executable, str(args.runtime/'run_ec_quality_round2.py'), '--scene', job['scene'], '--datadir', job['datadir'], '--condition', job['condition'], '--seed', str(job['seed']), '--hash-size', str(job['hash_size']), '--output', str(directory)]
        if diagnostic:
            command += ['--frames', '50']
        status['current'] = job['id']
        update()
        write(args.output/(job['id']+'.launch.json'), command)
        environment = os.environ.copy()
        environment['PYTHONHASHSEED'] = str(job['seed'])
        begin = time.time()
        with (args.output/(job['id']+'.log')).open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            try:
                while child.poll() is None:
                    if time.time()-begin > (1200 if diagnostic else 3600):
                        import signal
                        os.killpg(child.pid, signal.SIGTERM)
                        raise RuntimeError('EC job timeout; partial outputs retained: '+job['id'])
                    update()
                    time.sleep(15)
                if child.returncode:
                    raise RuntimeError('EC job failed: '+job['id'])
            except BaseException:
                if child.poll() is None:
                    import signal
                    os.killpg(child.pid, signal.SIGTERM)
                raise
        summary = read(directory/'summary.json')
        assert summary['finite_parameters_and_poses']
        assert summary['seed']==job['seed'] and summary['condition']==job['condition'] and summary['hash_size']==job['hash_size']
        assert summary['frames'] == (50 if diagnostic else summary['available_frames'])
        assert (directory/'calls.csv').is_file() and (directory/'iterations.csv').is_file()
        assert summary['purpose']=='native_complete_call_profile' and summary['warmup_frames']==20
        assert summary['tracking']['count']>0 and summary['bundle_adjustment']['count']>0
        if job['condition']=='MC':
            assert len(summary['permutation_check_max_abs']) == 1
        record = dict(id=job['id'], elapsed_seconds=time.time()-begin, diagnostic=diagnostic)
        write(args.output/(job['id']+'.done.json'), record)
        status['actual_runs'].append(record)
        if not diagnostic:
            status['completed'] += 1
        update()
    try:
        update()
        while not (args.after/'campaign_complete.json').is_file():
            if (args.after/'campaign_failure.json').exists():
                raise RuntimeError('Prior Co campaign failed; preserved, no overlap launched')
            time.sleep(15)
            update()
        from report_support_reference import gpu_idle
        while gpu_idle()['idle'] is not True:
            status['status']='WAITING_FOR_IDLE_GPU'
            update()
            time.sleep(15)
        with open('/tmp/dasip_round2_gpu0.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            status['status']='PREFLIGHT'
            for capacity in (16, 19):
                for condition in ('BASE', 'MC'):
                    job = dict(id=f'DIAGNOSTIC_office0_{condition}_H{capacity}_seed0', scene='office0', condition=condition, hash_size=capacity, seed=0, datadir=spec['jobs'][0]['datadir'])
                    run(job, diagnostic=True)
            status['preflight']='PASSED_4_DIAGNOSTICS'
            write(args.output/'preflight_passed.json', status.copy())
            status['status']='RUNNING'
            for job in spec['jobs']:
                run(job)
            status['status']='COMPLETE'
            status['current']=None
            update()
            write(args.output/'campaign_complete.json', status)
    except BaseException as error:
        status['status']='STOPPED: '+str(error)
        update()
        write(args.output/'campaign_failure.json', status)
        raise


if __name__=='__main__':
    main()
