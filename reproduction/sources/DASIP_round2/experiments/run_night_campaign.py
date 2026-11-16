#!/usr/bin/env python3
"""Bounded serial continuation: existing DP study, native MC, Replica, capacity."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from report_support_reference import gpu_idle, read_json, running


def atomic(path, text):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--after-campaign',type=Path,help='Wait for this prior campaign to finish before taking over status and GPU work')
    args = parser.parse_args()
    if args.after_campaign:
        args.output.mkdir(parents=True,exist_ok=True)
        while not (args.after_campaign/'campaign_complete.json').is_file():
            atomic(args.output/'continuation_wait_status.json',json.dumps(dict(status='WAITING_FOR_PRIOR_CAMPAIGN',after=str(args.after_campaign.resolve())),indent=2)+'\n')
            if (args.after_campaign/'campaign_failure.json').exists():
                raise RuntimeError('Prior campaign failed; continuation not launched')
            time.sleep(15)
    spec = read_json(args.campaign)
    base = args.campaign.resolve().parents[2]
    phases = spec['phases']
    runtime = base / spec['runtime']
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = {p['id']: read_json(base / p['manifest']) for p in phases}
    frozen = dict(runtime_manifest_sha256=hashlib.sha256((runtime / 'RUNTIME_MANIFEST.json').read_bytes()).hexdigest(),
                  manifests={p['id']: hashlib.sha256((base / p['manifest']).read_bytes()).hexdigest() for p in phases})
    identity = args.output / 'campaign_identity.json'
    if identity.exists() and read_json(identity) != frozen:
        raise RuntimeError('Campaign inputs changed')
    identity.write_text(json.dumps(frozen, indent=2) + '\n')
    state = 'RUNNING'
    active_phase = None
    def snapshot():
        records = []
        for phase in phases:
            queue = base / phase['output']
            jobs = manifests[phase['id']]['jobs']
            done = [j['id'] for j in jobs if (queue / (j['id'] + '.done.json')).is_file()]
            active = [j['id'] for j in jobs if (queue / (j['id'] + '.launch.json')).is_file() and j['id'] not in done]
            complete = (queue / 'queue_complete.json').is_file() and len(done) == len(jobs)
            records.append(dict(id=phase['id'], label=phase['label'], completed=len(done), total=len(jobs),
                                status='COMPLETE' if complete else ('RUNNING/WAITING' if phase['id'] == active_phase else 'QUEUED'),
                                current_jobs=active, output=str(queue)))
        now = datetime.now(ZoneInfo('Europe/Paris')).isoformat()
        record = dict(updated_paris=now, status=state, active_phase=active_phase,
                      completed=sum(p['completed'] for p in records), total=sum(p['total'] for p in records), phases=records,
                      automatic_email_delivery=False)
        atomic(args.output / 'campaign_status.json', json.dumps(record, indent=2) + '\n')
        text = '# Round 2 paper-run status\n\n'
        text += f"Updated {now} (Europe/Paris). **{state}: {record['completed']}/{record['total']} full-sequence runs complete.**\n\n"
        text += '| Phase | Completed | Status | Current/incomplete job |\n|---|---:|---|---|\n'
        for r in records:
            text += f"| {r['id']}: {r['label']} | {r['completed']}/{r['total']} | {r['status']} | {', '.join(r['current_jobs']) or '—'} |\n"
        text += '\nR1 validates the pruning candidate; R2–R4 measure the independent native-reference Morton contribution. These are full paper-candidate sequences, not 50-frame diagnostics. Completed runs are evidence, not automatic acceptance. Reconstruction metrics, EC-SLAM and engineering ablations are not completed by this campaign.\n\n'
        text += 'The serial queues and reports continue without an active chat on this powered-on host. No adaptive new-variant search, power-mode change or automatic email is performed. Email is reserved for the end of the major campaign after a freshly verified idle GPU and a decision summary. See `DASIP_round2/RUN_AGENDA.md`.\n'
        atomic(base / 'results/LIVE_STATUS.md', text)
        return record
    try:
        prior_queues = []
        for phase in phases:
            active_phase = phase['id']
            queue = base / phase['output']
            snapshot()
            if phase.get('existing'):
                while not (queue / 'queue_complete.json').is_file():
                    if not running():
                        raise RuntimeError('Existing queue stopped before completion: ' + str(queue))
                    snapshot()
                    time.sleep(15)
            elif not (queue / 'queue_complete.json').is_file():
                if queue.exists() and any(queue.iterdir()):
                    raise RuntimeError('Partial queue preserved; diagnose before restarting: ' + str(queue))
                while gpu_idle()['idle'] is not True or running():
                    state = 'WAITING FOR IDLE GPU'
                    snapshot()
                    time.sleep(15)
                state = 'RUNNING'
                if hashlib.sha256((runtime / 'RUNTIME_MANIFEST.json').read_bytes()).hexdigest() != frozen['runtime_manifest_sha256']:
                    raise RuntimeError('Runtime manifest changed')
                manifest = base / phase['manifest']
                if hashlib.sha256(manifest.read_bytes()).hexdigest() != frozen['manifests'][phase['id']]:
                    raise RuntimeError('Manifest changed')
                argv = [sys.executable, str(base / 'experiments/run_manifest_queue.py'), '--runtime', str(runtime),
                        '--manifest', str(manifest), '--output', str(queue)]
                with (args.output / (phase['id'] + '.log')).open('w') as log:
                    child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        snapshot()
                        time.sleep(15)
                    if child.returncode:
                        raise RuntimeError('Queue failed; inspect ' + phase['id'] + '.log')
            expected = len(manifests[phase['id']]['jobs'])
            assert sum((queue / (j['id'] + '.done.json')).is_file() for j in manifests[phase['id']]['jobs']) == expected
            saved_identity = read_json(queue / 'queue_identity.json')
            assert saved_identity['runtime_manifest_sha256'] == frozen['runtime_manifest_sha256']
            assert saved_identity['manifest_sha256'] == frozen['manifests'][phase['id']]
            prior_queues.append(queue)
            if phase['id'] != 'R1':
                command = [sys.executable, str(base / 'experiments/report_morton_reference.py')]
                for path in prior_queues:
                    command += ['--queue', str(path)]
                command += ['--output', str(base / phase['report'])]
                with (args.output / (phase['id'] + '_analysis.log')).open('w') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            snapshot()
        state = 'COMPLETE'
        active_phase = None
        record = snapshot()
        record['gpu_snapshot'] = gpu_idle()
        atomic(args.output / 'campaign_complete.json', json.dumps(record, indent=2) + '\n')
        print('CAMPAIGN_COMPLETE', record['completed'], flush=True)
    except Exception as error:
        state = 'STOPPED: ' + str(error)
        snapshot()
        atomic(args.output / 'campaign_failure.json', json.dumps(dict(error=str(error), phase=active_phase), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
