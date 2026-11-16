#!/usr/bin/env python3
"""Independent file-status monitor; optional local desktop alert, no GPU work."""
import argparse
from datetime import datetime, timezone
from email.message import EmailMessage
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def queue_running(lock_path):
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--status', type=Path)
    parser.add_argument('--poll', type=float, default=15)
    parser.add_argument('--lock', type=Path, default=Path('/tmp/dasip_round2_gpu0.lock'))
    parser.add_argument('--desktop', action='store_true', help='Attempt one local desktop notification on completion/failure')
    parser.add_argument('--email-to', help='Prepare a completion/failure email in the local outbox; does not send')
    args = parser.parse_args()
    if args.poll < 1:
        parser.error('Poll interval must be at least one second')
    queue = args.queue.resolve()
    status = args.status.resolve() if args.status else queue.parent / 'LIVE_STATUS.md'
    document = json.loads((queue / 'job_manifest.json').read_text())
    jobs = document['jobs']
    while True:
        done = [j['id'] for j in jobs if (queue / (j['id'] + '.done.json')).is_file()]
        active = [j['id'] for j in jobs if (queue / (j['id'] + '.launch.json')).is_file() and j['id'] not in done]
        state = 'RUNNING'
        if (queue / 'queue_complete.json').is_file():
            state = 'COMPLETE' if len(done) == len(jobs) else 'FAILED: inconsistent completion record'
        elif (queue / 'progress.json').is_file():
            progress = json.loads((queue / 'progress.json').read_text())
            if any(x.get('returncode', 0) for x in progress['jobs']):
                state = 'FAILED'
        if state == 'RUNNING' and not queue_running(args.lock):
            # Recheck after observing the lock release to avoid a completion-write race.
            if (queue / 'queue_complete.json').is_file() and len(done) == len(jobs):
                state = 'COMPLETE'
            else:
                state = 'STOPPED: inspect queue log and incomplete job'
        now = datetime.now(timezone.utc).isoformat()
        message = f'{state}: {len(done)}/{len(jobs)} jobs completed.'
        text = '# Round 2 experiment status\n\n'
        text += f'Updated: {now} (UTC).\n\n**{message}**\n\n'
        text += 'Current/incomplete job: ' + (', '.join(active) or 'none') + '.\n\n'
        text += f'Queue directory: `{queue}`.\n\n'
        text += 'Completed outputs are raw experiment results. Scientific acceptance, three-seed comparisons and independent mesh scoring require analysis.\n\n'
        text += 'This monitor runs independently of the chat and never changes algorithm settings, power mode or GPU jobs. It does not wake the Codex conversation.\n'
        if args.email_to:
            text += '\nEmail recipient recorded. Automatic email delivery is not configured; completion/failure will prepare an outbox message for later delivery.\n'
        atomic_write(status, text)
        if state != 'RUNNING':
            record = dict(state=state, completed=len(done), total=len(jobs), active=active, updated_utc=now,
                          status_file=str(status), desktop_requested=args.desktop)
            if args.email_to:
                email = EmailMessage()
                email['To'] = args.email_to
                email['Subject'] = 'DASIP Round 2 experiments: ' + state
                email.set_content(message + '\n\nCurrent/incomplete job: ' + (', '.join(active) or 'none')
                                  + '\nQueue: ' + str(queue) + '\nUpdated: ' + now
                                  + '\n\nRaw experiment outputs are ready for analysis when COMPLETE. '
                                  'Three-seed ATE/speed comparisons and independent mesh scoring must still be checked. '
                                  'No quality/speed threshold is implied by job completion.\n')
                pending = queue / 'PENDING_NOTIFICATION.eml'
                atomic_write(pending, email.as_string())
                record.update(email_recipient=args.email_to, email_status='prepared, not sent; no unattended transport configured',
                              email_outbox=str(pending))
            if args.desktop:
                executable = shutil.which('notify-send')
                if executable:
                    try:
                        result = subprocess.run([executable, '--app-name=DASIP Round 2', '--expire-time=0',
                                                 'DASIP experiments: ' + state, message + ' See ' + str(status)],
                                                capture_output=True, text=True, timeout=10)
                        record['desktop_returncode'] = result.returncode
                        record['desktop_error'] = result.stderr.strip()
                    except (OSError, subprocess.TimeoutExpired) as error:
                        record['desktop_error'] = str(error)
                else:
                    record['desktop_error'] = 'notify-send unavailable'
            atomic_write(queue / 'MONITOR_EVENT.json', json.dumps(record, indent=2) + '\n')
            print(json.dumps(record), flush=True)
            return
        time.sleep(args.poll)


if __name__ == '__main__':
    main()
