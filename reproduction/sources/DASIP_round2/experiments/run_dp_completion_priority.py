#!/usr/bin/env python3
"""Run missing DP scenes in an explicitly delegated, exclusive EC queue gap."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from checked_reconstruction import save_json,sha
from report_support_reference import gpu_idle,audit_job


def main():
    p=argparse.ArgumentParser();p.add_argument('--queue-pid',type=int,required=True);args=p.parse_args()
    base=Path(__file__).resolve().parents[1];owner=args.queue_pid
    output=base/'results/dp_completion_priority_v1';output.mkdir(exist_ok=False)
    manifest=base/'experiments/manifests/rtx_dp_missing_scenes.json'
    runtime=base/'experiments/coslam_reference_runtime_v1'
    spec=json.loads(manifest.read_text());jobs=spec['jobs']
    identity=dict(manifest_sha256=sha(manifest),runtime_manifest_sha256=sha(runtime/'RUNTIME_MANIFEST.json'),power_mode=None)
    save_json(output/'queue_identity.json',identity);save_json(output/'job_manifest.json',spec)
    state=dict(status='DRAINING_CURRENT_EC_RUN',completed=0,total=len(jobs),failed=0,current=None,queue_owner_pid=owner,failures=[],automatic_email=False)
    def snapshot():
        state['updated_paris']=datetime.now(ZoneInfo('Europe/Paris')).isoformat();save_json(output/'campaign_status.json',state)
        (base/'results/LIVE_STATUS.md').write_text('# Priority: complete conservative DP Tables 3 and 4\n\n'+f"Updated {state['updated_paris']}. **{state['status']}: {state['completed']}/{len(jobs)} missing-scene DP runs complete; {state['failed']} failed.** Current: {state['current'] or '—'}.\n\n24 configured scenes, three paired seeds; existing matching BASE runs are reused. The same frozen DP implementation is retained. Dataset means are for author review; previous thresholds are reference checks. After this priority series, the unchanged EC quality / engineering / reconstruction / timing queue resumes automatically.\n\nSee DASIP_round2/DP_TABLE_COMPLETION_PRIORITY.md and results/dp_completion_priority_v1/.\n")
    def stopped_owner():
        proc=Path(f'/proc/{owner}')
        assert b'run_ec_quality_campaign.py' in (proc/'cmdline').read_bytes()
        status=(proc/'status').read_text()
        assert next(x for x in status.splitlines() if x.startswith('State:')).split()[1] in ('T','t')
        # The suspended EC parent retains the ordinary GPU flock. All other
        # prepared queues remain excluded while this delegated series runs.
        assert any('FLOCK' in line and len(line.split())>4 and line.split()[4]==str(owner) for line in Path('/proc/locks').read_text().splitlines())
    command=Path(f'/proc/{owner}/cmdline').read_bytes();assert b'run_ec_quality_campaign.py' in command
    snapshot();paused_at=time.time();paused=False
    try:
        os.kill(owner,signal.SIGSTOP);paused=True;time.sleep(.1);stopped_owner()
        deadline=time.time()+1200
        while True:
            children=Path(f'/proc/{owner}/task/{owner}/children').read_text().split()
            active=[]
            for pid in children:
                path=Path(f'/proc/{pid}/stat')
                if path.exists() and path.read_text().rsplit(')',1)[1].split()[0]!='Z':active.append(pid)
            if not active:break
            if time.time()>deadline:raise RuntimeError('Current EC full run did not drain in 20 minutes')
            snapshot();time.sleep(5)
        assert gpu_idle()['idle'] is True,'GPU not idle after EC child exited'
        save_json(output/'GPU_DELEGATION.json',dict(owner_pid=owner,owner_stopped=True,current_full_child_exited=True,ordinary_gpu_flock_held_by_owner=True,policy='Only this prepared sequence may use the GPU until the owner is resumed in finally; no measured full run interrupted'))
        frozen=json.loads((runtime/'RUNTIME_MANIFEST.json').read_text())
        for name,expected in frozen['runtime_hashes'].items():assert sha(runtime/name)==expected
        state['status']='RUNNING';snapshot()
        for job in jobs:
            stopped_owner();assert sha(manifest)==identity['manifest_sha256']
            for name,expected in frozen['runtime_hashes'].items():assert sha(runtime/name)==expected
            state['current']=job['id'];snapshot()
            directory=output/job['id'];assert not directory.exists()
            argv=[sys.executable,str(runtime/'run_coslam_round2.py'),'--dataset',job['dataset'],'--scene',job['scene'],'--datadir',job['datadir'],'--condition','DP','--seed',str(job['seed']),'--frames','0','--warmup','20','--quality','--output',str(directory)]
            if job.get('export_mesh'):argv.append('--export-mesh')
            if 'hash_size' in job:argv+=['--hash-size',str(job['hash_size'])]
            save_json(output/(job['id']+'.launch.json'),argv);start=time.time()
            try:
                with (output/(job['id']+'.log')).open('w') as log:
                    child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        while child.poll() is None:
                            stopped_owner()
                            if time.time()-start>10800:raise RuntimeError('DP full-run timeout')
                            snapshot();time.sleep(10)
                    except BaseException:
                        if child.poll() is None:
                            os.killpg(child.pid,signal.SIGTERM)
                            try:child.wait(timeout=30)
                            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        raise
                    if child.returncode:raise RuntimeError('DP run exited with '+str(child.returncode))
                audited=audit_job(output,job,identity)
                expected=job['paired_base_effective_config_sha256']
                assert audited['effective_config_sha256']==expected,'BASE/DP settings mismatch'
                assert audited['frames']==job['paired_base_frames']
                save_json(output/(job['id']+'.done.json'),dict(id=job['id'],returncode=0,elapsed_seconds=time.time()-start))
                state['completed']+=1
            except Exception as error:
                save_json(output/(job['id']+'.failed.json'),dict(id=job['id'],error=str(error)))
                state['failed']+=1;state['failures'].append(dict(id=job['id'],error=str(error)))
            snapshot()
        state['status']='COMPLETE' if not state['failed'] else 'COMPLETE_WITH_FAILURES';state['current']=None;snapshot()
        save_json(output/'queue_complete.json',dict(identity=identity,completed=state['completed'],failed=state['failed']))
        save_json(output/'TABLES_3_4_RUNS_FINISHED.json',state)
    except BaseException as error:
        state['status']='STOPPED: '+str(error);snapshot();save_json(output/'campaign_failure.json',state);raise
    finally:
        if paused:
            os.kill(owner,signal.SIGCONT)
            save_json(output/'EC_QUEUE_RESUMED.json',dict(owner_pid=owner,priority_gap_seconds=time.time()-paused_at,ec_native_quality_summary_seconds_unchanged=True,ec_parent_monitor_elapsed_includes_priority_gap=True,resumed_paris=datetime.now(ZoneInfo('Europe/Paris')).isoformat()))
        save_json(output/'campaign_exit.json',state)


if __name__=='__main__':main()
