#!/usr/bin/env python3
"""EC training -> qualified engineering -> reconstruction; no night time cap."""
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from checked_reconstruction import sha,save_json


def main():
    base=Path(__file__).resolve().parents[1];output=base/'results/remaining_night_v1';output.mkdir(exist_ok=False)
    state=dict(status='WAITING_FOR_EC_TRAINING',engineering='PREPARATION',reconstruction='QUEUED_246',ec_timing='QUALIFICATION_FOR_96',automatic_email=False)
    def write():
        state['updated_paris']=datetime.now(ZoneInfo('Europe/Paris')).isoformat();save_json(output/'status.json',state)
        (output/'LIVE_STATUS.md').write_text('# Remaining RTX night queue\n\n'+json.dumps(state,indent=2)+'\n')
    def child(argv,label):
        with (output/(label+'.log')).open('w') as log:
            p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
            while p.poll() is None:
                write()
                if label=='engineering':
                    queue=base/'results/engineering_v1'
                    count=len(list(queue.glob('*.done.json'))) if queue.exists() else 0
                    (base/'results/LIVE_STATUS.md').write_text(f'# Round 2 RTX night status\n\nAll 312 scientific Co/EC training runs finished. Separate engineering comparison: **{count}/6 full runs complete**. The 246-mesh reconstruction stage follows automatically.\n\nSee results/remaining_night_v1/status.json and results/reconstruction_v1/.\n')
                time.sleep(15)
            return p.returncode
    write();ec=base/'results/ec_quality_v1'
    while not (ec/'campaign_complete.json').exists():
        if (ec/'campaign_failure.json').exists():raise RuntimeError('EC training failed; no overlapping continuation')
        time.sleep(15);write()
    from report_support_reference import gpu_idle,running
    while gpu_idle()['idle'] is not True or running():time.sleep(15)
    gate=base/'results/engineering_preflight_v1/status.json'
    runtime=base/'experiments/coslam_engineering_runtime_v1'
    validated=gate.exists() and json.loads(gate.read_text())['status']=='PASSED'
    if validated and json.loads(gate.read_text())['runtime_manifest_sha256']==sha(runtime/'RUNTIME_MANIFEST.json'):
        state.update(status='ENGINEERING',engineering='RUNNING_6_FULL_SEQUENCES');write()
        argv=[sys.executable,str(base/'experiments/run_manifest_queue.py'),'--runtime',str(runtime),'--manifest',str(base/'experiments/manifests/rtx_engineering_office0.json'),'--output',str(base/'results/engineering_v1')]
        code=child(argv,'engineering');state['engineering']='COMPLETE_6' if code==0 else 'FAILED_PRESERVED'
    else:
        state['engineering']='BLOCKED_PREFLIGHT_NOT_PASSED; independent reconstruction continues'
    state.update(status='RECONSTRUCTION',reconstruction='RUNNING_OR_REUSING_FIRST_REAL_SCORE');write()
    argv=[sys.executable,str(base/'experiments/run_reconstruction_campaign.py'),'--after',str(ec),'--runtime',str(base/'experiments/reconstruction_runtime_v2'),'--manifest',str(base/'experiments/manifests/rtx_reconstruction_all.json'),'--output',str(base/'results/reconstruction_v1')]
    code=child(argv,'reconstruction')
    completed=base/'results/reconstruction_v1/campaign_complete.json'
    state['reconstruction']=json.loads(completed.read_text())['status'] if code==0 and completed.exists() else 'FAILED_PRESERVED'
    state['status']='EC_TIMING';write()
    timing_runtime=base/'experiments/ec_timing_runtime_v1'
    timing_gate=base/'results/ec_timing_preflight_v1/status.json'
    ready=timing_gate.exists() and json.loads(timing_gate.read_text())['status']=='PASSED'
    if ready and json.loads(timing_gate.read_text())['runtime_manifest_sha256']==sha(timing_runtime/'RUNTIME_MANIFEST.json'):
        state['ec_timing']='RUNNING_96_FULL_PROFILES';write()
        argv=[sys.executable,str(base/'experiments/run_ec_timing_campaign.py'),'--after',str(ec),'--runtime',str(timing_runtime),'--manifest',str(base/'experiments/manifests/rtx_ec_complete_call_timing.json'),'--output',str(base/'results/ec_timing_v1')]
        code=child(argv,'ec_timing');state['ec_timing']='COMPLETE_96' if code==0 else 'FAILED_PRESERVED'
    else:state['ec_timing']='BLOCKED_PREFLIGHT_NOT_PASSED'
    state['status']='FINISHED';write();save_json(output/'campaign_complete.json',state)
    (output/'DECISIONS_READY.md').write_text('The prepared RTX night series has finished. Check engineering, reconstruction and EC timing statuses/failures, the updated working paper and per-scene reports before choosing claims. No unattended email was sent: no email transport is configured. Scientific pruning remains unaccepted unless separately established. EC engineering and matching the returned Orin timing protocol remain pending.\n')


if __name__=='__main__':main()
