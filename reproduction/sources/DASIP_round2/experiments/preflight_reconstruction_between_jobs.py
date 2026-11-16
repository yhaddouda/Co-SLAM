#!/usr/bin/env python3
"""Validate the scorer and finish its first real job in a clean EC inter-run gap."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from run_reconstruction_campaign import validate,execute_job
from checked_reconstruction import save_json
from report_support_reference import gpu_idle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-pid',required=True,type=int)
    args=parser.parse_args()
    command=Path(f'/proc/{args.queue_pid}/cmdline').read_bytes()
    if b'run_ec_quality_campaign.py' not in command:raise RuntimeError('Not the expected EC queue owner')
    base=Path(__file__).resolve().parents[1]
    output=base/'results/reconstruction_v1';output.mkdir(exist_ok=True)
    runtime=base/'experiments/reconstruction_runtime_v2'
    job=json.loads((base/'experiments/manifests/rtx_reconstruction_all.json').read_text())['jobs'][0]
    record=dict(status='WAITING_FOR_CURRENT_EC_CHILD',current=job['id'])
    def write():save_json(output/'early_preflight_status.json',record)
    write();os.kill(args.queue_pid,signal.SIGSTOP)
    try:
        start=time.time()
        while gpu_idle()['idle'] is not True:
            if time.time()-start>600:raise RuntimeError('No idle gap in ten minutes')
            time.sleep(5)
        record['status']='VALIDATING';write();validate(runtime,output)
        record['status']='FIRST_REAL_RECONSTRUCTION_JOB';write()
        metrics=execute_job(job,output,runtime,timeout=1800)
        record['status']='PASSED';record['first_real_metrics']=metrics['metrics'];record['first_job_seconds']=metrics['wall_seconds']
    except BaseException as error:
        record['status']='FAILED: '+str(error)
        raise
    finally:
        os.kill(args.queue_pid,signal.SIGCONT);record['ec_queue_resumed']=True;write()


if __name__=='__main__':main()
