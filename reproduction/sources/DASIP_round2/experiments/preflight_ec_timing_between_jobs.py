#!/usr/bin/env python3
"""Qualify native-threaded EC call timing between full quality runs."""
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from checked_reconstruction import sha,save_json
from report_support_reference import gpu_idle


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--queue-pid',required=True,type=int);a=p.parse_args()
    base=Path(__file__).resolve().parents[1];runtime=base/'experiments/ec_timing_runtime_v1'
    output=base/'results/ec_timing_preflight_v1';output.mkdir(exist_ok=False)
    record=dict(status='WAITING_FOR_CURRENT_EC_CHILD',runtime_manifest_sha256=sha(runtime/'RUNTIME_MANIFEST.json'),diagnostics=[])
    def write():save_json(output/'status.json',record)
    command=Path(f'/proc/{a.queue_pid}/cmdline').read_bytes();assert b'run_ec_quality_campaign.py' in command
    write();os.kill(a.queue_pid,signal.SIGSTOP)
    try:
        start=time.time()
        while gpu_idle()['idle'] is not True:
            if time.time()-start>600:raise RuntimeError('No idle gap within ten minutes')
            time.sleep(5)
        for capacity in (16,19):
            for condition in ('BASE','MC'):
                label=f'{condition}_H{capacity}';record['status']='DIAGNOSTIC_'+label;write()
                argv=[sys.executable,str(runtime/'run_ec_quality_round2.py'),'--scene','office0','--datadir','/home/yh279050/work/EC-SLAM/datasets/vmap/office_0/imap/00','--condition',condition,'--seed','0','--hash-size',str(capacity),'--frames','50','--output',str(output/label)]
                with (output/(label+'.log')).open('w') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
                s=json.loads((output/label/'summary.json').read_text());assert s['finite_parameters_and_poses'] and s['frames']==50
                assert s['tracking']['count']==30 and s['bundle_adjustment']['count']==7
                with (output/label/'calls.csv').open() as f:rows=list(csv.DictReader(f))
                assert sum(r['stage']=='tracking' for r in rows)==50 and sum(r['stage']=='bundle_adjustment' for r in rows)==11
                with (output/label/'iterations.csv').open() as f:iterations=list(csv.DictReader(f))
                assert sum(r['iteration_type']=='tracking' for r in iterations)==49*8
                assert sum(r['iteration_type']=='bundle_adjustment' for r in iterations)==300
                record['diagnostics'].append(dict(condition=condition,hash_size=capacity,tracking_calls=50,mapping_calls=11,legacy_iteration_rows=len(iterations),finite=True));write()
        record['status']='PASSED'
    except BaseException as error:record['status']='FAILED: '+str(error)
    finally:
        os.kill(a.queue_pid,signal.SIGCONT);record['ec_quality_queue_resumed']=True;write()


if __name__=='__main__':main()
