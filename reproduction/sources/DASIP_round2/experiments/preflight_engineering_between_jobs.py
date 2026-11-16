#!/usr/bin/env python3
"""Qualify the optional engineering stage without overlapping reconstruction."""
import argparse
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
    p=argparse.ArgumentParser();p.add_argument('--queue-pid',type=int,required=True);a=p.parse_args()
    base=Path(__file__).resolve().parents[1];runtime=base/'experiments/coslam_engineering_runtime_v1'
    output=base/'results/engineering_preflight_v1';output.mkdir(exist_ok=False)
    record=dict(status='WAITING_FOR_RECON_PREFLIGHT',runtime_manifest_sha256=sha(runtime/'RUNTIME_MANIFEST.json'))
    def write():save_json(output/'status.json',record)
    write()
    while True:
        path=base/'results/reconstruction_v1/early_preflight_status.json'
        if path.exists() and json.loads(path.read_text()).get('ec_queue_resumed'):break
        time.sleep(5)
    command=Path(f'/proc/{a.queue_pid}/cmdline').read_bytes()
    if b'run_ec_quality_campaign.py' not in command:raise RuntimeError('Unexpected queue owner')
    os.kill(a.queue_pid,signal.SIGSTOP)
    try:
        record['status']='WAITING_FOR_CURRENT_EC_CHILD';write();start=time.time()
        while gpu_idle()['idle'] is not True:
            if time.time()-start>600:raise RuntimeError('No idle gap in ten minutes')
            time.sleep(5)
        record['status']='GPU_EQUIVALENCE';write()
        with (output/'fixed_state.log').open('w') as log:
            subprocess.run([sys.executable,str(runtime/'round2_engineering.py'),'--validate',str(output/'fixed_state.json')],cwd=runtime,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        for eng in (False,True):
            label='ENG' if eng else 'BASE'
            argv=[sys.executable,str(runtime/'run_coslam_round2.py'),'--dataset','Replica','--scene','office0','--datadir','/home/yh279050/work/datasets/data/Replica/office0','--condition','BASE','--seed','0','--frames','50','--quality','--output',str(output/label)]
            if eng:argv+=['--engineering']
            record['status']='DIAGNOSTIC_'+label;write()
            with (output/(label+'.log')).open('w') as log:
                subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
            summary=json.loads((output/label/'summary.json').read_text());assert summary['finite_parameters_and_poses'] and summary['frames']==50 and summary['variant']==label
        import numpy as np
        with np.load(output/'BASE/trajectories.npz') as x,np.load(output/'ENG/trajectories.npz') as y:
            error=float(np.max(abs(x['estimated']-y['estimated'])))
        if error>.01:raise RuntimeError('Unexpected online diagnostic pose difference: '+str(error))
        record.update(status='PASSED',online_diagnostic_max_pose_element_difference=error,scope='fixed-state equivalence and short finite trajectory check; full three-seed ATE still required')
        with (output/'renderer_debug.log').open('w') as log:
            result=subprocess.run([sys.executable,str(base/'experiments/checked_reconstruction.py'),'--validate','--output',str(output/'renderer_debug')],stdout=log,stderr=subprocess.STDOUT,timeout=120)
        record['independent_renderer_debug_returncode']=result.returncode
    except BaseException as error:
        record['status']='FAILED: '+str(error)
    finally:
        os.kill(a.queue_pid,signal.SIGCONT);record['ec_queue_resumed']=True;write()


if __name__=='__main__':main()
