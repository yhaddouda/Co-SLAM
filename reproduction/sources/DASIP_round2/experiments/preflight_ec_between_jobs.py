#!/usr/bin/env python3
"""Let the current Co child finish, validate EC, then always resume its queue."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-pid',required=True,type=int)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    cmd=Path(f'/proc/{args.queue_pid}/cmdline').read_bytes()
    if b'run_manifest_queue.py' not in cmd or b'rtx_replica_h19_extension.json' not in cmd:
        raise RuntimeError('PID is not the expected Co continuation queue')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    base=Path(__file__).resolve().parents[1]
    manifest=json.loads((base/'experiments/manifests/rtx_ec_quality_all_replica.json').read_text())
    runtime=base/'experiments/ec_quality_runtime_v1'
    record=dict(status='WAITING_FOR_CURRENT_FULL_CO_CHILD',queue_pid=args.queue_pid,diagnostics=[])
    def save():
        (args.output/'status.json').write_text(json.dumps(record,indent=2)+'\n')
    save()
    os.kill(args.queue_pid,signal.SIGSTOP)
    try:
        from report_support_reference import gpu_idle
        begin=time.time()
        while gpu_idle()['idle'] is not True:
            if time.time()-begin>600:
                raise RuntimeError('No idle gap within ten minutes; no preflight launched')
            time.sleep(5)
        record['status']='PREFLIGHT'
        save()
        for capacity in (16,19):
            for condition in ('BASE','MC'):
                job=f'office0_{condition}_H{capacity}_50frames'
                command=[sys.executable,str(runtime/'run_ec_quality_round2.py'),'--scene','office0','--datadir',manifest['jobs'][0]['datadir'],'--condition',condition,'--seed','0','--hash-size',str(capacity),'--frames','50','--export-mesh','--output',str(args.output/job)]
                environment=os.environ.copy();environment['PYTHONHASHSEED']='0'
                with (args.output/(job+'.log')).open('w') as log:
                    child=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        child.wait(timeout=600)
                    except BaseException:
                        os.killpg(child.pid,signal.SIGTERM)
                        child.wait(timeout=30)
                        raise
                if child.returncode:
                    raise RuntimeError('EC diagnostic failed: '+job)
                summary=json.loads((args.output/job/'summary.json').read_text())
                assert summary['frames']==50 and summary['finite_parameters_and_poses']
                record['diagnostics'].append(summary)
                save()
        record['status']='PASSED'
    except BaseException as error:
        record['status']='FAILED: '+str(error)
        raise
    finally:
        os.kill(args.queue_pid,signal.SIGCONT)
        record['co_queue_resumed']=True
        save()


if __name__=='__main__':main()
