#!/usr/bin/env python3
"""Serial, resumable Round 2 queue. Never changes power mode or existing sources."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,default=Path(__file__).parent/'runtime')
    parser.add_argument('--power-mode',help='Exact current Orin nvpmodel name; never switched by this script')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if platform.machine()=='aarch64' and not args.power_mode:parser.error('Orin requires --power-mode with the author-selected current mode')
    document=json.loads(args.manifest.read_text());jobs=document['jobs']
    ids=[j['id'] for j in jobs]
    if len(set(ids))!=len(ids) or any(Path(i).name!=i or i in ('.','..') for i in ids):parser.error('Unsafe or repeated job identifiers')
    runtime=args.runtime.resolve();output=args.output.resolve()
    identity=dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  runtime_manifest_sha256=hashlib.sha256((runtime/'RUNTIME_MANIFEST.json').read_bytes()).hexdigest(),
                  power_mode=args.power_mode)
    if output.exists() and any(output.iterdir()):
        if not args.resume:parser.error('Queue output exists; use --resume only for the identical manifest/runtime/mode')
        if json.loads((output/'queue_identity.json').read_text())!=identity:parser.error('Queue identity differs')
    output.mkdir(parents=True,exist_ok=True)
    (output/'queue_identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    (output/'job_manifest.json').write_text(json.dumps(document,indent=2)+'\n')
    state=[]
    # A nonblocking, machine-local lock prevents accidentally launching two queues.
    with open('/tmp/dasip_round2_gpu0.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another Round 2 queue is active on this machine')
        for job in jobs:
            directory=output/job['id'];summary=directory/'summary.json'
            argv=[sys.executable,str(runtime/'run_coslam_round2.py'),
                  '--dataset',job['dataset'],'--scene',job['scene'],'--datadir',job['datadir'],
                  '--condition',job['condition'],'--seed',str(job['seed']),
                  '--frames',str(job.get('frames',0)),'--warmup','20','--output',str(directory)]
            if 'hash_size' in job:argv+=['--hash-size',str(job['hash_size'])]
            if job.get('quality'):argv+=['--quality']
            if job.get('export_mesh'):argv+=['--export-mesh']
            if job.get('wall_only'):argv+=['--no-iteration-events']
            if args.power_mode:argv+=['--expected-power-mode',args.power_mode]
            launch=output/(job['id']+'.launch.json')
            if args.resume and (output/(job['id']+'.done.json')).exists():
                if json.loads(launch.read_text())!=argv:raise RuntimeError('Saved launch differs: '+job['id'])
                result=json.loads(summary.read_text())
                if result['seed']!=job['seed'] or result['condition']!=job['condition']:raise RuntimeError('Saved summary differs')
                if (directory/'source_manifest.json').read_bytes()!=(runtime/'RUNTIME_MANIFEST.json').read_bytes():raise RuntimeError('Saved runtime differs')
                state.append(dict(id=job['id'],status='reused'));continue
            if directory.exists() or launch.exists():raise RuntimeError('Incomplete job preserved; choose a new queue output after diagnosing: '+job['id'])
            launch.write_text(json.dumps(argv,indent=2)+'\n')
            print('START',job['id'],flush=True);start=time.time()
            with (output/(job['id']+'.log')).open('w') as log:
                process=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            record=dict(id=job['id'],returncode=process.returncode,elapsed_seconds=time.time()-start)
            state.append(record)
            (output/'progress.json').write_text(json.dumps(dict(jobs=state,total_jobs=len(jobs)),indent=2)+'\n')
            if process.returncode:raise SystemExit('Job failed; queue stopped; inspect '+str(output/(job['id']+'.log')))
            result=json.loads(summary.read_text())
            if not result['finite_parameters_and_poses']:raise RuntimeError('Nonfinite job')
            if job.get('frames',0)==0 and not result['full_sequence']:raise RuntimeError('Full sequence did not complete')
            if job.get('export_mesh') and not (directory/'mesh_export.json').exists():raise RuntimeError('Mesh export incomplete')
            (output/(job['id']+'.done.json')).write_text(json.dumps(record,indent=2)+'\n')
            print('COMPLETE',job['id'],flush=True)
        (output/'queue_complete.json').write_text(json.dumps(dict(identity=identity,completed=len(state)),indent=2)+'\n')
        print('QUEUE_COMPLETE',len(state),flush=True)


if __name__=='__main__':main()
