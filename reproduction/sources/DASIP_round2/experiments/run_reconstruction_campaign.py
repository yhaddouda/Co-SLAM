#!/usr/bin/env python3
"""Persistent post-training reconstruction campaign with isolated GPU work."""
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
from checked_reconstruction import sha,save_json


def execute_job(job,output,runtime,timeout=7200,cache=None):
    """Execute one declared job; called only by the current exclusive GPU owner."""
    for path in ('upstream_complete','mesh','summary','source_manifest'):
        if not Path(job[path]).is_file():raise RuntimeError('Missing completed upstream input: '+job[path])
    summary=json.loads(Path(job['summary']).read_text())
    if not summary['full_sequence'] or not summary['finite_parameters_and_poses']:raise RuntimeError('Upstream sequence is incomplete/nonfinite')
    if summary['condition']!=job['condition'] or summary['seed']!=job['seed']:raise RuntimeError('Upstream run identity mismatch')
    if sha(job['source_manifest'])!=job['source_manifest_sha256'] or sha(job['asset'])!=job['asset_sha256']:raise RuntimeError('Upstream/asset identity mismatch')
    output=Path(output);directory=output/job['id']
    jobpath=output/'job_inputs'/(job['id']+'.json');jobpath.parent.mkdir(exist_ok=True)
    if directory.exists():raise RuntimeError('Partial output retained, no silent overwrite: '+str(directory))
    save_json(jobpath,job)
    argv=[sys.executable,str(runtime/'checked_reconstruction.py'),'--job',str(jobpath),'--output',str(directory),'--cache',str(cache or output/'gt_cache')]
    save_json(output/(job['id']+'.launch.json'),argv)
    env=os.environ.copy();env.update(PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.time()
    with (output/(job['id']+'.log')).open('w') as log:
        child=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:child.wait(timeout=timeout)
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    if child.returncode:raise RuntimeError('Scoring process failed with code '+str(child.returncode))
    metrics=json.loads((directory/'metrics.json').read_text())
    if metrics['input_mesh_sha256']!=sha(job['mesh']) or metrics['scorer_sha256']!=sha(runtime/'checked_reconstruction.py'):raise RuntimeError('Scored input identity mismatch')
    save_json(output/(job['id']+'.done.json'),dict(id=job['id'],elapsed_seconds=time.time()-start,metrics_sha256=sha(directory/'metrics.json')))
    return metrics


def validate(runtime,output):
    path=Path(output)/'gpu_validation'
    passed=path/'validation_passed.json'
    if passed.exists():
        assert json.loads(passed.read_text())['scorer_sha256']==sha(runtime/'checked_reconstruction.py')
        return
    if path.exists():raise RuntimeError('Failed validation retained: '+str(path))
    env=os.environ.copy();env['PYOPENGL_PLATFORM']='egl'
    with (Path(output)/'gpu_validation.log').open('w') as log:
        subprocess.run([sys.executable,str(runtime/'checked_reconstruction.py'),'--validate','--output',str(path)],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=120,check=True)
    assert passed.exists()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--cache',type=Path,help='Reuse only the scorer-verified, hash-bound common GT cache')
    args=parser.parse_args()
    for key in ('after','output','manifest','runtime'):setattr(args,key,getattr(args,key).resolve())
    if args.cache is not None:args.cache=args.cache.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(args.manifest.read_text());jobs=manifest['jobs']
    frozen=json.loads((args.runtime/'RUNTIME_MANIFEST.json').read_text())
    identity=dict(manifest_sha256=sha(args.manifest),runtime_manifest_sha256=sha(args.runtime/'RUNTIME_MANIFEST.json'))
    if (args.output/'campaign_identity.json').exists():assert json.loads((args.output/'campaign_identity.json').read_text())==identity
    save_json(args.output/'campaign_identity.json',identity)
    save_json(args.output/'job_manifest.json',manifest)
    base=Path(__file__).resolve().parents[1]
    state='WAITING_FOR_UPSTREAM';current=None
    def verify_runtime():
        assert sha(args.manifest)==identity['manifest_sha256']
        for path,expected in frozen['file_hashes'].items():
            if sha(path)!=expected:raise RuntimeError('Frozen scorer/asset changed: '+path)
    def snapshot():
        done=[j['id'] for j in jobs if (args.output/(j['id']+'.done.json')).exists()]
        failed=[j['id'] for j in jobs if (args.output/(j['id']+'.failed.json')).exists()]
        now=datetime.now(ZoneInfo('Europe/Paris')).isoformat()
        record=dict(status=state,updated_paris=now,current=current,completed=len(done),failed=len(failed),total=len(jobs),remaining=len(jobs)-len(done)-len(failed),waiting_for=str(args.after),done=done,failures=failed)
        save_json(args.output/'campaign_status.json',record)
        text=f'# Reconstruction scoring status\n\nUpdated {now}. **{state}: {len(done)}/{len(jobs)} scored, {len(failed)} failed; {record["remaining"]} remaining.**\n\nCurrent: {current or "—"}.\n\n'
        text+='Saved-mesh scoring only: GT-pose and virtual-view culling; accuracy, completion, completion ratio, Depth L1. Three paired seeds, declared scene/method/capacity coverage retained. No SLAM timing overlap. Negative outcomes and individual failures remain recorded.\n'
        (args.output/'LIVE_STATUS.md').write_text(text)
        if (args.after/'campaign_complete.json').exists():(base/'results/LIVE_STATUS.md').write_text('# Round 2 RTX reconstruction status\n\n'+manifest.get('purpose','Declared saved-mesh scoring campaign')+'\n\n'+text.split('\n\n',1)[1]+'\nPaper PDF updates automatically from verified complete paired comparisons. Queue: '+str(args.output.relative_to(base))+'.\n')
        return record
    try:
        verify_runtime();snapshot()
        while not (args.after/'campaign_complete.json').exists():
            if (args.after/'campaign_failure.json').exists():raise RuntimeError('Upstream EC campaign failed; do not overlap it')
            time.sleep(15);snapshot()
        from report_support_reference import gpu_idle
        while gpu_idle()['idle'] is not True:
            state='WAITING_FOR_IDLE_GPU';snapshot();time.sleep(15)
        with open('/tmp/dasip_round2_gpu0.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state='GPU_VALIDATION';snapshot();validate(args.runtime,args.output)
            state='RUNNING'
            for job in jobs:
                current=job['id'];verify_runtime()
                if (args.output/(job['id']+'.done.json')).exists():
                    marker=json.loads((args.output/(job['id']+'.done.json')).read_text())
                    path=args.output/job['id']/'metrics.json'
                    assert sha(path)==marker['metrics_sha256']
                    result=json.loads(path.read_text())
                    assert result['input_mesh_sha256']==sha(job['mesh']) and result['scorer_sha256']==sha(args.runtime/'checked_reconstruction.py')
                    snapshot();continue
                if (args.output/(job['id']+'.failed.json')).exists():continue
                snapshot();print('START',current,flush=True)
                try:execute_job(job,args.output,args.runtime,cache=args.cache)
                except Exception as error:
                    save_json(args.output/(job['id']+'.failed.json'),dict(id=job['id'],error=str(error),policy='retain failed comparison; continue independent declared jobs; no threshold/source changes'))
                    print('FAILED',current,str(error),flush=True)
                snapshot()
            current=None
            record=snapshot();state='COMPLETE' if record['failed']==0 else 'COMPLETE_WITH_FAILURES'
            record=snapshot();save_json(args.output/'campaign_complete.json',record)
        print('RECON_CAMPAIGN_COMPLETE',flush=True)
    except BaseException as error:
        state='STOPPED: '+str(error);record=snapshot();save_json(args.output/'campaign_failure.json',record)
        raise


if __name__=='__main__':main()
