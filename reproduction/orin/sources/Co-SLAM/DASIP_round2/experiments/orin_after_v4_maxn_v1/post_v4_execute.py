"""Deferred exclusive preparation and serial MAXN ScanNet/conditional Co ENG."""
import argparse
import csv
from datetime import datetime,timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parent
V4=ROOT.parent/'orin_continuous_v4'
CO=ROOT.parent/'orin_coslam_runtime_v1/runtime'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def maxn():
    mode=subprocess.check_output(['/usr/sbin/nvpmodel','-q'],text=True,timeout=10).strip()
    assert mode=='NV Power Mode: MAXN\n0','MAXN continuation stopped: '+mode
    return mode
def hardware():
    values={}
    for pattern in ['/sys/class/devfreq/17000000.gpu/*freq','/sys/class/thermal/thermal_zone*/temp','/sys/class/thermal/thermal_zone*/type']:
        for p in Path('/').glob(pattern.lstrip('/')):
            try:values[str(p)]=p.read_text().strip()
            except Exception as e:values[str(p)]={'error':repr(e)}
    return dict(utc=now(),power=maxn(),sysfs=values,clock_policy='Retain author-selected current settings; no clock changes; full jetson_clocks lock state remains unknown')
def readiness(label,baseline):
    # Earlier protocol records mode/frequency/temperature without claiming full clock locking.
    # Add an explicit idle settling record after preparation instead of silently timing a hot extractor/JIT.
    samples=[];deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        sample=hardware();samples.append(sample)
        for name in ['min_freq','max_freq']:
            key='/sys/class/devfreq/17000000.gpu/'+name
            if isinstance(baseline['sysfs'].get(key),str):assert sample['sysfs'].get(key)==baseline['sysfs'][key],'Clock policy changed; no automatic adjustment'
        if len(samples)>=3:
            keys=[k for k,v in sample['sysfs'].items() if k.endswith('/temp') and isinstance(v,str) and v.isdigit()]
            stable=bool(keys) and all(max(int(s['sysfs'][k]) for s in samples[-3:])-min(int(s['sysfs'][k]) for s in samples[-3:])<=2000 for k in keys if all(isinstance(s['sysfs'].get(k),str) and s['sysfs'][k].isdigit() for s in samples[-3:]))
            if stable:
                save(ROOT/'readiness'/f'{label}.json',dict(status='READY',criterion='Three idle temperature/frequency observations 30 s apart; temperature range <=2 C; retained GPU min/max; MAXN queried each time; no inferred full clock lock',samples=samples));return
        time.sleep(30)
    save(ROOT/'readiness'/f'{label}.json',dict(status='BLOCKED_NO_STABLE_SETTLING_RECORD',samples=samples));raise RuntimeError('Preparation settling/readiness did not stabilize within 15 minutes')
def update(s):
    from collections import Counter
    required=[j for j in s['jobs'] if j.get('required_for_completion',True) and j['status']!='excluded']
    counts=Counter(j['status'] for j in s['jobs']);counts['excluded']=len(s.get('excluded_jobs',[]))
    complete=bool(required) and s.get('v4_original_ready_verified',False) and all(j['status'] in ['done','reused'] for j in required)
    s.update(updated_utc=now(),counts=dict(counts),ALL_REQUIRED_MAXN_COMPLETE=complete)
    save(ROOT/'POST_V4_STATUS.json',s)
    text='# Post-v4 MAXN continuation\n\n'+s['updated_utc']+' UTC; phase '+s['phase']+'. Counts '+str(s['counts'])+'. **ALL_REQUIRED_MAXN_COMPLETE='+str(complete).lower()+'**. Engineering is Co-SLAM only; EC ENG is excluded from scope and completion requirements.\n\n'
    text+='V4 original status and results are preserved in `../orin_continuous_v4/`. Preparation starts only after normal exhaustion of all expected ready identities and exclusive workload ownership. No 22:00 or ALL_REQUIRED trigger.\n\n| ID | Status | Output / reason |\n|---|---|---|\n'
    for j in s['jobs']:text+='| '+j['id']+' | '+j['status']+' | '+str(j.get('output',j.get('reason',''))).replace('\n',' ')+' |\n'
    for j in s.get('excluded_jobs',[]):text+='| '+j['id']+' | excluded | '+j['reason']+' |\n'
    (ROOT/'ORIN_RUN_AGENDA.md').write_text(text)
def run(argv,log,fd,s):
    maxn()
    from post_v4_follower import idle
    for attempt in range(15):
        try:idle();break
        except AssertionError:
            if attempt==14:raise
            time.sleep(2)
    with Path(log).open('x') as stream:
        p=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,pass_fds=(fd,));s.update(active_child_pid=p.pid,active_command=argv,active_log=str(log));update(s)
        while p.poll() is None:time.sleep(20)
        rc=p.wait()
    s['active_child_pid']=None;maxn();return rc
def verify_output(j,out,engineering=False,diagnostic=False):
    a=read(out/'summary.json');cfg=read(out/'config.json');n=j['expected_native_frames']
    if diagnostic:n=50
    assert a['frames']==n and a['finite_parameters_and_poses'] and not a['quality'] and a['diagnostic']==diagnostic
    assert a['full_sequence']==(not diagnostic) and a['warmup_frames']==20
    assert a['seed']==j['seed'] and a['condition']==j['condition'] and a['scene']==j['scene']
    assert a['initial_power']==a['final_power']=='NV Power Mode: MAXN\n0'
    assert cfg['round2']['engineering']==engineering and cfg['grid']['hash_size']==j['hash_size']
    assert cfg['mapping']['n_pixels']==.05 and not cfg['mapping']['first_mesh'] and not cfg['mesh']['visualisation']
    with (out/'calls.csv').open() as f:calls=list(csv.DictReader(f))
    step=cfg['mapping']['map_every']
    for stage,ids in [('tracking',set(range(1,n))),('bundle_adjustment',set(range(step,n,step)))]:
        selected=[x for x in calls if x['stage']==stage];assert len(selected)==len(ids) and {int(x['frame_id']) for x in selected}==ids
    with (out/'iterations.csv').open() as f:events=list(csv.DictReader(f))
    for stage,iters,ids in [('tracking',cfg['tracking']['iter'],set(range(1,n))),('bundle_adjustment',cfg['mapping']['iters'],set(range(step,n,step)))]:
        selected=[x for x in events if x['iteration_type']==stage];assert len(selected)==len(ids)*iters
        assert {(int(x['frame_id']),int(x['iteration_id'])) for x in selected}=={(f,i) for f in ids for i in range(iters)}
    assert all(math.isfinite(float(x[k])) and float(x[k])>=0 for x in events for k in ['forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms'])
    assert all(math.isfinite(float(x[k])) and float(x[k])>=0 for x in calls for k in ['wall_ms','cuda_ms'])
    with (out/'queries.csv').open() as f:assert all(int(x['actual_queries'])==int(x['dense_queries']) for x in csv.DictReader(f))
    return a
def queue(j,runtime,fd,s):
    manifest=ROOT/'local_manifests'/(j['id']+'.json');save(manifest,dict(warmup_frames=20,jobs=[{k:v for k,v in j.items() if k in ['id','dataset','scene','datadir','condition','seed','hash_size','quality','export_mesh','frames','engineering']}]))
    out=ROOT/'results'/j['id'];assert not out.exists(),'Prior partial job preserved; no retry: '+str(out)
    argv=[sys.executable,str(ROOT/'run_queue_inherited_lock.py'),'--inherited-workload-lock-fd',str(fd),'--runtime',str(runtime),'--manifest',str(manifest),'--output',str(out),'--power-mode','MAXN']
    j.update(status='running',command=argv,output=str(out/j['id']),manifest_sha256=sha(manifest));update(s);start=time.monotonic()
    assert run(argv,ROOT/'logs'/(j['id']+'.log'),fd,s)==0,'Queue child failed; exact log preserved'
    assert (out/'queue_complete.json').is_file() and read(out/(j['id']+'.done.json'))['returncode']==0
    result=verify_output(j,out/j['id'],j.get('engineering',False));j.update(status='done',summary=result,elapsed_seconds=time.monotonic()-start,source_manifest_sha256=sha(out/j['id']/'source_manifest.json'),config_sha256=sha(out/j['id']/'config.json'));update(s)
def prior_failure_report():
    # This deliberately runs only AFTER the normal-exhaustion/exclusive gate.
    from engineering_followup_after_gate import failure_report
    failure_report()
def capacity_diagnostics(prepared,fd,s,baseline):
    result={};tested={16}
    for item in prepared:
        if item['status']!='ready':continue
        scene=next(x for x in read(ROOT/'SCANNET_SCENES.json')['scenes'] if x['scene']==item['scene']);h=scene['native_hash_size']
        if h in tested:continue
        try:
            for cond in ['BASE','MC']:
                j=dict(dataset='ScanNet',scene=scene['scene'],condition=cond,seed=0,hash_size=h,expected_native_frames=scene['frames'])
                out=ROOT/'diagnostics'/f'ScanNet_{scene["scene"]}_{cond}_H{h}';out.parent.mkdir(exist_ok=True)
                argv=[sys.executable,str(CO/'run_coslam_round2.py'),'--dataset','ScanNet','--scene',scene['scene'],'--datadir',item['path'],'--condition',cond,'--seed','0','--hash-size',str(h),'--frames','50','--warmup','20','--output',str(out),'--expected-power-mode','MAXN']
                assert run(argv,ROOT/'logs'/(out.name+'.log'),fd,s)==0,'New-capacity diagnostic failed'
                verify_output(j,out,diagnostic=True)
            tested.add(h);result[str(h)]='PASS'
        except Exception as e:
            item.update(status='blocked',reason='New-capacity preflight failed for this scene: '+repr(e));result[scene['scene']]=dict(status='FAIL',error=repr(e))
    save(ROOT/'NEW_CAPACITY_PREFLIGHT.json',result)
    readiness('after_data_and_preflight_before_scannet_timings',baseline)
def package(s):
    update(s);files={};target=ROOT.parents[1]/'ORIN_POST_V4_MAXN_RESULTS.zip';tmp=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            if not p.is_file() or any(x in p.parts for x in ['prepared_data','downloaded_data','__pycache__']) or p.suffix in ['.tmp','.sens']:continue
            name=p.relative_to(ROOT).as_posix();data=p.read_bytes();files[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        z.writestr('RETURN_MANIFEST.json',json.dumps(dict(utc=now(),counts=s['counts'],ALL_REQUIRED_MAXN_COMPLETE=s['ALL_REQUIRED_MAXN_COMPLETE'],excluded_jobs=s.get('excluded_jobs',[]),files=files),indent=2)+'\n')
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    tmp.replace(target);target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--workload-lock-fd',type=int,required=True);args=p.parse_args();fd=args.workload_lock_fd
    assert os.fstat(fd).st_ino==os.stat('/tmp/dasip_round2_gpu0.lock').st_ino
    fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert (ROOT/'NORMAL_EXHAUSTION_TRANSITION.json').exists();os.environ['DASIP_POST_V4_EXCLUSIVE']='1'
    maxn();baseline=hardware();save(ROOT/'POST_V4_ENVIRONMENT.json',baseline)
    jobs=read(ROOT/'manifests/orin_scannet_native.json')['jobs']
    jobs=[dict(j,status='waiting') for j in jobs]
    # Two additional BASE controls are explicitly declared for the new isolated RTX-reference runtime.
    for seed in [0,1]:
        for eng in [False,True]:jobs.append(dict(id=f'Co_RTXREF_office0_{"ENG" if eng else "BASE_CONTROL"}_seed{seed}',dataset='Replica',scene='office0',condition='BASE',engineering=eng,seed=seed,hash_size=16,frames=0,quality=False,export_mesh=False,expected_native_frames=2000,status='conditional',datadir='/home/yh279050/work/data/Replica/office0'))
    excluded=[dict(id=f'EC_ENG_seed{seed}',status='excluded',required_for_completion=False,reason='Author scope decision 19 October 2026: engineering is restricted to Co-SLAM; EC BASE/MC remain in scope') for seed in [0,1]]
    s=dict(phase='PREPARING',jobs=jobs,excluded_jobs=excluded,active_child_pid=None,extra_paired_BASE_controls=2,v4_original_ready_slots_preserved=168,v4_original_ready_verified=True,declared_total_MAXN_envelope_including_extra_controls=196,notification_policy='Only major planned/admitted series end, fresh idle and decision summary; never routine tasks or isolated failure')
    update(s)
    try:
        try:prior_failure_report()
        except Exception as error:
            save(ROOT/'ENGINEERING_FAILURE_REPORT.json',dict(status='REPORT_INCOMPLETE',error=repr(error),traceback=traceback.format_exc(),original_failure=read(V4/'ENGINEERING_PREFLIGHT_VALIDATION.json'),original_evidence_preserved=True))
            (ROOT/'ENGINEERING_FAILURE_REPORT.md').write_text('Engineering failure reporting is incomplete; see JSON and preserved v4 errors. This does not block independently ready ScanNet work.\n')
        from prepare_scannet_after_gate import prepare_all
        prepared=prepare_all();maxn()
        readiness('after_all_data_preparation',baseline);capacity_diagnostics(prepared,fd,s,baseline)
        for item in prepared:
            group=[j for j in jobs if j.get('dataset')=='ScanNet' and j['scene']==item['scene']]
            if item['status']!='ready':
                for j in group:j.update(status='blocked',reason=item.get('reason','Scene preparation failed'))
                update(s);continue
            scene=next(x for x in read(ROOT/'SCANNET_SCENES.json')['scenes'] if x['scene']==item['scene'])
            assert sha(CO/'configs/ScanNet'/(scene['scene']+'.yaml'))==scene['scene_config_sha256']
            for j in group:j.update(datadir=item['path'],status='queued')
        s['phase']='SCANNET_TIMINGS';update(s)
        for j in jobs:
            if j['status']!='queued':continue
            try:queue(j,CO,fd,s)
            except Exception as e:
                j.update(status='blocked',reason=repr(e));save(ROOT/'failures'/(j['id']+'.json'),dict(error=repr(e),traceback=traceback.format_exc()))
                for other in jobs:
                    if other.get('scene')==j['scene'] and other['status']=='queued':other.update(status='blocked',reason='Affected comparison stopped after '+j['id'])
                maxn()
            package(s)
        # No engineering work, downloads or extraction overlapped any full timing.
        s['phase']='ENGINEERING_INVESTIGATION';update(s)
        try:
            from engineering_followup_after_gate import prepare_and_validate
            runtime=prepare_and_validate(fd,s,run,baseline,readiness)
            for j in jobs:
                if j['status']=='conditional':j['status']='queued'
            s['phase']='CO_ENGINEERING_MATCHED_TIMINGS';update(s)
            for j in jobs:
                if j['status']=='queued':queue(j,runtime,fd,s);package(s)
        except Exception as e:
            for j in jobs:
                if j['status'] in ['conditional','queued','running']:j.update(status='blocked',reason='Co RTX-reference engineering validation/execution unresolved: '+repr(e))
            save(ROOT/'CO_ENGINEERING_STOP.json',dict(error=repr(e),traceback=traceback.format_exc()));maxn()
        s['phase']='COMPLETE_ADMITTED_WORK';update(s)
        save(ROOT/'POST_V4_COMPLETE.json',dict(utc=now(),major_admitted_series_terminal=True,counts=s['counts'],ALL_REQUIRED_MAXN_COMPLETE=s['ALL_REQUIRED_MAXN_COMPLETE'],power=maxn(),EC_engineering='Excluded from scope and completion requirements by author; EC BASE/MC unchanged'))
        package(s)
    except Exception as e:
        s.update(phase='STOPPED',error=repr(e));update(s);save(ROOT/'POST_V4_STOPPED.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),major_series_terminal=False));package(s);raise

if __name__=='__main__':main()
