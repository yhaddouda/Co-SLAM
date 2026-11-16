"""Frozen serial continuous Orin timing controller; never changes power/clocks."""
import collections
import csv
from datetime import datetime,timezone,timedelta
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
OLD=ROOT.parent/'orin_paper_night_v2'
CO=ROOT.parent/'orin_coslam_runtime_v1'
RUNTIME=CO/'runtime'
STATE=ROOT/'CONTINUOUS_STATUS.json'
SESSION='dasip_r2_continuous_v4_maxn'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def power():
    value=subprocess.check_output(['/usr/sbin/nvpmodel','-q'],text=True,timeout=10).strip()
    if value!='NV Power Mode: MAXN\n0':raise RuntimeError('POWER_MODE_STOP: '+value)
    return value
def freeze_check():
    for n,h in read(ROOT/'CONTINUOUS_FREEZE.json')['files'].items():assert sha(Path(n))==h,'Frozen file mismatch: '+n
def device_check():
    power()
    devices=sorted({str(p) for pattern in ['nvhost*gpu*','nvmap','nvidia*','nvgpu/**/*'] for p in Path('/dev').glob(pattern) if not p.is_dir()})
    assert devices,'No GPU devices visible'
    for attempt in range(15):
        r=subprocess.run(['fuser','-v',*devices],capture_output=True,text=True,timeout=10)
        load=int(Path('/sys/devices/platform/17000000.gpu/load').read_text())
        if r.returncode==1 and not r.stdout.strip() and not r.stderr.strip() and load==0:break
        time.sleep(2)
    else:raise RuntimeError('GPU_CONFLICT_STOP: '+r.stdout+r.stderr+' load='+str(load))
    with open('/tmp/dasip_round2_gpu0.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
def proc_children(pid):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            parent=next(int(l.split()[1]) for l in (p/'status').read_text().splitlines() if l.startswith('PPid:'))
            if parent==pid:result.append(int(p.name))
        except (OSError,ValueError,StopIteration):pass
    return result
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def validate(job,folder,diagnostic=False):
    spec=importlib.util.spec_from_file_location('old_checker',OLD/'paper_night_parent.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.ROOT=ROOT
    n=50 if diagnostic else job['expected_frames'];j=dict(job,frames=50 if diagnostic else 0)
    summary=mod.check_job(j,folder,n,job['system']=='EC-SLAM')
    assert summary['scene']==job['scene'] and summary['dataset']==job['dataset']
    cfg=read(folder/'config.json');ec=job['system']=='EC-SLAM';calls=rows(folder/'calls.csv')
    for stage,ids in [('tracking_frame' if ec else 'tracking',set(range(n)) if ec else set(range(1,n))),('bundle_adjustment',set(range(cfg['mapping']['every_frame' if ec else 'map_every'],n,cfg['mapping']['every_frame' if ec else 'map_every']))|({n-1} if ec else set()))]:
        selected=[r for r in calls if r['stage']==stage]
        assert len(selected)==len(ids) and {int(r['frame_id']) for r in selected}==ids,(stage,len(selected),len(ids))
    assert all(int(x['actual_queries'])==int(x['dense_queries']) for x in rows(folder/'queries.csv'))
    return summary
def report(s):
    jobs=s['jobs'];counts=collections.Counter(j['status'] for j in jobs)
    s.update(updated_utc=now(),counts={k:counts[k] for k in ['done','reused','running','queued','pending','blocked']},ALL_REQUIRED_MAXN_COMPLETE=all(j['status'] in ['done','reused'] for j in jobs))
    s['ready_jobs_remaining']=counts['queued']+counts['running']
    queued=[j for j in jobs if j['status']=='queued'];s['next_automatic_job']=queued[0]['id'] if queued else None
    samples=[j for j in jobs if j.get('elapsed_seconds') and j['status'] in ['done','reused']]
    def estimate(j):
        exact=[x['elapsed_seconds']/x['expected_frames'] for x in samples if all(x[k]==j[k] for k in ['system','dataset','hash_size','condition'])]
        if not exact:exact=[x['elapsed_seconds']/x['expected_frames'] for x in samples if x['system']==j['system']]
        return statistics.mean(exact)*j['expected_frames'] if exact else None
    remain=[estimate(j) for j in jobs if j['status'] in ['queued','running']];seconds=sum(x for x in remain if x is not None)
    s['forecast']=dict(method='Measured full-scene elapsed seconds per frame, matched system/dataset/capacity/condition when available; fallback within system; scene variation remains uncertain',remaining_ready_seconds=seconds,estimated_finish_utc=(datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat(),does_not_include_blocked_or_engineering=True,matched_second_mode_seconds_estimate=None,second_mode_note='15 W unmeasured: no responsible same-speed claim; author switch/instruction required',abstract_deadline='2026-10-20 Europe/Paris (exact cutoff unspecified)',paper_deadline='2026-10-27 Europe/Paris (exact cutoff unspecified)',risk='Missing ScanNet and unvalidated engineering prevent complete envelope; matching 15 W durations unknown; ETA applies only to executable MAXN jobs')
    save(STATE,s)
    groups={}
    for j in jobs:
        if j['status'] not in ['done','reused']:continue
        for stage in ['tracking','tracking_frame','bundle_adjustment']:
            for metric in ['mean_wall_ms','mean_cuda_ms','all_call_mean_wall_ms']:
                v=j.get('summary',{}).get(stage,{}).get(metric)
                if v is not None:groups.setdefault((j['system'],j['dataset'],j['scene'],j['hash_size'],j['condition'],stage,metric),[]).append((j['seed'],v))
    paired=[]
    for key,values in groups.items():
        paired.append(dict(zip(['system','dataset','scene','capacity','condition','stage','metric'],key),seeds=[x[0] for x in values],mean=statistics.mean(x[1] for x in values),sample_sd=statistics.stdev(x[1] for x in values) if len(values)>1 else None))
    save(ROOT/'TIMING_SUMMARIES.json',paired)
    md='# Continuous MAXN timing report\n\nMilliseconds; primary excludes frames 0–19. Sample SD across paired seeds; all-call means separately retained. EC scopes overlap and are non-additive. No Orin quality metrics or DP expansion.\n\n| System | Dataset | Scene | H | Condition | Stage | Metric | Seeds | Mean | Sample SD |\n|---|---|---|---:|---|---|---|---|---:|---:|\n'
    for x in paired:md+='| '+' | '.join(str(x[k]) for k in ['system','dataset','scene','capacity','condition','stage','metric','seeds','mean','sample_sd'])+' |\n'
    (ROOT/'TIMING_REPORT.md').write_text(md)
    md='# Orin continuous v4 — MAXN\n\nUpdated '+s['updated_utc']+' UTC. Detached session `'+SESSION+'`; parent PID '+str(s.get('parent_pid'))+'.\n\nCounts '+str(s['counts'])+'. **ALL_REQUIRED_MAXN_COMPLETE = '+str(s['ALL_REQUIRED_MAXN_COMPLETE']).upper()+'**. Reused full measurements are counted separately; diagnostics are excluded.\n\nExecutable ready jobs remaining: '+str(s['ready_jobs_remaining'])+'. Next automatic job: '+str(s['next_automatic_job'])+'.\n\nForecast: '+json.dumps(s['forecast'])+'\n\n| System | Dataset | Scene | H | Condition | Seed | Mode | Status | Output / reason |\n|---|---|---|---:|---|---:|---|---|---|\n'
    for j in jobs:md+='| '+' | '.join(str(j.get(k,'')) for k in ['system','dataset','scene','hash_size','condition','seed','mode','status'])+' | '+str(j.get('output',j.get('reason',''))).replace('\n',' ')+' |\n'
    md+='\nClock policy: retain current settings per author; full clock-lock status unknown. No automatic 15 W work. Engineering admission is separate and conditional. Exact commands/config/source identities are in CONTINUOUS_STATUS.json and per-job records.\n'
    (ROOT/'ORIN_RUN_AGENDA.md').write_text(md)
def run_process(argv,log,s,ec_lock=False,extra_freeze=None):
    freeze_check();device_check()
    if extra_freeze:
        for p,h in extra_freeze.items():assert sha(p)==h,p
    lock=open('/tmp/dasip_round2_gpu0.lock','a') if ec_lock else None
    if lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with Path(log).open('x') as stream:
            process=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT)
            s.update(active_child_pid=process.pid,active_command=argv,active_log=str(log));start=time.monotonic()
            while process.poll() is None:
                s['active_gpu_child_pids']=[process.pid] if ec_lock else proc_children(process.pid)
                report(s)
                # A changed mode is recorded while the child enforces its own checks. Never kill a valid full run to satisfy an ETA.
                try:power()
                except Exception as e:s['power_stop_evidence']=str(e)
                time.sleep(5)
            rc=process.wait();elapsed=time.monotonic()-start
        s.update(active_child_pid=None,active_gpu_child_pids=[],last_returncode=rc)
        power();freeze_check()
        return rc,elapsed
    finally:
        if lock:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
def ec_command(j,out,frames=0):
    return [sys.executable,str(ROOT/'run_ec_round2.py'),'--condition',j['condition'],'--seed',str(j['seed']),'--scene',j['scene'],'--hash-size',str(j['hash_size']),'--expected-power-mode','MAXN','--frames',str(frames),'--datadir',j['datadir'],'--output',str(out),'--co-runtime',str(RUNTIME)]
def ec_gate(s):
    evidence=[]
    for h in [16,19]:
        for condition in ['BASE','MC']:
            j=dict(system='EC-SLAM',dataset='Replica',scene='office0',condition=condition,seed=0,hash_size=h,datadir='/home/yh279050/work/EC-SLAM/datasets/vmap/office_0/imap/00')
            out=ROOT/'diagnostics'/('EC_H'+str(h)+'_'+condition);out.parent.mkdir(exist_ok=True)
            argv=ec_command(j,out,50);save(out.with_suffix('.launch.json'),argv)
            rc,elapsed=run_process(argv,out.with_suffix('.log'),s,ec_lock=True)
            assert rc==0,'EC preflight failed: '+str(out.with_suffix('.log'))
            summary=validate(j,out,diagnostic=True);evidence.append(dict(hash_size=h,condition=condition,elapsed_seconds=elapsed,summary=summary))
    save(ROOT/'EC_PREFLIGHT_VALIDATION.json',dict(status='PASS',diagnostics=evidence,estimator_and_timer_bytes_unchanged_from_v2=True,scope='Generalized wrapper, native H16 and explicit H19, both BASE/MC; native threads retained'))
    archive_ec()
def archive_ec():
    manifest=read(ROOT/'EC_RUNTIME_MANIFEST.json')
    with zipfile.ZipFile(ROOT/'EC_VALIDATED_SOURCE_ONLY.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in list(manifest['files'])+['EC_RUNTIME_MANIFEST.json','EC_NATIVE_CHANGE_AUDIT.json','EC_NATIVE_CHANGES.diff','EC_DRIVER_GENERALIZATION.diff','EC_PREFLIGHT_VALIDATION.json']:
            z.write(ROOT/name,name)
def engineering_gap(s):
    request=ROOT/'ENGINEERING_READY_REQUEST.json'
    if not request.exists() or s.get('engineering_gate_attempted'):return
    s['engineering_gate_attempted']=True;r=read(request)
    try:
        assert r['cpu_audit_status']=='PASS'
        rc,elapsed=run_process(r['preflight_command'],ROOT/'engineering_preflight.log',s,ec_lock=True,extra_freeze=r['files'])
        assert rc==0,'Engineering diagnostic failed; inspect engineering_preflight.log'
        result=read(ROOT/'ENGINEERING_PREFLIGHT_VALIDATION.json');assert result['status']=='PASS'
        admitted={j['id']:j for j in r['jobs']}
        for j in s['jobs']:
            if j['id'] in admitted:
                j.update(admitted[j['id']],status='queued',engineering_source_freeze=r['files'])
        s['engineering_gate']='PASS'
    except Exception as error:
        s['engineering_gate']='BLOCKED: '+repr(error)
        for j in s['jobs']:
            if j['condition']=='ENG':j.update(status='blocked',reason=s['engineering_gate'])
        save(ROOT/'ENGINEERING_GATE_FAILURE.json',dict(utc=now(),error=repr(error),traceback=traceback.format_exc()))
def package(s,terminal=False):
    report(s)
    if terminal:save(ROOT/'CONTINUOUS_COMPLETE.json',dict(utc=now(),major_admitted_series_terminal=True,counts=s['counts'],ALL_REQUIRED_MAXN_COMPLETE=s['ALL_REQUIRED_MAXN_COMPLETE'],power=power(),status='All admitted ready work terminal; dependencies remain explicit'))
    target=ROOT.parents[1]/'ORIN_CURRENT_MODE_RESULTS.zip';tmp=target.with_suffix('.zip.tmp');hashes={}
    candidates=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not any(x in p.parts for x in ['.git']) and p.suffix not in ['.tmp','.lock']]
    reused=[Path(j['output']) for j in s['jobs'] if j['status']=='reused']
    entries=[('continuous_v4/'+p.relative_to(ROOT).as_posix(),p) for p in candidates]
    for folder in reused:
        for p in folder.rglob('*'):
            if p.is_file():entries.append(('reused/'+folder.name+'/'+p.relative_to(folder).as_posix(),p))
        for suffix in ['.done.json','.launch.json','.log']:
            p=folder.with_name(folder.name+suffix)
            if p.exists():entries.append(('reused/'+folder.name+'/'+p.name,p))
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,p in entries:
            data=p.read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        z.writestr('RETURN_MANIFEST.json',json.dumps(dict(snapshot_utc=now(),final_snapshot=terminal,ALL_REQUIRED_MAXN_COMPLETE=s['ALL_REQUIRED_MAXN_COMPLETE'],counts=s['counts'],files=hashes),indent=2)+'\n')
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    tmp.replace(target);target.with_suffix('.zip.sha256').write_text(sha(target)+'  '+target.name+'\n')
def main():
    assert not STATE.exists(),'Preserve existing controller state; no silent resume/retry'
    power();freeze_check();device_check()
    parent_lock=open('/tmp/dasip_round2_continuous_v4.lock','a');fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    s=read(ROOT/'LOCAL_MAXN_PLAN.json');s.update(parent_pid=os.getpid(),session=SESSION,started_utc=now(),phase='PREPARING_EC_GATE',active_gpu_child_pids=[]);report(s)
    try:
        try:ec_gate(s);s['ec_gate']='PASS'
        except Exception as e:
            s['ec_gate']='BLOCKED: '+repr(e);save(ROOT/'EC_GATE_FAILURE.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc()))
            for j in s['jobs']:
                if j['system']=='EC-SLAM' and j['status']=='queued':j.update(status='blocked',reason=s['ec_gate'])
        s['phase']='FULL_RUNS';report(s);package(s)
        while True:
            power();engineering_gap(s)
            next_jobs=[j for j in s['jobs'] if j['status']=='queued']
            if not next_jobs:break
            j=next_jobs[0];out=ROOT/'results'/'MAXN'/j['id'];out.parent.mkdir(parents=True,exist_ok=True)
            assert not out.exists(),'Incomplete job preserved: '+str(out)
            j.update(output=str(out),status='running',started_utc=now())
            if j['condition']=='ENG':argv=j['command']
            elif j['system']=='Co-SLAM':
                local=ROOT/'local_manifests'/(j['id']+'.json');local.parent.mkdir(exist_ok=True)
                # These one-job manifests are materialized deterministically from the frozen local plan.
                save(local,dict(warmup_frames=20,jobs=[{k:j[k] for k in ['id','dataset','scene','datadir','condition','seed','hash_size','quality','export_mesh','frames']}]))
                queue_out=out.parent/(j['id']+'_queue')
                argv=[sys.executable,str(CO/'run_manifest_queue.py'),'--runtime',str(RUNTIME),'--manifest',str(local),'--output',str(queue_out),'--power-mode','MAXN']
                out=queue_out/j['id'];j['output']=str(out);j['local_manifest_sha256']=sha(local)
            else:argv=ec_command(j,out)
            records=ROOT/'controller_records';records.mkdir(exist_ok=True)
            j['command']=argv;save(records/(j['id']+'.controller_launch.json'),argv);report(s)
            try:
                log=records/(j['id']+'.controller.log');rc,elapsed=run_process(argv,log,s,ec_lock=j['system']=='EC-SLAM' or j['condition']=='ENG',extra_freeze=j.get('engineering_source_freeze'))
                assert rc==0,'Child failed with return code '+str(rc)+'; '+str(log)
                if j['condition']=='ENG':
                    summary=read(out/'summary.json');assert summary['finite_parameters_and_poses'] and summary['full_sequence'] and summary['frames']==2000 and summary['condition']=='ENG' and summary['seed']==j['seed'] and summary['engineering_validation']=='PASS'
                else:summary=validate(j,out)
                j.update(status='done',summary=summary,elapsed_seconds=elapsed,finished_utc=now(),actual_config_sha256=sha(out/'config.json'),actual_source_manifest_sha256=sha(out/'source_manifest.json'))
                save(records/(j['id']+'.controller_done.json'),dict(id=j['id'],returncode=rc,elapsed_seconds=elapsed,utc=now()))
            except Exception as e:
                j.update(status='blocked',reason=repr(e));save(records/(j['id']+'.controller_failure.json'),dict(error=repr(e),traceback=traceback.format_exc(),utc=now()))
                # A scene/system/capacity comparison stops as a unit; unrelated validated blocks continue.
                for other in s['jobs']:
                    if other['status']=='queued' and all(other[k]==j[k] for k in ['system','dataset','scene','hash_size']):other.update(status='blocked',reason='Affected comparison stopped after '+j['id']+': '+repr(e))
                power();freeze_check();device_check()
            report(s);package(s)
        s['phase']='TERMINAL';report(s);device_check();package(s,terminal=True)
        print('CONTINUOUS_MAJOR_SERIES_TERMINAL',s['counts'],s['ALL_REQUIRED_MAXN_COMPLETE'],flush=True)
    except Exception as error:
        s.update(phase='STOPPED',stop_reason=repr(error),stop_traceback=traceback.format_exc());report(s)
        save(ROOT/'CONTINUOUS_STOPPED.json',dict(utc=now(),reason=repr(error),major_series_terminal=False));print(traceback.format_exc(),flush=True)
        try:package(s)
        except Exception:print(traceback.format_exc(),flush=True)
        raise

if __name__=='__main__':main()
