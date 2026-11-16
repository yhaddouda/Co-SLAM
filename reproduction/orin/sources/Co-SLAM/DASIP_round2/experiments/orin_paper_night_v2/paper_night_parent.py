"""Bounded MAXN campaign: reuse existing work, validate, run ordered blocks, report."""
import csv
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parent
CO=ROOT.parent/'orin_coslam_runtime_v1'
RUNTIME=CO/'runtime'
RESULTS=ROOT/'results'
PY=sys.executable
ORIGINAL=CO/'results/full_maxn'
PLAN=ROOT/'PAPER_NIGHT_PLAN.json'
STATE=ROOT/'PAPER_NIGHT_STATUS.json'
AGENDA=ROOT/'ORIN_RUN_AGENDA.md'


def now():return datetime.now(timezone.utc).isoformat()
def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return None
def atomic(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def power():
    text=subprocess.check_output(['/usr/sbin/nvpmodel','-q'],text=True,timeout=10).strip()
    if text!='NV Power Mode: MAXN\n0':raise RuntimeError('POWER_MODE_STOP: '+text)
    return text


def lock_held():
    with open('/tmp/dasip_round2_gpu0.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return True
        fcntl.flock(lock,fcntl.LOCK_UN)
    return False


def rows(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def check_job(job,folder,expected_frames,ec=False):
    summary=read(folder/'summary.json');assert summary, str(folder)
    cfg=read(folder/'config.json')
    assert summary['condition']==job['condition'] and summary['seed']==job['seed']
    assert summary['frames']==expected_frames and summary['finite_parameters_and_poses'] and not summary['quality']
    assert summary['full_sequence']==(not job.get('frames',0))
    assert summary['warmup_frames']==20
    assert summary['initial_power']==summary['final_power']=='NV Power Mode: MAXN\n0'
    assert cfg['HashGrid' if ec else 'grid']['hash_size']==job.get('hash_size',16)
    assert cfg['keyframe']['kf_number_bg']==.05 if ec else cfg['mapping']['n_pixels']==.05
    assert cfg['round2']['seed']==job['seed'] and cfg['round2']['condition']==job['condition']
    assert cfg['round2']['engineering'] is False
    grid=cfg['HashGrid' if ec else 'grid']
    assert grid['hash']==('Morton' if job['condition'] in ('MC','MC_DP') else 'CoherentPrime')
    assert grid['morton_sort']==(job['condition'] in ('MC','MC_DP'))
    if 'datadir' in job:
        assert Path(cfg['data']['input_folder' if ec else 'datadir']).resolve()==Path(job['datadir']).resolve()
    expected_manifest=ROOT/'EC_RUNTIME_MANIFEST.json' if ec else RUNTIME/'RUNTIME_MANIFEST.json'
    assert (folder/'source_manifest.json').read_bytes()==expected_manifest.read_bytes()
    events=rows(folder/'iterations.csv');calls=rows(folder/'calls.csv')
    stage_iters={'tracking':cfg['tracking']['iters' if ec else 'iter'],'bundle_adjustment':cfg['mapping']['iters']}
    for stage,iters in stage_iters.items():
        frame_ids=set(range(1,expected_frames)) if stage=='tracking' else set(range(cfg['mapping']['every_frame' if ec else 'map_every'],expected_frames,cfg['mapping']['every_frame' if ec else 'map_every']))
        if ec and stage=='bundle_adjustment':frame_ids.add(expected_frames-1)
        selected=[r for r in events if r['iteration_type']==stage]
        assert len(selected)==len(frame_ids)*iters,(stage,len(selected),len(frame_ids)*iters)
        assert {(int(r['frame_id']),int(r['iteration_id'])) for r in selected}=={(f,i) for f in frame_ids for i in range(iters)}
        assert all(all(math.isfinite(float(r[k])) and float(r[k])>=0 for k in ['forward_cuda_ms','backward_cuda_ms','full_iteration_cuda_ms']) for r in selected)
    if not ec:
        for r in rows(folder/'queries.csv'):
            a,d=int(r['actual_queries']),int(r['dense_queries']);assert 0<a<=d
            iters=stage_iters[r['stage']];i=int(r['iteration'])
            if job['condition'] in ('BASE','MC') or i in (0,iters-1) or i%3==0:assert a==d
    assert all(math.isfinite(float(r['wall_ms'])) and math.isfinite(float(r['cuda_ms'])) for r in calls)
    return summary


def update(state):
    state['gpu_child_pids']=[]
    active_pid=state.get('active_parent_child_pid')
    if active_pid:
        if any('run_ec_round2.py' in s for s in state.get('active_command',[])):
            state['gpu_child_pids']=[active_pid]
        else:
            for p in Path('/proc').iterdir():
                if not p.name.isdigit():continue
                try:
                    command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                    parent=next(int(line.split()[1]) for line in (p/'status').read_text().splitlines() if line.startswith('PPid:'))
                    if parent==active_pid and 'run_coslam_round2.py' in command:state['gpu_child_pids'].append(int(p.name))
                except (OSError,StopIteration,ValueError):pass
    for block in state['blocks']:
        for job in block['jobs']:
            folder=Path(job['output'])
            if (folder/'config.json').exists():job['actual_config_sha256']=digest(folder/'config.json')
            launch=read(folder.parent/(job['id']+'.launch.json'))
            if launch:job['actual_command']=launch
            if job['status'] in ('done','blocked','deferred'):continue
            done=read(folder.parent/(job['id']+'.done.json'))
            if done:
                try:
                    summary=check_job(job,folder,job['expected_frames'],job['system']=='EC-SLAM')
                    job.update(status='done',elapsed_seconds=done.get('elapsed_seconds',summary['slam_work_wall_seconds']),summary=summary)
                except Exception as error:
                    job.update(status='blocked',reason='Completed artifact failed reconciliation: '+repr(error))
            elif (folder.parent/(job['id']+'.launch.json')).exists() or (job.get('pid') and Path('/proc',str(job['pid'])).exists()):
                job['status']='running'
                p=folder/'iterations.csv'
                if p.exists():
                    try:job['latest_logged_frame']=max(int(r['frame_id']) for r in rows(p))
                    except (ValueError,KeyError):pass
    state['updated_utc']=now()
    state['ALL_REQUIRED_MAXN_COMPLETE']=all(j['status']=='done' for b in state['blocks'] for j in b['jobs'])
    state['counts']={s:sum(j['status']==s for b in state['blocks'] for j in b['jobs']) for s in ['done','running','queued','deferred','blocked']}
    atomic(STATE,state)
    text='# Orin MAXN paper-run agenda\n\nUpdated: '+state['updated_utc']+' (UTC).\n\n'
    text+='Scope: 28 required full paper-candidate runs; four Co diagnostics and two conditional EC diagnostics are separate. No quality/mesh scoring. Current clock policy retained by author instruction.\n\n'
    text+='Planning anchor: '+state['planning_anchor_utc']+'; eight hours with 25% reserve gives admission cutoff '+state['admission_deadline_utc']+'. Complete paired blocks are admitted before inspecting gains. Running sequences are not killed for ETA.\n\n'
    text+='Actual counts: '+str(state['counts'])+'. **ALL_REQUIRED_MAXN_COMPLETE = '+str(state['ALL_REQUIRED_MAXN_COMPLETE']).upper()+'**.\n\n'
    text+='| Block | System / scene | Condition | Capacity | Seed | Frames | Mode | Status | Elapsed s / ETA |\n|---|---|---|---:|---:|---:|---|---|---|\n'
    for b in state['blocks']:
        for j in b['jobs']:
            text+=f"| {b['id']} | {j['system']} / {j['scene']} | {j['condition']} | {j.get('hash_size',16)} | {j['seed']} | {j['expected_frames']} | MAXN | {j['status']} | {j.get('elapsed_seconds','')} / {j.get('eta_utc','')} |\n"
        if b.get('note'):text+='\n'+b['id']+': '+b['note']+'\n\n'
    text+='\nExact commands, source/config identities, live PIDs and deferral/failure reasons are in [PAPER_NIGHT_STATUS.json](PAPER_NIGHT_STATUS.json).\n\n'
    text+='Pending MAXN work must be reviewed before a manual 15 W switch. No automatic power/clock changes or next-series launches are authorized. DP remains an unaccepted candidate; retain negative timings. EC timing scopes overlap on a shared stream and cannot be added to infer system FPS.\n'
    AGENDA.write_text(text)


def verify_freeze():
    freeze=read(ROOT/'PAPER_NIGHT_FREEZE.json')
    for name,expected in freeze['files'].items():assert digest(ROOT/name)==expected,'Frozen campaign source differs: '+name
    for name,expected in read(CO/'SHA256SUMS.json').items():assert digest(CO/name)==expected,'Runtime v1 changed: '+name


def wait_idle():
    deadline=time.monotonic()+120
    while True:
        power()
        devices=[str(p) for pattern in ['nvhost*gpu*','nvmap','nvidia*'] for p in Path('/dev').glob(pattern) if not p.is_dir()]
        r=subprocess.run(['fuser','-v',*devices],capture_output=True,text=True)
        if not lock_held() and r.returncode==1:return
        if time.monotonic()>deadline:raise RuntimeError('GPU gap unavailable; device users: '+r.stdout+r.stderr)
        time.sleep(5)


def run_process(command,log,state,job=None,ec=False):
    wait_idle();verify_freeze();power()
    lock=open('/tmp/dasip_round2_gpu0.lock','a') if ec else None
    if lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        with log.open('x') as stream:
            child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),start_new_session=True)
            start=time.time();state['active_parent_child_pid']=child.pid;state['active_command']=command;state['active_log']=str(log)
            if job:job.update(pid=child.pid,status='running',started_utc=now())
            update(state)
            while child.poll() is None:
                try:power()
                except Exception:
                    os.killpg(child.pid,signal.SIGTERM);child.wait();raise
                update(state);time.sleep(15)
            rc=child.returncode
            if job:job['elapsed_seconds']=time.time()-start
            state.pop('active_parent_child_pid',None)
            power()
            if rc:raise RuntimeError(f'Process return code {rc}; exact log: {log}')
    finally:
        if lock:lock.close()


def report_timings(state):
    table=[]
    for b in state['blocks']:
        grouped={}
        for j in b['jobs']:
            if j['status']!='done':continue
            summary=j.get('summary') or read(Path(j['output'])/'summary.json')
            for stage in (['tracking_frame','bundle_adjustment'] if j['system']=='EC-SLAM' else ['tracking','bundle_adjustment']):
                v=summary.get(stage,{})
                for metric in ['mean_wall_ms','mean_cuda_ms','all_call_mean_wall_ms']:
                    if v.get(metric) is not None:grouped.setdefault((j['system'],j['condition'],j.get('hash_size',16),stage,metric),[]).append((j['seed'],v[metric]))
        for key,values in grouped.items():
            table.append(dict(block=b['id'],system=key[0],condition=key[1],capacity=key[2],stage=key[3],metric=key[4],seeds=[v[0] for v in values],mean=statistics.mean(v[1] for v in values),sample_sd=statistics.stdev(v[1] for v in values) if len(values)>1 else None))
    atomic(ROOT/'TIMING_SUMMARIES.json',table)
    text='# MAXN paired timing summaries\n\nMilliseconds; primary excludes frames 0–19, all-call means separate. Sample SD uses n−1; a single seed has no SD. No quality claim or additive EC FPS.\n\n| Block | System | Condition | Capacity | Stage | Metric | Seeds | Mean | Sample SD |\n|---|---|---|---:|---|---|---|---:|---:|\n'
    for r in table:text+='| '+' | '.join(str(r[k]) for k in ['block','system','condition','capacity','stage','metric','seeds','mean','sample_sd'])+' |\n'
    (ROOT/'TIMING_REPORT.md').write_text(text)


def run_co(block,state):
    directory=Path(block['queue_output'])
    command=[PY,str(CO/'run_manifest_queue.py'),'--runtime',str(RUNTIME),'--manifest',block['manifest'],'--output',str(directory),'--power-mode','MAXN']
    block['command']=command
    run_process(command,RESULTS/(block['id']+'_queue.log'),state)
    for job in block['jobs']:
        check_job(job,Path(job['output']),job['expected_frames'])
    update(state)


def run_ec(block,state):
    evidence=[]
    for condition in ['BASE','MC']:
        name='EC_office0_'+condition+'_diagnostic'
        folder=RESULTS/name
        command=[PY,str(ROOT/'run_ec_round2.py'),'--condition',condition,'--seed','0','--frames','50','--datadir',state['ec_data'],'--output',str(folder),'--co-runtime',str(RUNTIME)]
        run_process(command,RESULTS/(name+'.log'),state,ec=True)
        evidence.append(check_job(dict(condition=condition,seed=0,frames=50),folder,50,True))
    atomic(ROOT/'EC_PREFLIGHT_VALIDATION.json',{'status':'PASS','diagnostics':evidence,'native_AST_audit':read(ROOT/'EC_NATIVE_CHANGE_AUDIT.json')})
    block['preflight']='PASS'
    names=read(ROOT/'EC_RUNTIME_MANIFEST.json')['files']
    with zipfile.ZipFile(ROOT/'EC_VALIDATED_SOURCE_ONLY.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:archive.write(ROOT/name,arcname=name)
        for name in ['EC_RUNTIME_MANIFEST.json','EC_NATIVE_CHANGE_AUDIT.json','EC_NATIVE_CHANGES.diff','EC_PREFLIGHT_VALIDATION.json']:archive.write(ROOT/name,arcname=name)
    # Conservative full-sequence forecast from both diagnostic elapsed times.
    ec_estimate=max(r['slam_work_wall_seconds']/50*2000*2 for r in evidence)
    if not admit(block,state,ec_estimate):return
    for job in block['jobs']:
        command=job['command'];folder=Path(job['output'])
        atomic(folder.parent/(job['id']+'.launch.json'),command)
        run_process(command,RESULTS/(job['id']+'.log'),state,job,True)
        summary=check_job(job,folder,2000,True)
        atomic(folder.parent/(job['id']+'.done.json'),{'id':job['id'],'returncode':0,'elapsed_seconds':job['elapsed_seconds']})
        job.update(status='done',summary=summary);update(state);report_timings(state)


def admit(block,state,per_job):
    deadline=datetime.fromisoformat(state['admission_deadline_utc'])
    finish=datetime.now(timezone.utc)+timedelta(seconds=per_job*len(block['jobs']))
    block['forecast_seconds']=per_job*len(block['jobs']);block['forecast_finish_utc']=finish.isoformat()
    for j in block['jobs']:j['eta_utc']=finish.isoformat()
    if finish>deadline:
        block['note']='Deferred before inspecting gains: complete paired block forecast exceeds six-hour usable window (eight hours with 25% reserve).'
        for job in block['jobs']:job.update(status='deferred',reason=block['note'])
        update(state);return False
    return True


def package(state):
    update(state);report_timings(state)
    atomic(ROOT/'PAPER_NIGHT_COMPLETE.json',{'updated_utc':now(),'ALL_REQUIRED_MAXN_COMPLETE':state['ALL_REQUIRED_MAXN_COMPLETE'],'counts':state['counts'],'status':'Planned admitted chain terminal; deferred/blocked coverage remains explicit','power':power()})
    archive_path=ROOT.parents[1]/'ORIN_MAXN_RESULTS.zip'
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock',)]
    # Include exact reused outputs/environment, not the original repository or data.
    reused=[p for d in [CO/'results/full_maxn',CO/'results/diagnostic_maxn',CO/'run_records'] for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    manifest={}
    with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths+reused:
            name=('paper_night/'+p.relative_to(ROOT).as_posix()) if p.is_relative_to(ROOT) else 'runtime_v1_reused/'+p.relative_to(CO).as_posix()
            data=p.read_bytes();manifest[name]=hashlib.sha256(data).hexdigest();archive.writestr(name,data)
        archive.writestr('RETURN_MANIFEST.json',json.dumps({'ALL_REQUIRED_MAXN_COMPLETE':state['ALL_REQUIRED_MAXN_COMPLETE'],'files':manifest},indent=2)+'\n')
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name,h in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==h
    archive_path.with_suffix('.zip.sha256').write_text(digest(archive_path)+'  '+archive_path.name+'\n')


def main():
    RESULTS.mkdir(exist_ok=True)
    state=read(PLAN);assert state
    if STATE.exists():raise RuntimeError('Existing campaign status preserved; no automatic rerun')
    with (ROOT/'paper_night_parent.lock').open('a') as parent_lock:
        fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state.update(parent_pid=os.getpid(),started_utc=now(),session='dasip_r2_paper_night_maxn')
        verify_freeze();power();update(state)
        while not (ORIGINAL/'queue_complete.json').exists():
            progress=read(ORIGINAL/'progress.json')
            if progress and any(j.get('returncode',0) for j in progress['jobs']):raise RuntimeError('Existing native O1 comparison failed; inspect original outputs')
            if not lock_held():raise RuntimeError('Existing queue stopped without completion')
            power();update(state);time.sleep(15)
        # Old launcher still performs its CPU-only validation/packaging.
        deadline=time.monotonic()+180
        while not read(CO/'run_records/MAXN_FINAL_VALIDATION.json'):
            if time.monotonic()>deadline:raise RuntimeError('Existing queue final validation missing')
            update(state);time.sleep(5)
        assert read(CO/'run_records/MAXN_FINAL_VALIDATION.json')['validation']=='PASS'
        update(state)
        for block in state['blocks']:
            if block['id']=='O1_existing':continue
            if block.get('readiness')=='deferred':update(state);continue
            try:
                verify_freeze();power()
                if block['id']=='O2':run_ec(block,state)
                else:
                    native=[j.get('elapsed_seconds',0) for b in state['blocks'] for j in b['jobs'] if j['system']=='Co-SLAM' and j['status']=='done' and j['scene']=='office0']
                    base=max(native or [780])
                    per_job=base*(2 if block['id']=='O3' else 1.6 if block['id']=='O1_additional' else .75)
                    if admit(block,state,per_job):run_co(block,state)
                report_timings(state)
            except Exception:
                error=traceback.format_exc();block['note']='Affected block stopped/blocked; exact evidence in '+block['id']+'_FAILURE.json'
                atomic(ROOT/(block['id']+'_FAILURE.json'),{'utc_time':now(),'error':error})
                for job in block['jobs']:
                    if job['status']!='done':job.update(status='blocked',reason=block['note'])
                update(state);report_timings(state)
                # A mode change stops all work; other failures leave independent
                # frozen, ready comparisons eligible under the declared agenda.
                power()
        package(state)
        print('PAPER_NIGHT_TERMINAL',json.dumps(state['counts']),state['ALL_REQUIRED_MAXN_COMPLETE'],flush=True)


if __name__=='__main__':main()
