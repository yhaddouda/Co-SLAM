"""Lightweight detached gate. Imports no CUDA or data preparation libraries."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
V4=ROOT.parent/'orin_continuous_v4'
GPU_LOCK='/tmp/dasip_round2_gpu0.lock'
FOLLOWER_LOCK='/tmp/dasip_round2_after_v4_maxn_v1.lock'
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def identity(pid):
    try:
        p=Path('/proc',str(pid));tail=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start_ticks=int(tail[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),exe=os.readlink(p/'exe'),cmdline=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1],cwd=os.readlink(p/'cwd'))
    except (OSError,ValueError):return None
def maxn():
    p=subprocess.check_output(['/usr/sbin/nvpmodel','-q'],text=True,timeout=10).strip()
    assert p=='NV Power Mode: MAXN\n0','MAXN required; no settings changed: '+p
    return p
def normal_state(s,end,expected):
    if s.get('phase')!='TERMINAL':return False
    assert end and end.get('major_admitted_series_terminal') is True,'Terminal state lacks normal completion marker'
    assert not s.get('active_child_pid') and not s.get('active_gpu_child_pids'),'Active child remains'
    assert not s.get('stop_reason'),'Stopped state is not normal exhaustion'
    jobs={j['id']:j for j in s['jobs']}
    assert not any(j['status'] in ['queued','running'] for j in jobs.values()),'Ready backlog remains'
    for key in expected:
        assert key in jobs and jobs[key]['status'] in ['done','reused'],'Expected ready identity not completed: '+key
    assert s['counts']==end['counts'],'Terminal counts and completion marker differ'
    assert len(expected)==sum(jobs[k]['status'] in ['done','reused'] for k in expected)
    return True
def idle():
    devices=sorted({str(p) for pattern in ['nvhost*gpu*','nvmap','nvidia*','nvgpu/**/*'] for p in Path('/dev').glob(pattern) if not p.is_dir()})
    assert devices,'Host GPU device nodes not visible'
    f=subprocess.run(['fuser','-v',*devices],capture_output=True,text=True,timeout=15)
    load=int(Path('/sys/devices/platform/17000000.gpu/load').read_text())
    assert f.returncode==1 and not f.stdout.strip() and not f.stderr.strip() and load==0,'GPU workload conflict: '+f.stdout+f.stderr+' load='+str(load)
    # Device ownership is authoritative for all CUDA programs, not only known SLAM names.
    programs=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
            if any(Path(a).name in ['run_coslam_round2.py','run_ec_round2.py','run_manifest_queue.py','run_queue_inherited_lock.py','ncu','nsys'] for a in args):programs.append(dict(pid=int(p.name),args=args))
        except OSError:pass
    assert not programs,'CUDA-capable workload launcher still present: '+repr(programs)
    return dict(utc=now(),devices=devices,fuser_returncode=f.returncode,device_users=f.stdout,gpu_load_raw=load,active_programs=programs)
def verify_install():
    for name,h in read(ROOT/'FOLLOWER_FREEZE.json')['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,'Installed source changed: '+name
def verify_completed(expected,s):
    """After exclusive ownership only: verify original identities/completion evidence."""
    import csv
    import math
    jobs={j['id']:j for j in s['jobs']};verified=[]
    for key,e in expected.items():
        j=jobs[key]
        for k in ['system','dataset','scene','condition','seed','hash_size','expected_frames','mode','datadir']:assert j[k]==e[k],(key,k)
        folder=Path(j['output']);summary=read(folder/'summary.json');cfg=read(folder/'config.json');ec=j['system']=='EC-SLAM'
        assert summary['full_sequence'] and not summary['diagnostic'] and summary['finite_parameters_and_poses'] and not summary['quality']
        assert summary['frames']==j['expected_frames'] and summary['condition']==j['condition'] and summary['seed']==j['seed'] and summary['scene']==j['scene']
        assert summary['initial_power']==summary['final_power']=='NV Power Mode: MAXN\n0' and summary['warmup_frames']==20
        assert cfg['round2']['engineering'] is False and cfg['HashGrid' if ec else 'grid']['hash_size']==j['hash_size']
        assert Path(cfg['data']['input_folder' if ec else 'datadir']).resolve()==Path(j['datadir']).resolve()
        digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if j['status']=='reused':
            provenance=j['reuse_provenance'];assert digest(folder/'config.json')==provenance['effective_config_sha256'] and digest(folder/'source_manifest.json')==provenance['source_manifest_sha256']
            done=folder.with_name(provenance['original_job']+'.done.json');assert digest(done)==provenance['completion_sha256']
        else:
            done=V4/'controller_records'/(key+'.controller_done.json')
            assert digest(folder/'config.json')==j['actual_config_sha256'] and digest(folder/'source_manifest.json')==j['actual_source_manifest_sha256']
        assert read(done)['returncode']==0
        with (folder/'calls.csv').open() as f:calls=list(csv.DictReader(f))
        n=j['expected_frames'];step=cfg['mapping']['every_frame' if ec else 'map_every']
        for stage,ids in [('tracking_frame' if ec else 'tracking',set(range(n)) if ec else set(range(1,n))),('bundle_adjustment',set(range(step,n,step))|({n-1} if ec else set()))]:
            selected=[x for x in calls if x['stage']==stage];assert len(selected)==len(ids) and {int(x['frame_id']) for x in selected}==ids
        assert all(math.isfinite(float(x[k])) for x in calls for k in ['wall_ms','cuda_ms'])
        assert (folder/'iterations.csv').is_file()
        verified.append(dict(id=key,output=str(folder),completion=str(done),config_sha256=digest(folder/'config.json'),source_manifest_sha256=digest(folder/'source_manifest.json')))
    return verified
def main():
    with open(FOLLOWER_LOCK,'a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=read(ROOT/'FOLLOWER_CONFIG.json');expected=config['expected_ready'];verify_install()
        me=identity(os.getpid());save(ROOT/'FOLLOWER_IDENTITY.json',me)
        print('FOLLOWER_WAITING',now(),'expected_ready',len(expected),'parent',config['v4_parent_identity'],flush=True)
        while True:
            maxn();s=read(V4/'CONTINUOUS_STATUS.json');end=read(V4/'CONTINUOUS_COMPLETE.json') if (V4/'CONTINUOUS_COMPLETE.json').exists() else None
            assert not (V4/'CONTINUOUS_STOPPED.json').exists(),'V4 stopped abnormally; continuation not admitted'
            current=identity(config['v4_parent_identity']['pid'])
            if s.get('phase')=='STOPPED':raise RuntimeError('V4 reports STOPPED')
            complete=normal_state(s,end,expected)
            save(ROOT/'FOLLOWER_STATUS.json',dict(updated_utc=now(),phase='WAITING_FOR_ENTIRE_V4_READY_BACKLOG',follower_pid=os.getpid(),parent_identity_matches=current==config['v4_parent_identity'],v4_counts=s['counts'],expected_ready_identities=len(expected),normal_terminal_state_seen=complete,trigger='Normal terminal completion of all expected identities, no active child, parent exited, exclusive shared workload lock and fresh idle; no clock-time or ALL_REQUIRED gate'))
            if current!=config['v4_parent_identity']:
                assert complete,'Original v4 parent exited/reused without verified normal exhaustion'
                break
            time.sleep(config['poll_seconds'])
        # Parent has exited normally. Acquire both controller and shared workload ownership.
        with open('/tmp/dasip_round2_continuous_v4.lock','a') as old_controller,open(GPU_LOCK,'a') as workload:
            fcntl.flock(old_controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(workload,fcntl.LOCK_EX|fcntl.LOCK_NB)
            maxn();verify_install();samples=[]
            for _ in range(3):samples.append(idle());time.sleep(2)
            s=read(V4/'CONTINUOUS_STATUS.json');end=read(V4/'CONTINUOUS_COMPLETE.json');assert normal_state(s,end,expected)
            assert 'CONTINUOUS_MAJOR_SERIES_TERMINAL' in (V4/'continuous_parent.log').read_text(),'No normal parent terminal log marker'
            for path,digest in read(V4/'CONTINUOUS_FREEZE.json')['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'V4 frozen source changed: '+path
            proof=verify_completed(expected,s)
            transition=dict(utc=now(),phase='EXCLUSIVE_PREPARATION',parent_identity=config['v4_parent_identity'],parent_exited=True,v4_completion=end,verified_ready_identities=proof,idle_samples=samples,workload_lock=GPU_LOCK,workload_lock_fd=workload.fileno(),ALL_REQUIRED_MAXN_COMPLETE=s['ALL_REQUIRED_MAXN_COMPLETE'])
            save(ROOT/'NORMAL_EXHAUSTION_TRANSITION.json',transition)
            save(ROOT/'FOLLOWER_STATUS.json',dict(updated_utc=now(),phase='PREPARING',follower_pid=os.getpid(),trigger_satisfied=True,worker_command=config['worker_command']))
            print('NORMAL_EXHAUSTION_VERIFIED_PREPARATION_START',now(),flush=True)
            command=config['worker_command']+['--workload-lock-fd',str(workload.fileno())]
            process=subprocess.Popen(command,pass_fds=(workload.fileno(),))
            save(ROOT/'WORKER_LAUNCH.json',dict(utc=now(),pid=process.pid,command=command))
            rc=process.wait();save(ROOT/'FOLLOWER_WORKER_EXIT.json',dict(utc=now(),returncode=rc))
            assert rc==0,'Post-v4 worker stopped; see preserved worker status/log'
            save(ROOT/'FOLLOWER_STATUS.json',dict(updated_utc=now(),phase='COMPLETE',follower_pid=os.getpid(),worker_returncode=rc))
            print('POST_V4_ADMITTED_SERIES_TERMINAL',now(),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(ROOT/'FOLLOWER_ERROR.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),preparation_not_authorized_by_abnormal_exit=True))
        print(traceback.format_exc(),flush=True);raise
