"""Durable dependency chain: normal ScanNet -> old DP export -> audited 88-job series."""
import argparse
import fcntl
import importlib.util
import subprocess
import sys
import time
import traceback
import zipfile
from common import *


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def normal_scannet(cfg):
    s=read(SCAN/'SCANNET_STATUS.json')
    if s['phase']!='COMPLETE_ADMITTED_SCANNET_SERIES':return False,'ScanNet series is not terminal'
    marker=SCAN/'SCANNET_COMPLETE.json'
    if not marker.exists() or not read(marker).get('major_admitted_series_terminal'):return False,'No normal completion marker'
    if (SCAN/'STOPPED.json').exists():return False,'ScanNet abnormal-stop evidence exists'
    if s.get('active_child_pid'):return False,'ScanNet still records an active child'
    jobs={j['id']:j for j in s['jobs']}
    for j in cfg['expected_scannet_jobs']:
        if jobs.get(j['id'],{}).get('status') not in ['done','reused']:return False,'Expected ScanNet identity unfinished: '+j['id']
        for k in ['dataset','scene','condition','seed','hash_size','expected_native_frames']:
            if jobs[j['id']][k]!=j[k]:return False,'ScanNet identity changed: '+j['id']+'/'+k
        out=Path(jobs[j['id']]['output'])
        try:
            a=read(out/'summary.json');done=read(out.parent/(out.name+'.done.json'))
            assert done['returncode']==0 and (out.parent/'queue_complete.json').is_file()
            assert a['frames']==j['expected_native_frames'] and a['full_sequence'] and not a['diagnostic'] and a['finite_parameters_and_poses']
            assert a['condition']==j['condition'] and a['seed']==j['seed'] and a['scene']==j['scene']
            assert a['initial_power']==a['final_power']=='NV Power Mode: MAXN\n0' and a['warmup_frames']==20 and not a['quality']
        except Exception as e:return False,'ScanNet completion evidence invalid: '+j['id']+' '+repr(e)
    if identity(cfg['scannet_parent_identity']['pid'])==cfg['scannet_parent_identity']:return False,'Normal ScanNet parent has not exited/released ownership'
    return True,'All 24 normal full identities recorded complete; parent exited; no child'


def materialize_bundle(cfg):
    """No archive extraction occurs until exclusive idle after the dependency gates."""
    target=ROOT/'bundle'
    if not target.exists():
        assert sha(cfg['bundle_zip'])==cfg['bundle_sha256']
        target.mkdir()
        with zipfile.ZipFile(cfg['bundle_zip']) as z:
            assert z.testzip() is None
            for entry in z.infolist():
                path=(target/entry.filename).resolve();assert path.is_relative_to(target.resolve()) and not entry.is_dir()
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(entry))
    subprocess.run([sys.executable,'-B',str(target/'validate_bundle.py')],check=True)


def archive_verified(receipt):
    assert receipt['status']=='EXPORTED_AND_VERIFIED' and receipt['full_runs']==8 and receipt['diagnostics']==2 and receipt['csv_validation']=='PASS'
    p=Path(receipt['archive']);assert sha(p)==receipt['sha256']
    assert p.with_suffix('.zip.sha256').read_text().split()[0]==receipt['sha256']
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None;m=json.loads(z.read('RETURN_MANIFEST.json'))
        assert m['full_runs']==8 and m['diagnostics']==2 and m['runtime_manifest_sha256']=='9786e843d6856dcc443fb08ebb9646a5d759384e19ab3456552b51e2281e616d'
        for name,h in m['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    return dict(utc=now(),archive=str(p),sha256=receipt['sha256'],entries=len(m['files']),status='PASS')


def wake(cfg,message,filename):
    """Existing same-conversation wake path; this function never sends email."""
    if (ROOT/filename).exists():return
    try:
        p=subprocess.run([cfg['notification']['codex_executable'],'queue','--thread',cfg['notification']['thread_id'],'--message',message],capture_output=True,text=True,timeout=60)
        receipt=dict(utc=now(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
        save(ROOT/(filename if p.returncode==0 else filename+'.error.json'),receipt)
    except Exception as e:save(ROOT/(filename+'.error.json'),dict(utc=now(),error=repr(e)))


def recover_export(cfg):
    """One archival repair attempt at idle; never repeat an experiment or relax audits."""
    if (ROOT/'EXPORT_RECOVERY_ATTEMPT.json').exists():return False
    save(ROOT/'EXPORT_RECOVERY_ATTEMPT.json',dict(utc=now(),reason='Exporter exited without a valid successful archive/receipt',CUDA_work=False))
    backups=ROOT/'preserved_export_failure'/now().replace(':','-');backups.mkdir(parents=True)
    ecfg=read(EXPORT/'EXPORT_FOLLOWER_CONFIG.json');target=Path(ecfg['archive'])
    for p in [EXPORT/'EXPORT_STOPPED.json',EXPORT/'EXPORT_STATUS.json',EXPORT/'export_follower.log',EXPORT/'EXPORT_RECEIPT.json',target,target.with_suffix('.zip.tmp'),target.with_suffix('.zip.sha256')]:
        if not p.exists():continue
        if p.suffix in ['.zip','.tmp'] or p.name=='EXPORT_RECEIPT.json' or p.name.endswith('.zip.sha256'):
            p.rename(backups/p.name)
        else:
            import shutil;shutil.copyfile(p,backups/p.name)
    try:
        mod=load_module('existing_dp_archival_recovery',EXPORT/'export_when_idle.py')
        mod.export(read(EXPORT/'ORIN_EXISTING_DP_INVENTORY.json'),ecfg)
        verified=archive_verified(read(EXPORT/'EXPORT_RECEIPT.json'))
        save(ROOT/'EXPORT_RECOVERY_RESULT.json',dict(status='PASS',verification=verified));return True
    except Exception as e:
        save(ROOT/'EXPORT_RECOVERY_RESULT.json',dict(status='FAIL',error=repr(e),traceback=traceback.format_exc()))
        wake(cfg,'The authorized MAXN DP continuation is waiting on an existing-results export orchestration error. ScanNet has ended and the existing exporter exited. Read '+str(ROOT/'EXPORT_RECOVERY_RESULT.json')+' and preserve its exact failure. Resolve the archival error at verified idle, without rerunning experiments or loosening scientific checks, then let the durable follower continue the authorized 88-job plan. Do not email for this isolated error or change power/clocks.','EXPORT_REPAIR_WAKE.json')
        return False


def final_notification(cfg):
    end=read(ROOT/'DP_COMPLETION_COMPLETE.json')
    if end.get('new_full_completed',0)<4:return
    for _ in range(3):
        try:sample=idle()
        except AssertionError:return
        save(ROOT/'EMAIL_CAMPAIGN_STATUS.json',dict(utc=now(),major_series_terminal=True,latest_fresh_idle=sample,scope='Expanded native/ENG 196 plus pruning/composition 96; no intermediate ScanNet/export email'))
        time.sleep(15)
    message=('The user-authorized expanded Orin MAXN DP/MC_DP major series has reached its recorded end. Read '+str(ROOT/'DP_COMPLETION_STATUS.json')+', '+str(ROOT/'DP_COMPLETION_COMPLETE.json')+', '+str(ROOT/'TIMING_REPORT.md')+', '+str(ROOT/'UNIQUE_RUN_INDEX.json')+' and '+str(ROOT/'RETURN_BUNDLE_VERIFIED.json')+'. Verify the new incremental archive and the prior existing-eight DP archive receipt '+str(EXPORT/'EXPORT_RECEIPT.json')+'. Recheck fresh host GPU idle and experiment process status. State ALL_REQUIRED_MAXN_COMPLETE truthfully against the 292-slot envelope (196 native/Co engineering + 96 pruning/composition); list every blocked/unresolved identity and EC ENG as excluded. Only for this completed major series with fresh idle send the authorized concise decision email to youssefhaddouda233@gmail.com via connected Gmail. Check '+str(ROOT/'EMAIL_SENT.json')+' and Gmail Sent to avoid duplicates, and save a real sent message ID/receipt. Earlier v4 email 1a158b5953892a7f covered the prior 172 slots, not this expanded series. No individual-run, intermediate-export or isolated-failure email. No next series, power/clock changes, meshes/ATE or automatic 15 W. A 15 W template is prepared but requires the author manual switch and instruction.')
    wake(cfg,message,'EMAIL_WAKE_RECEIPT.json')


def main():
    cfg=read(ROOT/'INSTALLATION_CONFIG.json')
    with open('/tmp/dasip_round2_dp_completion_maxn_v1.lock','a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(ROOT/'FOLLOWER_IDENTITY.json',identity(os.getpid()))
        s=read(ROOT/'DP_COMPLETION_STATUS.json');s['follower_pid']=os.getpid();update(s)
        print('DP_FOLLOWER_INSTALLED_WAITING',now(),'ScanNet',cfg['scannet_parent_identity']['pid'],'exporter',cfg['exporter_identity']['pid'],flush=True)
        while True:
            support.maxn()
            ok,reason=normal_scannet(cfg)
            if not ok:
                prior=read(SCAN/'SCANNET_STATUS.json')
                if prior['phase'] in ['STOPPED','COMPLETE_ADMITTED_SCANNET_SERIES'] and identity(cfg['scannet_parent_identity']['pid'])!=cfg['scannet_parent_identity']:
                    save(ROOT/'SCANNET_GATE_UNRESOLVED.json',dict(utc=now(),reason=reason,scan_counts=prior['counts']))
                    wake(cfg,'The authorized DP continuation is waiting for normal completion evidence for all 24 ScanNet identities. Inspect '+str(ROOT/'SCANNET_GATE_UNRESOLVED.json')+' and preserve exact ScanNet failures. Resolve the missing authorized work without repeating completed experiments or changing scientific settings; the DP follower must retain its normal-ScanNet and existing-export gates. No email for this isolated blockage and no power/clock change.','SCANNET_GATE_REPAIR_WAKE.json')
                s.update(phase='WAITING_FOR_NORMAL_SCANNET_COMPLETION',dependency_reason=reason);update(s);time.sleep(30);continue
            exporter=identity(cfg['exporter_identity']['pid'])
            if exporter==cfg['exporter_identity']:
                s.update(phase='WAITING_FOR_EXISTING_DP_EXPORTER_TO_FINISH',dependency_reason='Original export follower is still alive; do not take ownership ahead of it');update(s);time.sleep(30);continue
            if not (ROOT/'NOTIFICATION_HANDOFF.json').exists():
                s.update(phase='WAITING_FOR_NOTIFICATION_HANDOFF');update(s);time.sleep(30);continue
            with open('/tmp/dasip_round2_gpu0.lock','a') as workload:
                try:fcntl.flock(workload,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    s.update(phase='WAITING_FOR_RELEASED_EXCLUSIVE_OWNERSHIP');update(s);time.sleep(30);continue
                try:
                    samples=[]
                    for _ in range(3):samples.append(idle());time.sleep(2)
                except AssertionError as e:
                    fcntl.flock(workload,fcntl.LOCK_UN);s.update(phase='WAITING_FOR_FRESH_IDLE',dependency_reason=str(e));update(s);time.sleep(30);continue
                ok,reason=normal_scannet(cfg);assert ok,reason
                for path,h in read(ROOT/'INSTALLATION_FREEZE.json')['files'].items():assert sha(path)==h,'Installed orchestration source changed: '+path
                try:verified=archive_verified(read(EXPORT/'EXPORT_RECEIPT.json'))
                except Exception:
                    s.update(phase='RESOLVING_EXISTING_EXPORT_AT_IDLE');update(s)
                    recover_export(cfg)
                    # Always release archival ownership before accepting receipt/handoff for GPU work.
                    fcntl.flock(workload,fcntl.LOCK_UN);s.update(phase='WAITING_FOR_VERIFIED_EXPORT_RECEIPT');update(s);time.sleep(30);continue
                save(ROOT/'EXPORT_DEPENDENCY_VERIFIED.json',verified)
                materialize_bundle(cfg)
                save(ROOT/'DEPENDENCY_GATE_PASSED.json',dict(utc=now(),normal_scannet=True,expected_full_identities=24,scannet_parent_exited=True,
                     exporter_identity=cfg['exporter_identity'],exporter_exited=True,export_verification=verified,exclusive_workload_lock=True,idle_samples=samples,power=support.maxn()))
                command=[sys.executable,'-B','-u',str(ROOT/'run_dp_series.py'),'--workload-lock-fd',str(workload.fileno())]
                with (ROOT/'worker.log').open('x') as log:
                    p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,pass_fds=(workload.fileno(),))
                    save(ROOT/'WORKER_LAUNCH.json',dict(utc=now(),pid=p.pid,command=command,log=str(ROOT/'worker.log')))
                    while p.poll() is None:time.sleep(20)
                    rc=p.wait()
                save(ROOT/'WORKER_EXIT.json',dict(utc=now(),returncode=rc))
                if rc:
                    wake(cfg,'The authorized MAXN DP continuation worker stopped. Inspect '+str(ROOT/'WORKER_STOPPED.json')+' and '+str(ROOT/'worker.log')+'. Preserve exact failures and completed identities; resolve orchestration at idle and continue independently ready authorized work. Do not email for an isolated stoppage, change scientific settings, repeat completed jobs, or change power/clocks.','WORKER_REPAIR_WAKE.json')
                    raise RuntimeError('DP worker stopped; exact evidence preserved and active-conversation repair requested')
                idle();save(ROOT/'FOLLOWER_COMPLETE.json',dict(utc=now(),worker_returncode=rc,major_admitted_series_terminal=True))
                break
        # Workload lock has been released after all timed work and final export.
        final_notification(cfg)


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(ROOT/'FOLLOWER_STOPPED.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),major_admitted_series_terminal=False))
        try:wake(read(ROOT/'INSTALLATION_CONFIG.json'),'The authorized MAXN DP continuation has an orchestration/gate error. Read '+str(ROOT/'FOLLOWER_STOPPED.json')+' and preserve the exact failure. Resolve it and reinstall/continue the durable chain without interrupting a valid timing job, bypassing normal ScanNet/export gates, changing scientific settings or repeating completed experiments. Do not email for this isolated error or change power/clocks.','FOLLOWER_REPAIR_WAKE.json')
        except Exception:pass
        raise
