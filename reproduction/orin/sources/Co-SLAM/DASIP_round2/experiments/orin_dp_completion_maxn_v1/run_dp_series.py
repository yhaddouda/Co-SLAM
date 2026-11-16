"""Boundary audit and serial missing-identity execution. Not a scientific runtime."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import zipfile
from common import *


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def run(argv,log,fd,s):
    support.maxn();idle()
    with Path(log).open('x') as f:
        p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,pass_fds=(fd,))
        s.update(active_child_pid=p.pid,active_log=str(log),active_command=argv);update(s)
        while p.poll() is None: time.sleep(20);update(s)
        rc=p.wait()
    s['active_child_pid']=None;support.maxn()
    for _ in range(15):
        try:idle();break
        except AssertionError:time.sleep(2)
    else:raise RuntimeError('Experiment child/device ownership did not clear')
    update(s);return rc


def data_probe(cfg,expected):
    """Native CPU loader, unchanged settings; never instantiate SLAM or write data."""
    import numpy as np
    sys.path.insert(0,str(CO))
    from datasets.dataset import get_dataset
    ds=get_dataset(cfg);assert len(ds)==expected
    # Native trainskip may address a subset of a larger on-disk sequence.
    if hasattr(ds,'depth_paths'):assert len(ds.depth_paths)>=max(ds.frame_ids)+1
    probes=[]
    for i in sorted({0,expected//2,expected-1}):
        b=ds[i];rgb=b['rgb'].numpy();depth=b['depth'].numpy()
        assert np.isfinite(rgb).all() and np.isfinite(depth).all()
        probes.append(dict(frame=i,rgb_shape=list(rgb.shape),depth_shape=list(depth.shape),pose_finite=bool(np.isfinite(b['c2w'].numpy()).all())))
    return dict(status='PASS',frames=len(ds),native_loader=True,no_data_changes=True,probes=probes)


def prepare_morton():
    import torch
    from torch.utils.cpp_extension import load
    support.maxn();major,minor=torch.cuda.get_device_capability();os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    ext=load(name=f'dasip_round2_morton_{major}{minor}',sources=[str(CO/'model/round2_morton/morton_cuda.cu')],extra_cuda_cflags=['-O3'],verbose=True)
    points=torch.tensor([[0.,0.,0.],[.5,.5,.5],[1.,1.,1.]],device='cuda',dtype=torch.float32)
    assert ext.morton3d_keys(points.contiguous(),128).numel()==3;torch.cuda.synchronize()
    print('MORTON_CACHE_READY',json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,capability=[major,minor],binary=ext.__file__,compiled_before_full_timings=True)),flush=True)


def report_and_package(s):
    update(s);groups={};comparisons=[]
    for j in s['jobs']:
        if j['status'] not in ['done','reused']:continue
        for stage in ['tracking','bundle_adjustment']:
            for metric in ['mean_wall_ms','mean_cuda_ms','all_call_mean_wall_ms']:
                value=j['summary'][stage][metric]
                groups.setdefault((j['dataset'],j['scene'],j['hash_size'],j['condition'],stage,metric),[]).append((j['seed'],value))
            for cond in (['BASE'] if j['condition']=='DP' else ['BASE','MC']):
                control=s['controls'][j['control_ids'][cond]];a=j['summary'][stage]['mean_wall_ms'];b=control['summary'][stage]['mean_wall_ms']
                comparisons.append(dict(id=j['id'],seed=j['seed'],stage=stage,scope='complete_call_wall_primary_frames_ge_20',control_id=j['control_ids'][cond],condition_mean_ms=a,control_mean_ms=b,reduction_pct=100*(1-a/b)))
    paired=[dict(zip(['dataset','scene','hash_size','condition','stage','metric'],k),seeds=[x[0] for x in v],mean=statistics.mean(x[1] for x in v),sample_sd=statistics.stdev(x[1] for x in v) if len(v)>1 else None) for k,v in groups.items()]
    save(ROOT/'TIMING_SUMMARIES.json',paired);save(ROOT/'MATCHED_COMPARISONS.json',comparisons)
    (ROOT/'TIMING_REPORT.md').write_text('# MAXN DP / MC_DP timing evidence\n\nPrimary means exclude frames 0–19; all-call means separate; milliseconds. Complete-call wall/CUDA timings and summed iteration CUDA events are separate scopes. Negative reductions are retained. DP vs BASE, MC_DP vs BASE and MC_DP vs MC are directly matched; percentages are never added. No Orin quality/mesh/ATE.\n\n'+json.dumps(paired,indent=2)+'\n')
    index=[{k:j.get(k) for k in ['id','dataset','scene','condition','seed','hash_size','status','output','reason','config_sha256','source_manifest_sha256','control_ids','completion_record']} for j in s['jobs']]
    save(ROOT/'UNIQUE_RUN_INDEX.json',dict(targets=96,entries=index,existing_eight_archive=read(EXPORT/'EXPORT_RECEIPT.json'),matched_controls=s['controls']))
    template=read(ROOT/'bundle/manifests/orin_dp_mc_dp_all_targets.json');template.update(power_mode='15W',author_manual_switch_and_actual_mode_verification_required=True,do_not_launch_automatically=True,MAXN_controls_not_reusable_in_15W=True)
    for j in template['jobs']:
        candidate=next(x for x in s['jobs'] if x['id']==j['id'])
        if candidate.get('datadir'):j['datadir']=candidate['datadir']
    save(ROOT/'manifests/15W_TEMPLATE_NOT_LAUNCHED.json',template)
    target=TOP/'ORIN_DP_COMPLETION_MAXN_RESULTS.zip';temp=target.with_suffix('.zip.tmp');assert not target.exists() and not temp.exists()
    hashes={}
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in ROOT.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ['.zip','.tmp','.so','.sens','.pt','.ply'] or p.name.startswith('EMAIL_'):continue
            data=p.read_bytes();name=p.relative_to(ROOT).as_posix();hashes[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        z.writestr('RETURN_MANIFEST.json',json.dumps(dict(utc=now(),files=hashes,counts=s['counts'],expanded_required=292,ALL_REQUIRED_MAXN_COMPLETE=s['ALL_REQUIRED_MAXN_COMPLETE'],old_eight=read(EXPORT/'EXPORT_RECEIPT.json')),indent=2)+'\n')
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    temp.replace(target);digest=sha(target);target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n')
    save(ROOT/'RETURN_BUNDLE_VERIFIED.json',dict(utc=now(),path=str(target),sha256=digest,entries=len(hashes)))


def main(fd):
    assert os.fstat(fd).st_ino==os.stat('/tmp/dasip_round2_gpu0.lock').st_ino
    support.maxn();idle();s=read(ROOT/'DP_COMPLETION_STATUS.json');s.update(phase='IDLE_AUDIT_AND_RESOLUTION',worker_pid=os.getpid(),active_child_pid=None);update(s)
    source_hash=read(ROOT/'bundle/REUSE_AND_CONTROLS.json')['runtime_manifest_sha256'];assert sha(CO/'RUNTIME_MANIFEST.json')==source_hash
    for name,h in read(CO/'RUNTIME_MANIFEST.json')['runtime_hashes'].items():assert sha(CO/name)==h,name
    # Complete the ScanNet return increment here, after the existing exporter released ownership.
    scan_archive=TOP/'ORIN_SCANNET_MAXN_RESULTS.zip'
    if scan_archive.exists():
        assert sha(scan_archive)==scan_archive.with_suffix('.zip.sha256').read_text().split()[0]
    else:load_module('scannet_archiver',SCAN/'completion_observer.py').package(read(SCAN/'SCANNET_STATUS.json'))
    native={j['id']:j for j in read(SCAN/'CURRENT_SCOPE_STATUS.json')['jobs']};controls={};control_errors={}
    all_jobs=s['jobs'];target_by_id={j['id']:j for j in all_jobs}
    for cid in sorted({v for j in all_jobs for v in j['control_ids'].values()}):
        try:
            original=native[cid];assert original['status'] in ['done','reused'] and original.get('system','Co-SLAM')=='Co-SLAM'
            out=Path(original['output']);cfg=read(out/'config.json');j=dict(original,expected_native_frames=next(x['expected_native_frames'] for x in all_jobs if cid in x['control_ids'].values()),datadir=cfg['data']['datadir'])
            done=out.parent/(out.name+'.done.json')
            if done.is_file():assert read(done)['returncode']==0
            else:
                controller=ROOT.parent/'orin_continuous_v4/controller_records'/(cid+'.controller_done.json');assert read(controller)['returncode']==0;done=controller
            checked=audit(j,out,source_hash);checked.pop('stages');checked['completion_record']=str(done);controls[cid]=checked
        except Exception as e:control_errors[cid]=dict(error=repr(e),traceback=traceback.format_exc())
    s['controls']=controls;save(ROOT/'CONTROL_AUDIT.json',dict(verified=controls,errors=control_errors));update(s)
    for j in all_jobs:
        try:
            base=controls[j['control_ids']['BASE']];mc=controls[j['control_ids']['MC']]
            assert config_identity(base['config'])==config_identity(mc['config'])
            j['datadir']=base['config']['data']['datadir']
        except Exception as e:j.update(status='blocked',reason='Matched native control unavailable/incompatible: '+repr(e))
    # Index additional completed pruning outputs without walking datasets; never repeat a recorded success.
    candidates={}
    for exp in ROOT.parent.iterdir():
        if not exp.is_dir():continue
        base=exp/'results'
        if not base.is_dir():continue
        for p in base.rglob('summary.json'):
            a=read(p)
            if a.get('condition') not in ['DP','MC_DP'] or a.get('diagnostic') or not a.get('full_sequence'):continue
            if a.get('initial_power')!='NV Power Mode: MAXN\n0' or a.get('final_power')!='NV Power Mode: MAXN\n0' or a.get('quality') or a.get('warmup_frames')!=20:continue
            if sha(p.parent/'source_manifest.json')!=source_hash:continue
            cfg=read(p.parent/'config.json');key=(a['dataset'],a['scene'],a['condition'],a['seed'],cfg['grid']['hash_size'])
            candidates.setdefault(key,[]).append(p.parent)
    reuse_plan={j['id']:j for j in read(ROOT/'bundle/REUSE_AND_CONTROLS.json')['existing_full_runs']}
    for j in all_jobs:
        if j['status']=='blocked':continue
        key=(j['dataset'],j['scene'],j['condition'],j['seed'],j['hash_size']);found=candidates.get(key,[])
        if j['id'] in reuse_plan:
            expected=Path(reuse_plan[j['id']]['original_output']);found=[expected]+[p for p in found if p!=expected]
        if found:
            errors=[]
            for out in found:
                try:
                    assert read(out.parent/(out.name+'.done.json'))['returncode']==0
                    a=audit(j,out,source_hash,controls[j['control_ids']['BASE']]['config']);j.update(status='reused',output=str(out),summary=a['summary'],config_sha256=a['config_sha256'],source_manifest_sha256=source_hash,completion_record=str(out.parent/(out.name+'.done.json')));break
                except Exception as e:errors.append(dict(output=str(out),error=repr(e)))
            else:j.update(status='unresolved_reuse',reason='Existing full success is not repeated; audit unresolved',reuse_errors=errors)
        else:j['status']='queued'
    # Audit all scene paths before the first new timing child.
    data_records={}
    for j in all_jobs:
        key=(j['dataset'],j['scene'])
        if key in data_records or j['status'] not in ['queued','reused']:continue
        try:data_records[key]=data_probe(controls[j['control_ids']['BASE']]['config'],j['expected_native_frames'])
        except Exception as e:data_records[key]=dict(status='FAIL',error=repr(e),traceback=traceback.format_exc())
    for j in all_jobs:
        if data_records.get((j['dataset'],j['scene']),{}).get('status')=='FAIL' and j['status']=='queued':j.update(status='blocked',reason='Native full-sequence data audit failed')
    save(ROOT/'DATA_AUDIT.json',{'/'.join(k):v for k,v in data_records.items()});update(s)
    baseline=read(SCAN/'ENVIRONMENT.json');save(ROOT/'ENVIRONMENT.json',support.hardware())
    if any(j['status']=='queued' for j in all_jobs):
        cache_log=ROOT/'logs/morton_cache_before_timings.log'
        if cache_log.exists():cache_log=ROOT/'logs'/('morton_cache_before_timings_'+now().replace(':','-')+'.log')
        cache_rc=run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--prepare-morton'],cache_log,fd,s)
        if cache_rc:
            save(ROOT/'MORTON_CACHE_FAILURE.json',dict(utc=now(),returncode=cache_rc,exact_log=str(cache_log),no_hash_substitution=True))
            for j in all_jobs:
                if j['status']=='queued' and j['condition']=='MC_DP':j.update(status='blocked',reason='Morton cache/API preparation failed; exact log preserved; independent DP remains ready')
            update(s)
    adapter=ROOT/'run_queue_inherited_lock.py';shutil.copyfile(OLD/'run_queue_inherited_lock.py',adapter)
    original_order=read(ROOT/'bundle/manifests/orin_dp_mc_dp_missing_maxn.json')['jobs']
    resolved=[]
    for original in original_order:
        j=target_by_id[original['id']]
        if j['status']!='queued':continue
        saved={k:v for k,v in j.items() if k in original};resolved.append(saved)
        save(ROOT/'manifests/jobs'/(j['id']+'.json'),dict(warmup_frames=20,jobs=[saved]))
    save(ROOT/'manifests/RESOLVED_MISSING_MAXN.json',dict(power_mode='MAXN',warmup_frames=20,jobs=resolved))
    freeze=[ROOT/'common.py',Path(__file__).resolve(),adapter,ROOT/'manifests/RESOLVED_MISSING_MAXN.json']+list((ROOT/'manifests/jobs').glob('*.json'))+[CO/name for name in read(CO/'RUNTIME_MANIFEST.json')['runtime_hashes']]+[CO/'RUNTIME_MANIFEST.json']
    save(ROOT/'EXECUTION_FREEZE.json',dict(files={str(p):sha(p) for p in freeze}))
    s['phase']='SETTLING_BEFORE_DP_TIMINGS';update(s)
    if resolved:support.readiness('after_audit_exports_and_morton_before_DP',baseline)
    s['phase']='RUNNING_MISSING_DP_MAXN';update(s)
    for original in original_order:
        j=target_by_id[original['id']]
        if j['status']!='queued':continue
        try:
            for p,h in read(ROOT/'EXECUTION_FREEZE.json')['files'].items():assert sha(p)==h,p
            out=ROOT/'results'/j['id'];assert not out.exists(),'Existing partial output preserved; no retry'
            manifest=ROOT/'manifests/jobs'/(j['id']+'.json');argv=[sys.executable,str(adapter),'--inherited-workload-lock-fd',str(fd),'--runtime',str(CO),'--manifest',str(manifest),'--output',str(out),'--power-mode','MAXN']
            j.update(status='running',output=str(out/j['id']),started_utc=now());update(s)
            assert run(argv,ROOT/'logs'/(j['id']+'.log'),fd,s)==0,'Queue failed; exact output/log preserved'
            assert read(out/(j['id']+'.done.json'))['returncode']==0 and (out/'queue_complete.json').is_file()
            a=audit(j,out/j['id'],source_hash,controls[j['control_ids']['BASE']]['config'])
            j.update(status='done',summary=a['summary'],config_sha256=a['config_sha256'],source_manifest_sha256=source_hash,completion_record=str(out/(j['id']+'.done.json')))
            save(ROOT/'validation'/(j['id']+'.json'),a);update(s)
        except Exception as e:
            j.update(status='blocked',reason=repr(e));save(ROOT/'failures'/(j['id']+'.json'),dict(error=repr(e),traceback=traceback.format_exc()))
            support.maxn();idle()
            for other in all_jobs:
                if (other['dataset'],other['scene'])==(j['dataset'],j['scene']) and other['status']=='queued':other.update(status='blocked',reason='Affected scene comparison stopped after '+j['id'])
            update(s)
    s.update(phase='COMPLETE_ADMITTED_DP_SERIES',active_child_pid=None);update(s)
    new_completed=sum(j['status'] in ['done','reused'] and Path(j.get('output','/')).is_relative_to(ROOT/'results') for j in all_jobs)
    save(ROOT/'DP_COMPLETION_COMPLETE.json',dict(utc=now(),major_admitted_series_terminal=True,counts=s['counts'],new_full_completed=new_completed,ALL_REQUIRED_MAXN_COMPLETE=s['ALL_REQUIRED_MAXN_COMPLETE'],expanded_required=292,expanded_completed=s['expanded_completed'],fresh_idle=idle(),power=support.maxn()))
    report_and_package(s)
    print('DP_COMPLETION_MAJOR_SERIES_TERMINAL',s['counts'],flush=True)


if __name__=='__main__':
    if '--prepare-morton' in sys.argv:prepare_morton()
    else:
        p=argparse.ArgumentParser();p.add_argument('--workload-lock-fd',type=int,required=True);a=p.parse_args()
        try:main(a.workload_lock_fd)
        except Exception as e:
            save(ROOT/'WORKER_STOPPED.json',dict(utc=now(),error=repr(e),traceback=traceback.format_exc(),major_admitted_series_terminal=False));raise
