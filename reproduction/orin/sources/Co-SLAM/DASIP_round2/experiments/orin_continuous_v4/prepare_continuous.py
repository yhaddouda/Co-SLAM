"""CPU-only local data validation, compatible reuse and explicit backlog."""
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'orin_paper_night_v2'
CO=ROOT.parent/'orin_coslam_runtime_v1'
RUNTIME=CO/'runtime'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    # Verify the supplied unchanged runtime and already-installed package, not the working repositories.
    assert sha(ROOT/'orin_coslam_runtime_v1.zip')==sha(CO.parents[1]/'orin_coslam_runtime_v1.zip')
    for n,h in read(CO/'SHA256SUMS.json').items():assert sha(CO/n)==h,n
    for n,h in read(OLD/'EC_RUNTIME_MANIFEST.json')['files'].items():assert sha(OLD/n)==h,n
    for n,h in read(ROOT/'SHA256SUMS.json').items():assert sha(ROOT/n)==h,n
    os.chdir(RUNTIME);sys.path.insert(0,str(RUNTIME))
    import config
    from datasets.dataset import get_dataset
    import torch
    import cv2
    import numpy as np
    torch.set_num_threads(1);cv2.setNumThreads(1)
    inventory={}
    expected={(x['dataset'],x['scene']):x for x in read(ROOT/'ALL_CONFIGURED_NONREPLICA_INVENTORY.json')['scenes']}
    all_jobs=[]
    for manifest,system in [('orin_continuous_coslam.json','Co-SLAM'),('orin_continuous_ec.json','EC-SLAM')]:
        for raw in read(ROOT/'manifests'/manifest)['jobs']:
            job=dict(raw,system=system,mode='MAXN')
            dataset,scene=job['dataset'],job['scene'];key=system+'/'+dataset+'/'+scene
            if key not in inventory:
                hint=Path(job['datadir'])
                if system=='EC-SLAM':
                    path=Path('/home/yh279050/work/EC-SLAM/datasets/vmap')/(scene[:-1]+'_'+scene[-1])/'imap/00'
                elif dataset=='Replica':path=Path('/home/yh279050/work/data/Replica')/scene
                elif dataset in ('Synthetic','Tum'):path=Path('/home/yh279050/work/data')/('TUM' if dataset=='Tum' else 'neural_rgbd_data')/hint.name
                else:
                    candidates=[hint,Path('/home/yh279050/work/data/scans')/hint.name,Path('/home/yh279050/work/data/scannet')/hint.name,Path('/home/yh279050/work/Co-SLAM/data/scannet')/hint.name]
                    path=next((p for p in candidates if p.is_dir()),candidates[1])
                item=dict(system=system,dataset=dataset,scene=scene,datadir=str(path.resolve()),hint=str(hint),status='blocked')
                try:
                    assert path.is_dir(),'Dataset missing; transfer native RGB/depth/pose data to '+str(path)
                    if system=='Co-SLAM':
                        cfg=config.load_config('configs/'+dataset+'/'+scene+'.yaml');cfg['data']['datadir']=str(path)
                        ds=get_dataset(cfg);n=len(ds);assert n>20
                        paths=getattr(ds,'img_files',getattr(ds,'color_paths',[]));depths=getattr(ds,'depth_paths',[])
                        assert len(paths)==len(depths)==n,(len(paths),len(depths),n)
                        assert all(Path(p).is_file() for p in paths+depths),'Missing selected RGB/depth file'
                        probes=[]
                        for i in sorted({0,n//2,n-1}):
                            frame=ds[i]
                            assert all(bool(torch.isfinite(frame[k]).all()) for k in ['rgb','depth','direction','c2w'])
                            probes.append(dict(frame=i,rgb_shape=list(frame['rgb'].shape),depth_shape=list(frame['depth'].shape)))
                        native_hash=cfg['grid']['hash_size'];item['mapping_iters']=cfg['mapping']['iters'];item['scene_config_sha256']=sha(RUNTIME/'configs'/dataset/(scene+'.yaml'))
                        if (dataset,scene) in expected:
                            e=expected[(dataset,scene)];assert n==e['frames'] and native_hash==e['native_hash_size']
                            assert item['scene_config_sha256']==e['scene_config_sha256']
                        if dataset=='Synthetic' and scene=='ma':item['path_override_reason']='Public ma.yaml repeats gwr; actual morning_apartment path explicitly selected; all other native settings retained'
                    else:
                        rgb=[path/'rgb'/('rgb_'+str(i)+'.png') for i in range(2000)]
                        depth=[path/'depth'/('depth_'+str(i)+'.png') for i in range(2000)]
                        assert all(p.is_file() for p in rgb+depth)
                        assert len(list((path/'depth').iterdir()))==2000
                        poses=np.loadtxt(path/'traj_w_c.txt').reshape(-1,4,4);assert len(poses)==2000 and np.isfinite(poses).all()
                        probes=[]
                        for i in [0,1000,1999]:
                            a=cv2.imread(str(rgb[i]));b=cv2.imread(str(depth[i]),-1);assert a is not None and b is not None
                            probes.append(dict(frame=i,rgb_shape=list(a.shape),depth_shape=list(b.shape)))
                        n=2000;native_hash=16
                    item.update(status='ready',frames=n,native_hash_size=native_hash,first_middle_last=probes,all_selected_paths_exist=True)
                except Exception as error:item['reason']=repr(error)
                inventory[key]=item;print('DATA',key,item['status'],item.get('frames'),item.get('reason',''),flush=True)
            item=inventory[key];job['datadir']=item['datadir'];job['expected_frames']=item.get('frames');job['status']='queued' if item['status']=='ready' else 'blocked'
            if item['status']!='ready':job['reason']=item['reason']
            all_jobs.append(job)
    # Existing full jobs get full iteration/call checks; names alone never establish reuse.
    spec=importlib.util.spec_from_file_location('old_parent',OLD/'paper_night_parent.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.verify_freeze();prior=read(OLD/'PAPER_NIGHT_STATUS.json');reuse=[]
    for b in prior['blocks']:
        for j in b['jobs']:
            if j['status']!='done' or j['condition'] not in ('BASE','MC'):continue
            out=Path(j['output']);summary=old.check_job(j,out,j['expected_frames'],j['system']=='EC-SLAM')
            assert (out.parent/(j['id']+'.done.json')).is_file()
            # Validate complete call identity/coverage in addition to inherited iteration checks.
            calls=old.rows(out/'calls.csv');cfg=read(out/'config.json');n=j['expected_frames'];ec=j['system']=='EC-SLAM'
            for stage,ids in [('tracking_frame' if ec else 'tracking',set(range(n)) if ec else set(range(1,n))),('bundle_adjustment',set(range(cfg['mapping']['every_frame' if ec else 'map_every'],n,cfg['mapping']['every_frame' if ec else 'map_every']))|({n-1} if ec else set()))]:
                selected=[r for r in calls if r['stage']==stage];assert len(selected)==len(ids) and {int(r['frame_id']) for r in selected}==ids
            matches=[x for x in all_jobs if all(x[k]==j[k] for k in ['system','dataset','scene','condition','seed','hash_size'])]
            assert len(matches)==1;target=matches[0]
            assert Path(target['datadir']).resolve()==Path(j['datadir']).resolve() and target['expected_frames']==n
            target.update(status='reused',output=str(out),summary=summary,elapsed_seconds=j['elapsed_seconds'],reuse_provenance=dict(original_job=j['id'],source_manifest_sha256=sha(out/'source_manifest.json'),effective_config_sha256=sha(out/'config.json'),completion_sha256=sha(out.parent/(j['id']+'.done.json')),warmup_frames=20,mode='MAXN',effective_adapter_compatibility='EC generalized driver retains exact original estimator/timing; office0 h16 default effective config unchanged' if ec else 'Exact unchanged runtime v1'))
            reuse.append(target['id'])
    all_jobs+= [dict(x,mode='MAXN',status='pending',reason=x['requires']) for x in read(ROOT/'manifests/orin_pending_engineering.json')['jobs']]
    def priority(j):
        if j.get('scene')=='office0' and j['condition']!='ENG':return (0,0 if j['system']=='Co-SLAM' else 1,j['hash_size'],j['scene'],j['seed'],j['condition'])
        if j['condition']=='ENG':return (1,j['system'],j['seed'])
        return (3 if j['dataset']=='Replica' and j['hash_size']==19 else 2,{'Synthetic':0,'Tum':1,'ScanNet':2,'Replica':3}[j['dataset']],j['scene'],j['system'],j['seed'],j['condition'])
    all_jobs.sort(key=priority)
    save(ROOT/'LOCAL_DATA_VALIDATION.json',dict(gpu_work=False,scenes=list(inventory.values()),searched_roots=['/home/yh279050/work/data','/home/yh279050/work/EC-SLAM/datasets'],missing_scannet='No scans/scannet scene folders in local project data roots; six scene transfers required'))
    save(ROOT/'LOCAL_MAXN_PLAN.json',dict(mode='MAXN',clock_policy='Retain current settings as explicitly authorized; full jetson_clocks lock state unknown',total_slots=len(all_jobs),jobs=all_jobs,priority='office0 both systems/capacities; validated G1; other native available scenes; remaining Replica h19',no_overnight_cap=True))
    save(ROOT/'REUSE_AUDIT.json',dict(reused=len(reuse),jobs=[j for j in all_jobs if j['status']=='reused']))
    save(ROOT/'LOCAL_15W_MATCHING_PLAN_NOT_AUTHORIZED.json',dict(status='PREPARED_ONLY_WAIT_FOR_MANUAL_SWITCH_AND_EXPLICIT_INSTRUCTION',power_mode='Exact 15 W nvpmodel name must be queried after author switch',jobs=[{k:v for k,v in j.items() if k not in ['summary','reuse_provenance','elapsed_seconds','output','status','mode','reason']} for j in all_jobs]))
    print('PLAN',collections.Counter(j['status'] for j in all_jobs),flush=True)

if __name__=='__main__':main()
