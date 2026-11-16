#!/usr/bin/env python3
"""Prepare the complete saved-mesh evaluation list without retraining SLAM."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    base=Path(__file__).resolve().parents[1]
    runtime=base/'experiments/reconstruction_runtime_v1'
    assets=base/'experiments/reconstruction_assets_v1'
    if runtime.exists() or assets.exists():raise RuntimeError('Do not overwrite frozen evaluation inputs')
    runtime.mkdir();assets.mkdir()
    shutil.copy2(base/'experiments/checked_reconstruction.py',runtime/'checked_reconstruction.py')
    co_runtime=base/'experiments/coslam_reference_runtime_v1'
    sys.path.insert(0,str(co_runtime));os.chdir(co_runtime)
    from datasets.dataset import get_dataset
    roots=['support_reference_v1','morton_core_v1','replica_native_extension_v1','capacity_office0_h19_v1','replica_h19_extension_v1','all_nonreplica_extension_v1']
    co_jobs=[];scene_configs={}
    for name in roots:
        queue=base/'results'/name
        assert (queue/'queue_complete.json').exists()
        for job in json.loads((queue/'job_manifest.json').read_text())['jobs']:
            if job['dataset'] not in ('Replica','Synthetic'):continue
            directory=queue/job['id'];cfg=json.loads((directory/'config.json').read_text())
            assert cfg['data']['sc_factor']==1 and cfg['data']['translation']==0
            summary=json.loads((directory/'summary.json').read_text());assert summary['full_sequence']
            meshes=list((directory/'artifacts').glob('mesh_track*.ply'));assert len(meshes)==1
            assert meshes[0].stem=='mesh_track'+str(summary['frames']-1)
            scene_configs.setdefault((job['dataset'],job['scene']),directory/'config.json')
            co_jobs.append(dict(id=f"Co_{job['dataset']}_{job['scene']}_{job['condition']}_H{cfg['grid']['hash_size']}_seed{job['seed']}",system='Co-SLAM',dataset=job['dataset'],scene=job['scene'],condition=job['condition'],hash_size=cfg['grid']['hash_size'],seed=job['seed'],mesh=str(meshes[0]),summary=str(directory/'summary.json'),source_manifest=str(directory/'source_manifest.json'),source_manifest_sha256=sha(directory/'source_manifest.json'),upstream_complete=str(queue/(job['id']+'.done.json'))))
    evaluation_data=Path('/home/yh279050/work/datasets/vv_data')
    alignment={}
    for (dataset,scene),config_path in sorted(scene_configs.items()):
        cfg=json.loads(config_path.read_text());data=get_dataset(cfg)
        real=np.array([np.array(data.poses[i],dtype=np.float32) for i in data.frame_ids[::2]])
        dirname=Path(cfg['data']['datadir']).name
        gt_root=evaluation_data/('Replica' if dataset=='Replica' else 'neural_rgbd_data')/dirname
        gt_mesh=gt_root/'gt_mesh_cull_virt_cams.ply';metric_poses=gt_root/'sampled_poses_1000.npz'
        assert gt_mesh.is_file() and metric_poses.is_file()
        virtual_files=sorted((gt_root/'virtual_cameras').glob('*.txt'),key=lambda p:int(p.stem))
        assert len(virtual_files)>0
        virtual=[]
        for p in virtual_files:
            pose=np.loadtxt(p).reshape(4,4);pose[:3,1:3]*=-1;virtual.append(pose.astype(np.float32))
        all_poses=np.concatenate([real,np.asarray(virtual)])
        assert np.isfinite(all_poses).all()
        metric=np.load(metric_poses)['poses'];assert metric.shape==(1000,4,4) and np.isfinite(metric).all()
        pose_path=assets/(dataset+'_'+scene+'_cull_poses.npz')
        np.savez(pose_path,poses=all_poses,real_count=len(real))
        K=[[cfg['cam']['fx'],0,cfg['cam']['cx']],[0,cfg['cam']['fy'],cfg['cam']['cy']],[0,0,1.]]
        asset=dict(dataset=dataset,scene=scene,width=cfg['cam']['W'],height=cfg['cam']['H'],K=K,gt_mesh=str(gt_mesh),metric_poses=str(metric_poses),cull_poses=str(pose_path),cull_real_stride=2,real_views=len(real),virtual_views=len(virtual),metric_view_convention='OpenCV',cull_view_convention='OpenGL',units='metres; source scaling 1 and translation 0',hashes={str(p):sha(p) for p in [gt_mesh,metric_poses,pose_path,config_path,co_runtime/'datasets/dataset.py']})
        (assets/(dataset+'_'+scene+'.json')).write_text(json.dumps(asset,indent=2)+'\n')
        if dataset=='Replica':
            co=np.loadtxt(Path(cfg['data']['datadir'])/'traj.txt').reshape(-1,4,4)
            ec_dir=Path('/home/yh279050/work/EC-SLAM/datasets/vmap')/(scene[:-1]+'_'+scene[-1])/'imap/00'
            ec=np.loadtxt(ec_dir/'traj_w_c.txt').reshape(-1,4,4)
            assert np.array_equal(co,ec),scene
            alignment[scene]=dict(gt_raw_camera_matrices_identical=True,maximum_difference=float(np.max(abs(co-ec))),co_sha256=sha(Path(cfg['data']['datadir'])/'traj.txt'),ec_sha256=sha(ec_dir/'traj_w_c.txt'))
    assert len(scene_configs)==15
    all_jobs=co_jobs
    ec_queue=base/'results/ec_quality_v1'
    ec_manifest=base/'experiments/ec_quality_runtime_v1/RUNTIME_MANIFEST.json'
    for job in json.loads((ec_queue/'job_manifest.json').read_text())['jobs']:
        directory=ec_queue/job['id']
        all_jobs.append(dict(id=f"EC_Replica_{job['scene']}_{job['condition']}_H{job['hash_size']}_seed{job['seed']}",system='EC-SLAM',dataset='Replica',scene=job['scene'],condition=job['condition'],hash_size=job['hash_size'],seed=job['seed'],mesh=str(directory/'final_mesh.ply'),summary=str(directory/'summary.json'),source_manifest=str(directory/'source_manifest.json'),source_manifest_sha256=sha(ec_manifest),upstream_complete=str(ec_queue/(job['id']+'.done.json'))))
    for job in all_jobs:
        job['asset']=str(assets/(job['dataset']+'_'+job['scene']+'.json'))
        job['asset_sha256']=sha(job['asset'])
    def order(j):
        priority=0 if j['dataset']=='Replica' and j['hash_size']==16 and j['condition']!='DP' else 1 if j['dataset']=='Synthetic' and j['condition']!='DP' else 2 if j['condition']!='DP' else 3
        condition_order=['BASE','MC','DP'] if j['seed']%2==0 else ['MC','DP','BASE']
        return priority,j['scene'],j['system'],j['seed'],condition_order.index(j['condition'])
    all_jobs.sort(key=order)
    assert len(all_jobs)==246 and len({j['id'] for j in all_jobs})==246
    document=dict(purpose='Common reconstruction evaluation of every saved/queued Replica and Synthetic final mesh, no retraining or TUM/ScanNet mesh scoring',jobs=all_jobs)
    (base/'experiments/manifests/rtx_reconstruction_all.json').write_text(json.dumps(document,indent=2)+'\n')
    (assets/'CO_EC_GT_ALIGNMENT_AUDIT.json').write_text(json.dumps(alignment,indent=2)+'\n')
    hashes={str(p):sha(p) for p in [runtime/'checked_reconstruction.py',*assets.glob('*')]}
    (runtime/'RUNTIME_MANIFEST.json').write_text(json.dumps(dict(purpose='post-SLAM scoring only; no training timing claims',file_hashes=hashes),indent=2)+'\n')
    print(json.dumps(dict(mesh_evaluations=len(all_jobs),co=150,ec=96,scenes=15,legacy_gt_depth_caches_reused=False,common_ec_co_gt_alignment='exact identity in all eight scenes'),indent=2))


if __name__=='__main__':main()
