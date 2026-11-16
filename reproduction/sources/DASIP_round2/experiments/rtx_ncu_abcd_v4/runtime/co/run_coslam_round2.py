#!/usr/bin/env python3
"""Co-SLAM Round 2: public-reference estimator, optional scientific samplers.

Whole-call measurements supplement the inherited Orin iteration CSV schema.
Quality is optional, and mesh export occurs after all training measurements.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time


def write_csv(path, rows, fields=None):
    if not rows:return
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def ate(gt, estimated):
    import numpy as np
    truth=gt[:,:3,3].astype(np.float64);pred=estimated[:,:3,3].astype(np.float64)
    valid=np.isfinite(truth).all(1)&np.isfinite(pred).all(1)
    if valid.sum()<3:raise ValueError('Not enough valid poses for rigid ATE')
    x=pred[valid];y=truth[valid];xc=x-x.mean(0);yc=y-y.mean(0)
    u,_,v=np.linalg.svd(xc.T@yc);sign=np.eye(3);sign[-1,-1]=np.linalg.det(u@v)
    rotation=u@sign@v
    return float(np.sqrt(np.mean(np.sum((xc@rotation-yc)**2,axis=1)))),int(valid.sum())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['Replica','Synthetic','Tum','ScanNet'],required=True)
    parser.add_argument('--scene',required=True)
    parser.add_argument('--datadir',type=Path,required=True)
    parser.add_argument('--condition',choices=['BASE','MC','DP','MC_DP'],required=True)
    parser.add_argument('--hash-size',type=int,help='Explicit capacity sensitivity; default is public scene setting')
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--frames',type=int,default=0,help='0=full sequence; positive values are diagnostic only')
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quality',action='store_true')
    parser.add_argument('--export-mesh',action='store_true')
    parser.add_argument('--no-iteration-events',action='store_true',help='Separate wall-only profiling pass')
    parser.add_argument('--expected-power-mode',help='On Orin, exact nvpmodel name, e.g. MAXN; no mode change')
    parser.add_argument('--ncu-variant',choices=['A','B','C','D'],required=True)
    args=parser.parse_args()
    ncu_variants={'A': {'hash': 'CoherentPrime', 'morton_sort': False}, 'B': {'hash': 'Morton', 'morton_sort': False}, 'C': {'hash': 'CoherentPrime', 'morton_sort': True}, 'D': {'hash': 'Morton', 'morton_sort': True}}
    ncu_spec=ncu_variants[args.ncu_variant]
    if args.condition!='BASE' or args.quality or args.export_mesh:parser.error('NCU ablation uses dense BASE computation, no pruning/quality/meshes')
    if args.frames<0 or args.warmup<0:parser.error('Invalid frame limit or warmup')
    output=args.output.resolve();datadir=args.datadir.resolve()
    if output.exists() and any(output.iterdir()):parser.error('Use a fresh output directory')
    if not datadir.is_dir():parser.error('Dataset directory does not exist: '+str(datadir))
    if os.environ.get('PYTHONHASHSEED')!=str(args.seed):
        environment=dict(os.environ,PYTHONHASHSEED=str(args.seed))
        os.execve(sys.executable,[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],environment)
    # Resolve paths before switching to the isolated public source tree.
    runtime=Path(__file__).resolve().parent
    os.chdir(runtime);sys.path.insert(0,str(runtime))
    import numpy as np
    import torch
    import config
    import coslam
    from torch.utils.data import DataLoader
    from round2_sampling import install
    from round2_timing import IterationBreakdownCudaTimer
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    cfg=config.load_config('configs/'+args.dataset+'/'+args.scene+'.yaml')
    cfg['data'].update(datadir=str(datadir),output=str(output),exp_name='artifacts')
    cfg['mapping']['first_mesh']=False
    cfg['mesh']['visualisation']=False
    cfg['grid'].update(hash=ncu_spec['hash'],morton_sort=ncu_spec['morton_sort'])
    if args.hash_size is not None:cfg['grid']['hash_size']=args.hash_size
    if cfg['tracking']['iter_point']!=0:raise ValueError('Point-cloud tracking is outside this study')
    cfg['round2']=dict(condition=args.condition,seed=args.seed,sampler='support' if args.condition in ('DP','MC_DP') else 'dense',
                      margin_scene_units=.15,anchor_every=3,engineering=False,seed_policy='once before model construction; ordinary RNG evolution')
    cfg['round2'].update(ncu_variant=args.ncu_variant,kernel_profile_only=True)
    coslam.cfg=cfg
    output.mkdir(parents=True,exist_ok=True);(output/'artifacts').mkdir()
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    manifest=json.loads((runtime/'RUNTIME_MANIFEST.json').read_text())
    for name,digest in manifest['runtime_hashes'].items():
        if hashlib.sha256((runtime/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Runtime source changed: '+name)
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def power():
        if not args.expected_power_mode:return None
        import subprocess,shutil
        path=shutil.which('nvpmodel') or '/usr/sbin/nvpmodel'
        value=subprocess.check_output([path,'-q'],text=True,timeout=10).strip()
        lines=value.splitlines();name=next((s.split(':',1)[1].strip() for s in lines if s.startswith('NV Power Mode:')),None)
        if name!=args.expected_power_mode:raise RuntimeError('Unexpected power mode: '+value)
        return value
    initial_power=power()
    slam=coslam.CoSLAM(cfg)
    install(slam.model,prune=False,morton=ncu_spec['morton_sort'])
    timer=IterationBreakdownCudaTimer(enabled=not args.no_iteration_events,
        output_csv=output/'iterations.csv',warmup_frames=0)
    slam.coslam_timing=timer
    calls=[];frames=[]
    def measure(method, stage, batch, frame):
        model=slam.model
        model._r2_stage=stage;model._r2_frame=int(frame);model._r2_support=None
        timer.set_context(int(frame),stage)
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize();wall=time.perf_counter();start.record()
        result=method(batch,frame)
        end.record();torch.cuda.synchronize();elapsed=(time.perf_counter()-wall)*1000
        calls.append(dict(frame_id=int(frame),stage=stage,wall_ms=elapsed,cuda_ms=start.elapsed_time(end)))
        model._r2_stage=None
        return result
    slam.create_optimizer()
    loader=DataLoader(slam.dataset,num_workers=cfg['data']['num_workers'])
    torch.cuda.synchronize();run_start=time.perf_counter();last=-1
    for frame,batch in enumerate(loader):
        if args.frames and frame>=args.frames:break
        if frame%100==0:power()
        last=frame;before=len(calls);frame_start=time.perf_counter()
        if frame==0:
            slam.first_frame_mapping(batch,cfg['mapping']['first_iters'])
        else:
            measure(slam.tracking_render,'tracking',batch,frame)
            if frame%cfg['mapping']['map_every']==0:
                measure(slam.global_BA,'bundle_adjustment',batch,frame)
            if frame%cfg['mapping']['keyframe_every']==0:
                slam.keyframeDatabase.add_keyframe(batch,filter_depth=cfg['mapping']['filter_depth'])
        torch.cuda.synchronize()
        recent=calls[before:]
        frames.append(dict(frame_id=frame,frame_work_wall_ms=(time.perf_counter()-frame_start)*1000,
                           tracking_wall_ms=sum(x['wall_ms'] for x in recent if x['stage']=='tracking'),
                           ba_wall_ms=sum(x['wall_ms'] for x in recent if x['stage']=='bundle_adjustment'),
                           has_ba=int(any(x['stage']=='bundle_adjustment' for x in recent))))
        timer.flush()
    torch.cuda.synchronize();run_seconds=time.perf_counter()-run_start;final_power=power()
    finite=all(bool(torch.isfinite(p).all()) for p in slam.model.parameters()) and all(bool(torch.isfinite(p).all()) for p in slam.est_c2w_data.values())
    if not finite:raise RuntimeError('Nonfinite model parameters or estimated poses')
    write_csv(output/'calls.csv',calls);write_csv(output/'frames.csv',frames)
    write_csv(output/'queries.csv',slam.model._r2_queries)
    # Preserve the historical headerless frame iteration-sum contract as a companion.
    if (output/'iterations.csv').exists():
        byframe={}
        for row in csv.DictReader((output/'iterations.csv').open()):
            totals=byframe.setdefault(int(row['frame_id']),[0.,0.])
            totals[0 if row['iteration_type']=='tracking' else 1]+=float(row['full_iteration_cuda_ms'])
        with (output/'frame_iteration_sums.csv').open('w',newline='') as file:
            writer=csv.writer(file)
            for frame,(tracking,ba) in sorted(byframe.items()):writer.writerow([frame,tracking+ba,tracking,ba])
    summary=dict(ncu_variant=args.ncu_variant,ncu_spec=ncu_spec,kernel_profile_only=True,dataset=args.dataset,scene=args.scene,condition=args.condition,seed=args.seed,
        frames=last+1,full_sequence=last+1==len(slam.dataset),diagnostic=args.frames>0,
        device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
        finite_parameters_and_poses=finite,slam_work_wall_seconds=run_seconds,
        warmup_frames=args.warmup,initial_power=initial_power,final_power=final_power,quality=args.quality,
        timing_scope='Complete estimator call incl. setup and finalization; CSV resolution outside invocation; total run includes initialization/loading/keyframe work, excludes construction/evaluation/mesh',
        public_reference=manifest['public_revision'])
    for stage in ('tracking','bundle_adjustment'):
        selected=[r for r in calls if r['stage']==stage and r['frame_id']>=args.warmup]
        allcalls=[r for r in calls if r['stage']==stage]
        summary[stage]={label:float(np.mean([r[field] for r in selected])) if selected else None
            for label,field in [('mean_wall_ms','wall_ms'),('mean_cuda_ms','cuda_ms')]}
        summary[stage]['count']=len(selected)
        summary[stage]['all_call_mean_wall_ms']=float(np.mean([r['wall_ms'] for r in allcalls])) if allcalls else None
        rows=[r for r in slam.model._r2_queries if r['stage']==stage]
        dense=sum(r['dense_queries'] for r in rows);actual=sum(r['actual_queries'] for r in rows)
        summary[stage].update(dense_queries=dense,actual_queries=actual,queries_pruned_pct=100*(1-actual/dense) if dense else None)
    if args.quality:
        ids=sorted(slam.est_c2w_data)
        truth=np.stack([slam.pose_gt[i].detach().cpu().numpy() for i in ids])
        estimate=np.stack([slam.est_c2w_data[i].detach().cpu().numpy() for i in ids])
        relative=slam.convert_relative_pose();corrected=np.stack([relative[i].detach().cpu().numpy() for i in ids])
        np.savez(output/'trajectories.npz',frame_ids=ids,gt=truth,estimated=estimate,corrected=corrected)
        value,count=ate(truth,estimate);value_r,_=ate(truth,corrected)
        summary.update(ate_rmse_m=value/cfg['data']['sc_factor'],corrected_ate_rmse_m=value_r/cfg['data']['sc_factor'],valid_gt_frames=count)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('ROUND2_RESULT '+json.dumps(summary),flush=True)
    if args.export_mesh:
        start=time.perf_counter();slam.save_mesh(last,voxel_size=cfg['mesh']['voxel_final'])
        (output/'mesh_export.json').write_text(json.dumps(dict(frame_id=last,excluded_from_training_timings=True,wall_seconds=time.perf_counter()-start))+'\n')


if __name__=='__main__':main()
