"""Pinned native EC, once-only seeds, native threads, explicit measurement scope."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition',choices=['BASE','MC'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--scene',default='office0',choices=['office0','office1','office2','office3','office4','room0','room1','room2'])
    parser.add_argument('--hash-size',type=int,default=16,choices=[16,19])
    parser.add_argument('--expected-power-mode',default=None)
    parser.add_argument('--frames',type=int,default=0)
    parser.add_argument('--datadir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--co-runtime',type=Path,required=True)
    parser.add_argument('--ncu-variant',choices=['A','B','C','D'],required=True)
    args=parser.parse_args()
    ncu_variants={'A': {'hash': 'CoherentPrime', 'morton_sort': False}, 'B': {'hash': 'Morton', 'morton_sort': False}, 'C': {'hash': 'CoherentPrime', 'morton_sort': True}, 'D': {'hash': 'Morton', 'morton_sort': True}}
    ncu_spec=ncu_variants[args.ncu_variant]
    if args.condition!='BASE':parser.error('NCU ablation uses dense BASE computation without pruning or engineering')
    if os.environ.get('PYTHONHASHSEED')!=str(args.seed):
        os.execve(sys.executable,[sys.executable,*sys.argv],dict(os.environ,PYTHONHASHSEED=str(args.seed)))
    root=Path(__file__).resolve().parent;runtime=root/'ec_runtime'
    output=args.output.resolve();data=args.datadir.resolve();co=args.co_runtime.resolve()
    if output.exists():raise RuntimeError('Preserving existing EC output')
    freeze=json.loads((root/'EC_RUNTIME_MANIFEST.json').read_text())
    for name,digest in freeze['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    def power():
        if not args.expected_power_mode:return None
        text=subprocess.check_output(['/usr/sbin/nvpmodel','-q'],text=True,timeout=10).strip()
        name=next((line.split(':',1)[1].strip() for line in text.splitlines() if line.startswith('NV Power Mode:')),None)
        if name!=args.expected_power_mode:raise RuntimeError('Unexpected mode: '+text)
        return text
    initial_power=power()
    import numpy as np
    import torch
    from ec_timing import Timer
    os.chdir(runtime);sys.path.insert(0,str(runtime))
    from src.utils import load_config
    import src.Model as model_module
    import src.Datasets as datasets
    from src.system import EC_SLAM
    assert torch.cuda.is_available()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    cfg=load_config('configs/Replica/'+args.scene+'.yaml','configs/EC_SLAM.yaml')
    cfg['data'].update(input_folder=str(data),output=str(output/'artifacts'))
    cfg['round2_timing_only']=True
    cfg['HashGrid'].update(hash_size=args.hash_size,hash=ncu_spec['hash'],morton_sort=ncu_spec['morton_sort'])
    cfg['round2']=dict(condition=args.condition,seed=args.seed,engineering=False,seed_policy='once before construction; native thread RNG evolution',warmup_frames=20)
    cfg['round2'].update(ncu_variant=args.ncu_variant,kernel_profile_only=True)
    model_module.ROUND2_HASH=ncu_spec['hash']
    full_length=len(list((data/'depth').glob('depth_*.png')))
    native_length=datasets.Replica.__len__
    if args.frames:
        datasets.Replica.__len__=lambda self:min(args.frames,native_length(self))
    output.mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (output/'source_manifest.json').write_text((root/'EC_RUNTIME_MANIFEST.json').read_text())
    (output/'command.json').write_text(json.dumps(sys.argv,indent=2)+'\n')
    timer=Timer();queries=[];query_lock=threading.Lock()
    original=model_module.HashGrid.forward
    key_function=None
    if ncu_spec['morton_sort']:
        from torch.utils.cpp_extension import load
        major,minor=torch.cuda.get_device_capability();os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
        extension=load(name=f'dasip_round2_morton_{major}{minor}',sources=[str(co/'model/round2_morton/morton_cuda.cu')],extra_cuda_cflags=['-O3'],verbose=False)
        key_function=extension.morton3d_keys
    def query(self,points,onlySdf=False):
        if key_function is None:
            result=original(self,points,onlySdf)
        else:
            flat=points.reshape(-1,3)
            normalized=(flat-self.bounding_box[:,0])/(self.bounding_box[:,1]-self.bounding_box[:,0])
            with torch.no_grad():perm=torch.argsort(key_function(normalized.float().contiguous(),128),stable=False).long()
            values=original(self,flat[perm],onlySdf)
            result=torch.empty_like(values);result[perm]=values
        with query_lock:
            queries.append(dict(frame_id=getattr(timer.local,'frame',-1),stage=getattr(timer.local,'stage','construction'),iteration=getattr(timer.local,'iteration',-1),dense_queries=points.numel()//3,actual_queries=points.numel()//3))
        return result
    model_module.HashGrid.forward=query
    slam=EC_SLAM(cfg,SimpleNamespace(output=None,input_folder=None))
    slam.tracker._r2_timer=timer;slam.mapper._r2_timer=timer
    def wrap(method,stage):
        def call(*inputs,**kwargs):
            selected='first_frame_mapping' if stage=='bundle_adjustment' and int(inputs[0])==0 else stage
            with timer.call(selected):return method(*inputs,**kwargs)
        return call
    slam.tracker.optimize_cam_in_batch=wrap(slam.tracker.optimize_cam_in_batch,'tracking_optimizer_call')
    slam.mapper.optimize_map=wrap(slam.mapper.optimize_map,'bundle_adjustment')
    errors=[]
    def guarded(method,rank):
        try:method(rank)
        except BaseException:
            import traceback
            errors.append(traceback.format_exc())
    start=time.perf_counter()
    # Same two native workers, start order and internal wait/transfer locations.
    workers=[threading.Thread(target=guarded,args=(slam.tracking,0),daemon=True),threading.Thread(target=guarded,args=(slam.mapping,1),daemon=True)]
    for worker in workers:worker.start()
    next_power=0
    while any(w.is_alive() for w in workers):
        if errors:raise RuntimeError('\n'.join(errors))
        if time.monotonic()>=next_power:
            power();next_power=time.monotonic()+30
            print('EC_PROGRESS',int(slam.idx[0]),int(slam.mapping_idx[0]),flush=True)
        for worker in workers:worker.join(.2)
    if errors:raise RuntimeError('\n'.join(errors))
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    events=timer.save(output)
    from ec_timing import write_csv
    write_csv(output/'queries.csv',queries)
    frames=args.frames or full_length
    tracked=[r for r in timer.calls if r['stage']=='tracking_frame']
    assert len(tracked)==frames and {r['frame_id'] for r in tracked}==set(range(frames))
    assert int(slam.mapping_idx[0])==frames-1
    finite=all(bool(torch.isfinite(p).all()) for p in slam.vis_dict[0].vMapModel.parameters()) and bool(torch.isfinite(slam.estimate_c2w_list).all())
    if not finite:raise RuntimeError('Nonfinite EC model/poses')
    summary=dict(ncu_variant=args.ncu_variant,ncu_spec=ncu_spec,kernel_profile_only=True,system='EC-SLAM',dataset='Replica',scene=args.scene,condition=args.condition,seed=args.seed,hash_size=args.hash_size,frames=frames,full_sequence=frames==full_length,diagnostic=bool(args.frames),quality=False,finite_parameters_and_poses=finite,slam_work_wall_seconds=elapsed,warmup_frames=20,initial_power=initial_power,final_power=power(),device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_runtime=torch.version.cuda,iterations=len(events),timing_scope='Native overlapping tracker/mapper threads; synchronized complete tracking frame body includes transfers, mapping waits and pose finalization; mapper frame includes loader/keyframe work; bundle_adjustment covers optimize_map prep through pose commit. Shared default CUDA stream/global synchronization may include other-thread work; costs are non-additive; no complete-system FPS inferred.',native_thread_schedule_preserved=True)
    for stage in ('tracking_frame','mapping_frame','tracking_optimizer_call','bundle_adjustment'):
        all_rows=[r for r in timer.calls if r['stage']==stage];measured=[r for r in all_rows if r['frame_id']>=20]
        summary[stage]=dict(count=len(measured),mean_wall_ms=float(np.mean([r['wall_ms'] for r in measured])) if measured else None,mean_cuda_ms=float(np.mean([r['cuda_ms'] for r in measured])) if measured else None,all_call_mean_wall_ms=float(np.mean([r['wall_ms'] for r in all_rows])) if all_rows else None)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('EC_RESULT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
