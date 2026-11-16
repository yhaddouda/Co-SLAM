"""Isolated execution-only ports; scientific runtimes never import this file."""
import ast
import inspect
import random
import textwrap
import torch

def _replace(source,old,new):
    assert source.count(old)==1,old
    return source.replace(old,new)

def _preload(slam,current_rays):
    mapping=slam.config['mapping'];rays_all=[];ids_all=[]
    n=max(mapping['sample']//len(slam.keyframeDatabase.frame_ids),mapping['min_pixels_cur'])
    for _ in range(mapping['iters']):
        rays,ids=slam.keyframeDatabase.sample_global_rays(mapping['sample'])
        selected=random.sample(range(0,slam.dataset.H*slam.dataset.W),n)
        rays_all.append(torch.cat([rays,current_rays[selected,:]],dim=0))
        ids_all.append(torch.cat([ids//mapping['keyframe_every'],-torch.ones(len(selected))]).to(torch.int64))
    return torch.stack(rays_all).to(slam.device),torch.stack(ids_all)

def install_co(module):
    old=textwrap.dedent(inspect.getsource(module.CoSLAM.global_BA));new=old
    new=_replace(new,"    for i in range(self.config['mapping']['iters']):","    assert self.config['mapping']['iters'] % self.config['mapping']['pose_accum_step'] == 0\n    staged_rays, staged_ids = _preload(self, current_rays)\n    for i in range(self.config['mapping']['iters']):")
    begin=new.index("            rays, ids = self.keyframeDatabase.sample_global_rays")
    end=new.index("            rays_d = torch.sum",begin)
    new=new[:begin]+"            rays = staged_rays[i]\n            ids_all = staged_ids[i]\n            rays_d_cam = rays[..., :3]\n            target_s = rays[..., 3:6]\n            target_d = rays[..., 6:7]\n"+new[end:]
    new=_replace(new,"self.matrix_from_tensor(cur_rot[i:i + 1], cur_trans[i:i + 1]).detach().clone()[0]","pose_optim[i].detach().clone()")
    new=_replace(new,"self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]","pose_optim[-1].detach().clone()")
    namespace=dict(vars(module),_preload=_preload)
    exec(compile(ast.parse(new),'<isolated-co-engineering>','exec'),namespace)
    module.CoSLAM.global_BA=namespace['global_BA']
    return {'global_BA_before':old,'global_BA_after':new,'flags':{'grouped_ray_staging':True,'reuse_final_ba_pose_batch':True,'independent_pose_clones':True,'tracking_unchanged':True}}

def install_ec(tracker_module,mapper_module):
    # Keep native Tracker.run, transfers, waits, pose ownership and strict loss selection.
    # Only mapper repeated pose transfers/inverses and duplicate final CPU transfer change.
    old=textwrap.dedent(inspect.getsource(mapper_module.Mapper.optimize_map));new=old
    new=_replace(new,"    optimized_ids = []","    inverse_reference = torch.inverse(currentC2w)\n    optimized_ids = []")
    new=_replace(new,"self.pose_difference(currentC2w, estimatedC2w)","_pose_difference(self, inverse_reference, estimatedC2w)")
    new=_replace(new,"        BAPose = self.estimate_c2w_list.clone()[optIDs].to(self.device)","        pose_reference = self.estimate_c2w_list.clone().to(self.device)\n        device_opt_ids = torch.tensor(optIDs, device=self.device, dtype=torch.long)\n        BAPose = pose_reference[device_opt_ids]")
    new=_replace(new,"                currentEstimation = self.estimate_c2w_list.clone().to(self.device)\n                currentEstimation[optIDs] = c2w","                currentEstimation = pose_reference.index_copy(0, device_opt_ids, c2w)")
    new=_replace(new,"        self.estimate_c2w_list[optIDs] = c2w.cpu().clone()\n        self.estimate_relative_c2w_list[optIDs] = c2w.cpu().clone()","        committed = c2w.cpu().clone()\n        self.estimate_c2w_list[optIDs] = committed\n        self.estimate_relative_c2w_list[optIDs] = committed")
    namespace=dict(vars(mapper_module),_pose_difference=_pose_difference)
    exec(compile(ast.parse(new),'<isolated-ec-engineering>','exec'),namespace)
    mapper_module.Mapper.optimize_map=namespace['optimize_map']
    return {'optimize_map_before':old,'optimize_map_after':new,'flags':{'cache_mapping_pose_reference':True,'reuse_window_inverse':True,'single_final_pose_transfer':True,'native_tracker_unchanged':True,'native_wait_transfer_order_unchanged':True}}

def _pose_difference(self,inverse_reference,estimated):
    # Same native arithmetic and CPU pose conversion, with invariant inverse reused.
    import numpy as np
    from src.utils import get_tensor_from_camera
    diff_quad=get_tensor_from_camera(inverse_reference@estimated)
    t=diff_quad[-3:];quad=diff_quad[:4]
    return torch.norm(t).item(),2*np.arccos(abs(torch.dot(quad,torch.tensor([1,0,0,0],dtype=torch.float32))))*(180/np.pi)
