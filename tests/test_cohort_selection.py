"""Rotating cohort measure and actual CPU BA pose-block scheduling checks."""
import ast
import copy
from contextlib import nullcontext
from pathlib import Path
import pickle
import random
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from optimization.cohort_selection import choose_pose_cohort
from optimization.ba_experiments import ba_bucket_mode, configure_ba_embed_eps


class CohortSelectionTests(unittest.TestCase):
    def test_four_phases_cover_every_old_pose_once_and_anchors_always(self):
        n=23;recent=4;count=4
        frequencies=np.zeros(n,dtype=np.int64)
        estimates=[];phases=[]
        for step in range(count):
            active,probability,phase=choose_pose_cohort(n,500+5*step,5,count,recent,7)
            self.assertEqual(active.dtype,np.int64);self.assertEqual(probability.dtype,np.float32)
            self.assertTrue(np.all(probability>0))
            self.assertTrue(np.all(np.diff(active)>0))
            self.assertTrue(set([0,*range(n-recent,n)]).issubset(active.tolist()))
            frequencies[active]+=1
            estimate=np.zeros(n);estimate[active]=1./probability[active]
            estimates.append(estimate);phases.append(phase)
        np.testing.assert_array_equal(frequencies[1:n-recent],np.ones(n-recent-1))
        np.testing.assert_array_equal(frequencies[[0,*range(n-recent,n)]],np.full(recent+1,count))
        np.testing.assert_allclose(np.mean(estimates,axis=0),np.ones(n),rtol=0,atol=0)
        self.assertEqual(sorted(phases),list(range(count)))

    def test_full_count_and_small_populations_keep_all_poses(self):
        for n in (0,1,2,7,100):
            active,probability,phase=choose_pose_cohort(n,50,5,1,4,3)
            np.testing.assert_array_equal(active,np.arange(n))
            np.testing.assert_array_equal(probability,np.ones(n))
            self.assertEqual(phase,0)
        for n in range(6):
            active,probability,_=choose_pose_cohort(n,50,5,4,4,3)
            np.testing.assert_array_equal(active,np.arange(n))
            np.testing.assert_array_equal(probability,np.ones(n))

    def test_zero_recent_and_global_rng_untouched(self):
        py_state=random.getstate();np_state=pickle.dumps(np.random.get_state())
        active,probability,phase=choose_pose_cohort(10,50,5,4,0,0)
        self.assertIn(0,active);self.assertEqual(probability[0],1.)
        np.testing.assert_array_equal(probability[1:],np.full(9,.25))
        self.assertEqual(py_state,random.getstate())
        self.assertEqual(np_state,pickle.dumps(np.random.get_state()))
        self.assertEqual(phase,2)

    def test_invalid_inputs_rejected(self):
        base=dict(num_keyframes=10,frame=50,map_every=5,count=4,recent=4,seed=0)
        for name,value in (('num_keyframes',-1),('frame',-1),('map_every',0),
                           ('count',0),('recent',-1),('seed',1.5),('count',True)):
            bad=dict(base);bad[name]=value
            with self.assertRaises(ValueError):choose_pose_cohort(**bad)


class Timer:
    def stage(self,*args):return nullcontext()
    def set_context(self,*args):pass
    def flush(self):pass


class CPUModel:
    def __init__(self,slam):
        self.slam=slam;self.parameter=torch.nn.Parameter(torch.tensor(.4));self.full_batch_sizes=[]
    def set_timing_context(self,*args):pass
    def flush_scene_timing(self):pass
    def forward(self,origins,directions,color,depth,**kwargs):
        self.full_batch_sizes.append(len(depth))
        residual=(origins-.8).square().sum(1)+directions.square().sum(1)
        if self.slam.config['sample_pruning']['policy']=='ray_cohort':
            ids=self._ray_group_ids.numpy()
            keep=(ids==-1)|np.isin(ids,self._cohort_active_ids)
            probability=np.ones(len(ids),dtype=np.float32)
            historical=ids>=0
            probability[historical]=self._cohort_probabilities[ids[historical]]
            value=(residual[torch.from_numpy(keep)]/torch.from_numpy(probability[keep])).sum()/len(depth)
        else:value=residual.mean()
        return dict(loss=value+self.parameter.square())


class CPUSlam:
    def __init__(self,n=10,count=4,optim_cur=True):
        self.device=torch.device('cpu');self.dataset=SimpleNamespace(H=1,W=1)
        self.config=dict(mapping=dict(keyframe_every=5,map_every=5,iters=2,sample=n,
            min_pixels_cur=1,optim_cur=optim_cur,map_accum_step=1,map_wait_step=0,
            pose_accum_step=1,lr_rot=.001,lr_trans=.001),grid=dict(morton_sort=False),
            sample_pruning=dict(enabled=True,policy='ray_cohort',cohort_count=count,recent_keyframes=2,seed=0))
        self.est_c2w_data={}
        for frame in range(0,n*5+1,5):
            pose=torch.eye(4);pose[:3,3]=torch.tensor([frame*.01,.1,.2]);self.est_c2w_data[frame]=pose
        self.keyframeDatabase=SimpleNamespace(frame_ids=list(range(0,n*5,5)),sample_global_rays=self.sample_rays)
        self.sample_calls=[];self.pose_parameter_counts=[]
        self.model=CPUModel(self)
        self.map_optimizer=torch.optim.Adam([self.model.parameter],lr=.01)
        self.coslam_timing=Timer()
    def sample_rays(self,count):
        self.sample_calls.append(count)
        rays=torch.tensor([.1,.2,1.,.3,.4,.5,1.]).repeat(count,1)
        return rays,torch.arange(count,dtype=torch.long)*5
    def get_pose_param_optim(self,poses):
        self.pose_parameter_counts.append(len(poses))
        rotation=torch.nn.Parameter(torch.zeros(len(poses),3))
        translation=torch.nn.Parameter(poses[:,:3,3].detach().clone())
        return rotation,translation,torch.optim.Adam([
            dict(params=rotation,lr=self.config['mapping']['lr_rot']),
            dict(params=translation,lr=self.config['mapping']['lr_trans'])])
    def matrix_from_tensor(self,rotation,translation):
        # A differentiable CPU stand-in isolates bookkeeping from SE(3) math.
        top=torch.cat((torch.eye(3)[None,:,:].expand(len(rotation),-1,-1)+torch.diag_embed(rotation),
                       translation[:,:,None]),dim=-1)
        bottom=torch.tensor([0.,0.,0.,1.])[None,None,:].expand(len(rotation),-1,-1)
        return torch.cat((top,bottom),dim=1)
    def get_loss_from_ret(self,ret,smooth=False):return ret['loss']
    def start_profiling(self,**kwargs):pass
    def step_profiler(self):pass


def actual_method(name):
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/'coslam.py').read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='CoSLAM')
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)
    namespace=dict(torch=torch,np=np,random=random,record_function=lambda *a:nullcontext(),
        nvtx_range=lambda *a:nullcontext(),ba_bucket_mode=ba_bucket_mode,
        configure_ba_embed_eps=configure_ba_embed_eps)
    exec(compile(ast.Module(body=[method],type_ignores=[]),'actual_coslam_'+name,'exec'),namespace)
    return namespace[name]

CPUSlam.global_BA=actual_method('global_BA')
CPUSlam.global_BA_cohort=actual_method('global_BA_cohort')


class CohortBATests(unittest.TestCase):
    def batch(self):
        return dict(direction=torch.tensor([[[[.1,.2,1.]]]]),
            rgb=torch.full((1,1,1,3),.5),depth=torch.ones(1,1,1))

    def test_only_active_pose_blocks_change_and_full_ray_draws_remain(self):
        for optim_cur in (False,True):
            with self.subTest(optim_cur=optim_cur):
                slam=CPUSlam(optim_cur=optim_cur)
                before={frame:pose.clone() for frame,pose in slam.est_c2w_data.items()}
                slam.global_BA(self.batch(),50)
                audit=slam._cohort_pose_diagnostics[-1]
                optimized={index*5 for index in audit['optimized_history_indices']}
                if optim_cur:optimized.add(50)
                for frame,pose in slam.est_c2w_data.items():
                    self.assertEqual(torch.equal(pose,before[frame]),frame not in optimized)
                self.assertEqual(slam.pose_parameter_counts,[len(optimized)])
                self.assertEqual(slam.sample_calls,[10,10])
                self.assertEqual(slam.model.full_batch_sizes,[11,11])
                self.assertEqual(audit['optimized_pose_count'],len(optimized))
                self.assertEqual(audit['optim_cur'],optim_cur)

    def test_count_one_matches_actual_native_ba_results(self):
        for n in (1,4,10):
            for optim_cur in (False,True):
                with self.subTest(history=n,optim_cur=optim_cur):
                    cohort=CPUSlam(n=n,count=1,optim_cur=optim_cur)
                    native=CPUSlam(n=n,count=1,optim_cur=optim_cur)
                    native.config['sample_pruning'].update(enabled=False,policy='dense')
                    random.seed(17);cohort.global_BA(self.batch(),n*5)
                    random.seed(17);native.global_BA(self.batch(),n*5)
                    for frame in native.est_c2w_data:
                        torch.testing.assert_close(cohort.est_c2w_data[frame],native.est_c2w_data[frame])
                    torch.testing.assert_close(cohort.model.parameter,native.model.parameter)
                    self.assertEqual(cohort.pose_parameter_counts,native.pose_parameter_counts)
                    self.assertEqual(cohort.sample_calls,native.sample_calls)
                    self.assertEqual(cohort.model.full_batch_sizes,native.model.full_batch_sizes)

if __name__=='__main__':unittest.main(verbosity=2)
