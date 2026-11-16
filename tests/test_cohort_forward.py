"""Finite-population loss calibration and inactive-ray gradients for BA cohorts."""
import unittest
from contextlib import nullcontext
from unittest.mock import patch
import numpy as np
import torch
import pruning_cpu_fixtures as ray_tests
LOSSES=ray_tests.LOSSES
from optimization.sample_pruning import forward_cohort_ba,validate_pruning

class CohortForwardTests(unittest.TestCase):
 def setUp(self):
  self.patch=patch('torch.cuda.nvtx.range',lambda *a,**k:nullcontext());self.patch.start();self.addCleanup(self.patch.stop)
  self.helper=ray_tests.RayFieldFixture()
 def setup(self,mode='oscillating'):
  m=self.helper.model(mode=mode,stage='Bundle_Adjustment');m.config['sample_pruning']['policy']='ray_cohort'
  args=self.helper.inputs((0.,1.2,2.3,6.,.8,2.1,0.,3.2,1.3,4.4))
  groups=torch.tensor([-1,0,1,2,3,4,5,6,7,7]);m._ray_group_ids=groups
  p=np.full(8,.25,dtype=np.float32);p[[0,6,7]]=1.;m._cohort_probabilities=p
  return m,args,groups
 def test_complete_phase_cycle_recovers_native_objective_and_gradients(self):
  for mode in ('oscillating','no_crossing','late_surface'):
   m,args,g=self.setup(mode);dense=self.helper.dense(m,args);rng=torch.get_rng_state();values=[]
   for phase in range(4):
    m._cohort_active_ids=np.array([v for v in range(8) if v in (0,6,7) or v%4==phase])
    torch.manual_seed(197);actual=forward_cohort_ba(m,*args);values.append(actual)
    torch.testing.assert_close(torch.get_rng_state(),rng)
    mask=(g<0)|torch.tensor(np.isin(g.numpy(),m._cohort_active_ids))
    count=int(mask.sum());diag=m.pruning_diagnostics[-1]
    self.assertEqual(diag['selected_rays'],count)
    self.assertEqual(diag['queried_samples'],m.queried_points[-1].numel()//3)
    self.assertEqual(diag['weight_sum'],4*diag['cohort_old_rays']+diag['anchor_rays'])
    grads=torch.autograd.grad(sum(actual[k] for k in LOSSES),(args[0],args[1]),retain_graph=True)
    for gradient in grads:self.assertEqual(float(gradient[~mask].abs().sum()),0.)
   for key in LOSSES:
    mean=torch.stack([v[key] for v in values]).mean()
    torch.testing.assert_close(mean,dense[key],rtol=3e-5,atol=2e-6)
    ga=torch.autograd.grad(mean,(m.coefficients,args[0],args[1]),retain_graph=True)
    gb=torch.autograd.grad(dense[key],(m.coefficients,args[0],args[1]),retain_graph=True)
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=5e-5,atol=2e-5)
 def test_all_groups_restore_native(self):
  m,args,_=self.setup();m._cohort_probabilities=np.ones(8,dtype=np.float32);m._cohort_active_ids=np.arange(8)
  dense=self.helper.dense(m,args);state=torch.get_rng_state();torch.manual_seed(197);actual=forward_cohort_ba(m,*args)
  torch.testing.assert_close(torch.get_rng_state(),state)
  for k in (*LOSSES,'rgb','depth'):torch.testing.assert_close(actual[k],dense[k],rtol=3e-5,atol=2e-6)
 def test_native_weighting_matches_selected_native_renderer_losses_and_gradients(self):
  # Use the actual native forward as the independent reference. Supply its
  # K-ray jitter from the selected rows of the original N-ray draw, so the
  # reference exercises native selected-batch masks/normalizers without
  # duplicating the implementation of any loss formula.
  for retain_all in (False,True):
   for mode in ('oscillating','no_crossing','late_surface'):
    with self.subTest(retain_all=retain_all,mode=mode):
     m,args,g=self.setup(mode)
     m.config['sample_pruning']['cohort_weighting']='native'
     m._cohort_active_ids=np.arange(8) if retain_all else np.array([0,4,6,7])
     if retain_all:m._cohort_probabilities=np.ones(8,dtype=np.float32)
     selected=torch.tensor(np.flatnonzero((g.numpy()<0)|np.isin(g.numpy(),m._cohort_active_ids)))
     native_rand=torch.rand;calls=[]
     def selected_full_jitter(size,*positional,**keywords):
      calls.append(tuple(size))
      self.assertEqual(size[0],len(selected))
      return native_rand((len(g),size[1]),*positional,**keywords)[selected]
     m.config['sample_pruning']['enabled']=False
     torch.manual_seed(197)
     with patch('torch.rand',side_effect=selected_full_jitter):
      expected=m.forward(*(value[selected] for value in args))
     expected_points=m.queried_points[-1].clone();state=torch.get_rng_state()
     self.assertEqual(len(calls),1)
     m.config['sample_pruning']['enabled']=True
     torch.manual_seed(197)
     actual=forward_cohort_ba(m,*args)
     torch.testing.assert_close(torch.get_rng_state(),state)
     torch.testing.assert_close(m.queried_points[-1],expected_points,rtol=0,atol=0)
     for key in (*LOSSES,'rgb','depth'):
      torch.testing.assert_close(actual[key],expected[key],rtol=3e-5,atol=2e-6)
     parameters=(m.coefficients,args[0],args[1])
     for key in LOSSES:
      a=torch.autograd.grad(actual[key],parameters,retain_graph=True)
      b=torch.autograd.grad(expected[key],parameters,retain_graph=True)
      for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=5e-5,atol=2e-5)
     row=m.pruning_diagnostics[-1]
     self.assertEqual(row['weight_sum'],len(selected))
     self.assertEqual(row['selected_rays'],len(selected))
     self.assertEqual(row['queried_samples'],len(selected)*calls[0][1])
     if retain_all:
      dense=self.helper.dense(m,args)
      for key in (*LOSSES,'rgb','depth'):
       torch.testing.assert_close(actual[key],dense[key],rtol=3e-5,atol=2e-6)
 def test_invalid_cohort_configuration_rejected(self):
  for name,value in [('cohort_count',0),('cohort_count',True),('cohort_count',2.5),('recent_keyframes',-1)]:
   m,_,_=self.setup();m.config['sample_pruning'][name]=value
   with self.assertRaises(ValueError):validate_pruning(m.config)
if __name__=='__main__':unittest.main()
