"""Cyclic complements: coverage and actual native conditional objective."""
import pickle
import random
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
import torch
import pruning_cpu_fixtures as ray_tests
from optimization.ray_selection import cyclic_ray_subsets
from optimization.sample_pruning import pruning_policy


class RayCycleTests(unittest.TestCase):
    def test_complete_cycle_has_exact_balanced_inclusion_and_private_rng(self):
        py_state=random.getstate();np_state=pickle.dumps(np.random.get_state())
        torch_state=torch.get_rng_state()
        for n in (*range(2,18),1024):
            x=np.arange(n,dtype=float)
            metadata=np.column_stack((np.sin(x),np.cos(x),np.ones(n),x[::-1]))
            subsets=cyclic_ray_subsets(n,23,metadata)
            again=cyclic_ray_subsets(n,23,metadata)
            counts=np.zeros(n,dtype=int)
            for indices,other in zip(subsets,again):
                self.assertEqual(indices.dtype,np.int64)
                self.assertTrue(np.all(np.diff(indices)>0))
                np.testing.assert_array_equal(indices,other)
                counts[indices]+=1
            np.testing.assert_array_equal(counts,np.full(n,len(subsets)-1))
            self.assertLessEqual(max(map(len,subsets))-min(map(len,subsets)),1)
            if n==1024:self.assertEqual([len(x) for x in subsets],[768]*4)
        self.assertEqual(random.getstate(),py_state)
        self.assertEqual(pickle.dumps(np.random.get_state()),np_state)
        torch.testing.assert_close(torch.get_rng_state(),torch_state)
        np.testing.assert_array_equal(cyclic_ray_subsets(1,0)[0],[0])

    def test_native_conditional_losses_gradients_and_fresh_cycle_coverage(self):
        with patch('torch.cuda.nvtx.range',lambda *a,**k:nullcontext()):
            helper=ray_tests.RayFieldFixture()
            model=helper.model()
            model.config['sample_pruning'].update(policy='ray_cohort',cohort_tracking='ray_cycle')
            args=helper.inputs((0.,1.2,2.3,6.,1.1,2.5,0.,3.1))
            model._ray_sampling_metadata=torch.cat((args[1].detach(),args[2],args[3]),dim=-1)
            count=np.zeros(8,dtype=int);stored=None
            for iteration in range(4):
                model._pruning_iteration=iteration
                self.assertEqual(pruning_policy(model),'ray_cohort')
                self.assertEqual(model._pruning_iteration,iteration)
                torch.manual_seed(197)
                actual=model.forward(*args)
                state=torch.get_rng_state()
                indices=model._ray_cycle_selection[1][iteration]
                count[indices.numpy()]+=1
                if stored is not None:self.assertIs(model._ray_cycle_selection,stored)
                stored=model._ray_cycle_selection
                expected=helper.dense(model,tuple(value[indices] for value in args))
                torch.testing.assert_close(torch.get_rng_state(),state)
                for key in (*ray_tests.LOSSES,'rgb','depth'):
                    torch.testing.assert_close(actual[key],expected[key],rtol=3e-5,atol=2e-6)
                parameters=(model.coefficients,args[0],args[1])
                ga=torch.autograd.grad(sum(actual[k] for k in ray_tests.LOSSES),parameters)
                gb=torch.autograd.grad(sum(expected[k] for k in ray_tests.LOSSES),parameters)
                for a,b in zip(ga,gb):torch.testing.assert_close(a,b,rtol=5e-5,atol=2e-5)
                row=model.pruning_diagnostics[-1]
                self.assertEqual(row['selected_rays'],6)
                self.assertEqual(row['queried_samples']*4,row['dense_samples']*3)
            np.testing.assert_array_equal(count,np.full(8,3))
            model.config['sample_pruning']['tracking_anchor_every']=3
            model._pruning_iteration=0
            self.assertIsNone(pruning_policy(model))
            model._pruning_stage='Bundle_Adjustment';model._pruning_iteration=0
            self.assertEqual(pruning_policy(model),'ray_cohort')


if __name__=='__main__':unittest.main()
