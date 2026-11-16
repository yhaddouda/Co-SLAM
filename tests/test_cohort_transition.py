"""Maturation-band probabilities, exact estimators, and actual BA scheduling."""
import copy
import random
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import numpy as np
import torch
import test_cohort_selection as pose_tests
import test_cohort_forward as forward_tests
from optimization.cohort_selection import choose_pose_cohort
from optimization.sample_pruning import forward_cohort_ba, validate_pruning


class CohortTransitionTests(unittest.TestCase):
    def setUp(self):
        nvtx=patch('torch.cuda.nvtx.range',lambda *a,**k:nullcontext())
        nvtx.start();self.addCleanup(nvtx.stop)

    def test_phase_measure_has_anchor_transition_and_mature_inclusion(self):
        for count in (4,6,8):
            population=23;frequencies=np.zeros(population,dtype=int);estimates=[]
            for seed in range(count):
                active,p,phase=choose_pose_cohort(population,115,5,count,2,seed,
                                                transition_keyframes=2)
                self.assertEqual(phase,(23+seed)%count)
                frequencies[active]+=1
                estimate=np.zeros(population);estimate[active]=1./p[active]
                estimates.append(estimate)
                np.testing.assert_array_equal(p[[0,21,22]],np.ones(3))
                np.testing.assert_array_equal(p[[19,20]],np.full(2,.5))
                np.testing.assert_allclose(p[1:19],np.full(18,1./count))
            np.testing.assert_array_equal(frequencies[[0,21,22]],np.full(3,count))
            np.testing.assert_array_equal(frequencies[[19,20]],np.full(2,count//2))
            np.testing.assert_array_equal(frequencies[1:19],np.ones(18))
            np.testing.assert_allclose(np.mean(estimates,axis=0),np.ones(population),rtol=1e-7)

    def test_transition_zero_restores_original_selector_exactly(self):
        for n in range(24):
            for count in (1,3,4,6):
                for recent in (0,2,4):
                    for seed in (0,1,2):
                        default=choose_pose_cohort(n,50,5,count,recent,seed)
                        explicit=choose_pose_cohort(n,50,5,count,recent,seed,transition_keyframes=0)
                        indices=np.arange(n,dtype=np.int64)
                        anchor=(indices==0)|(indices>=max(1,n-recent))
                        phase=(10+seed)%count
                        expected=indices[anchor|(indices%count==phase)]
                        probability=np.full(n,1./count,dtype=np.float32);probability[anchor]=1.
                        for result in (default,explicit):
                            np.testing.assert_array_equal(result[0],expected)
                            np.testing.assert_array_equal(result[1],probability)
                            self.assertEqual(result[2],phase)

    def test_all_seed_phases_get_bounded_maturation_and_mature_revisit(self):
        keyframe=11
        for seed in range(4):
            selected=[]
            for age in range(1,25):
                n=keyframe+age
                active,p,_=choose_pose_cohort(n,n*5,5,4,2,seed,transition_keyframes=2)
                if keyframe in active:selected.append(age)
                self.assertEqual(p[keyframe],1. if age<=2 else .5 if age<=4 else .25)
            self.assertEqual(selected[:2],[1,2])
            self.assertIn(selected[2],(3,4))
            self.assertLessEqual(np.diff(selected).max(),4)

    def test_invalid_transition_configuration_rejected(self):
        base=dict(training={},sample_pruning=dict(enabled=True,policy='ray_cohort',
                  cohort_count=4,recent_keyframes=2,cohort_transition_keyframes=2))
        validate_pruning(base)
        for field,value in (('cohort_count',2),('cohort_count',3),('cohort_count',5),
                            ('cohort_transition_keyframes',-1),('cohort_transition_keyframes',True),
                            ('cohort_transition_keyframes',1.5),('cohort_transition_period',1),
                            ('cohort_transition_period',4)):
            bad=copy.deepcopy(base);bad['sample_pruning'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                validate_pruning(bad)
        for count in (1,2,3,5):
            with self.assertRaises(ValueError):
                choose_pose_cohort(10,50,5,count,2,0,transition_keyframes=2)

    def test_four_phase_actual_loss_and_gradient_expectation(self):
        fixture=forward_tests.CohortForwardTests()
        fixture.helper=forward_tests.ray_tests.RayFieldFixture()
        for mode in ('oscillating','late_surface'):
            model,args,groups=fixture.setup(mode)
            model.config['sample_pruning']['cohort_transition_keyframes']=2
            dense=fixture.helper.dense(model,args);native_rng=torch.get_rng_state()
            values=[]
            for seed in range(4):
                active,p,_=choose_pose_cohort(8,40,5,4,2,seed,transition_keyframes=2)
                model._cohort_active_ids=active;model._cohort_probabilities=p
                torch.manual_seed(197)
                actual=forward_cohort_ba(model,*args);values.append(actual)
                torch.testing.assert_close(torch.get_rng_state(),native_rng)
                row=model.pruning_diagnostics[-1]
                self.assertEqual(row['cohort_old_rays'],row['cohort_transition_rays']+row['cohort_mature_rays'])
                self.assertEqual(row['weight_sum'],row['anchor_rays']+2*row['cohort_transition_rays']+4*row['cohort_mature_rays'])
                selected=(groups<0)|torch.from_numpy(np.isin(groups.numpy(),active))
                self.assertEqual(row['selected_rays'],int(selected.sum()))
                self.assertEqual(row['queried_samples'],len(model.queried_points[-1]))
                for gradient in torch.autograd.grad(sum(actual[k] for k in forward_tests.LOSSES),
                                                     (args[0],args[1]),retain_graph=True):
                    self.assertEqual(float(gradient[~selected].abs().sum()),0.)
            parameters=(model.coefficients,args[0],args[1])
            for key in forward_tests.LOSSES:
                mean=torch.stack([value[key] for value in values]).mean()
                torch.testing.assert_close(mean,dense[key],rtol=3e-5,atol=2e-6)
                actual=torch.autograd.grad(mean,parameters,retain_graph=True)
                expected=torch.autograd.grad(dense[key],parameters,retain_graph=True)
                for a,b in zip(actual,expected):
                    torch.testing.assert_close(a,b,rtol=5e-5,atol=2e-5)

    def test_actual_ba_optimizes_selected_transition_poses_only(self):
        slam=pose_tests.CPUSlam(n=10,count=4)
        slam.config['sample_pruning'].update(seed=2,cohort_transition_keyframes=2)
        before={frame:pose.clone() for frame,pose in slam.est_c2w_data.items()}
        slam.global_BA(pose_tests.CohortBATests().batch(),50)
        audit=slam._cohort_pose_diagnostics[-1]
        self.assertEqual(audit['active_history_indices'],[0,4,6,8,9])
        self.assertEqual(audit['cohort_transition_keyframes'],2)
        self.assertEqual(audit['cohort_transition_period'],2)
        optimized={4*5,6*5,8*5,9*5,50}
        for frame,pose in slam.est_c2w_data.items():
            self.assertEqual(torch.equal(pose,before[frame]),frame not in optimized)
        self.assertEqual(slam.sample_calls,[10,10])
        self.assertEqual(slam.model.full_batch_sizes,[11,11])

    def test_actual_ba_transition_zero_is_bitwise_native_cohort_path(self):
        default=pose_tests.CPUSlam(n=10,count=4)
        zero=pose_tests.CPUSlam(n=10,count=4)
        zero.config['sample_pruning']['cohort_transition_keyframes']=0
        batch=pose_tests.CohortBATests().batch()
        random.seed(17);default.global_BA(batch,50)
        random.seed(17);zero.global_BA(batch,50)
        for frame in default.est_c2w_data:
            torch.testing.assert_close(default.est_c2w_data[frame],zero.est_c2w_data[frame],rtol=0,atol=0)
        torch.testing.assert_close(default.model.parameter,zero.model.parameter,rtol=0,atol=0)
        self.assertEqual(default._cohort_pose_diagnostics,zero._cohort_pose_diagnostics)


if __name__=='__main__':unittest.main()
