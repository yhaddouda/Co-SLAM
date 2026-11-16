"""Scientific checks for query reduction, objective compensation and guards."""
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from test_ba_experiments import AnalyticField
from model.utils import get_sdf_loss
from optimization.sample_pruning import (lattice_selection, weighted_geometry,
    forward_pruned, render_weights, validate_pruning, feature_uniform_pixels,
    refine_surface_lattice)


from optimization.sample_pruning import forward_supported, pruning_policy

class SamplePruningTests(unittest.TestCase):
    def test_supported_without_pruning_matches_native_values_gradients_rng(self):
        for surface in (2.03,20.):
            m=self.model('support')
            m.surface=torch.tensor(surface,requires_grad=True)
            m.config['training']['perturb']=1
            m.config['sample_pruning']['margin']=1000000.
            origin=torch.zeros(3,3,requires_grad=True)
            direction=torch.tensor([[0.,0.,1.]]).repeat(3,1)
            color=torch.full((3,3),.5)
            depth=torch.tensor([[2.],[0.],[3.]])
            m.config['sample_pruning']['enabled']=False
            torch.manual_seed(11)
            dense=m.forward(origin,direction,color,depth)
            state=torch.random.get_rng_state()
            m.config['sample_pruning']['enabled']=True
            torch.manual_seed(11)
            sparse=forward_supported(m,origin,direction,color,depth)
            torch.testing.assert_close(torch.random.get_rng_state(),state)
            keys=('rgb_loss','depth_loss','sdf_loss','fs_loss')
            for key in (*keys,'rgb','depth'):
                torch.testing.assert_close(sparse[key],dense[key],rtol=3e-5,atol=1e-6)
            ga=torch.autograd.grad(sum(sparse[k] for k in keys),(origin,m.surface))
            gb=torch.autograd.grad(sum(dense[k] for k in keys),(origin,m.surface))
            for a,b in zip(ga,gb):
                torch.testing.assert_close(a,b,rtol=3e-5,atol=1e-6)

    def test_supported_preserves_all_geometry_and_missing_depth_queries(self):
        from test_ba_guarded_fields import CurvedField
        for mode in ('oscillating','late_surface','no_crossing','boundary_surface'):
            m=CurvedField(mode)
            m.config['training']['perturb']=1
            m.config['sample_pruning']=dict(enabled=True,policy='support',margin=.15,anchor_every=2)
            m._pruning_stage,m._pruning_frame,m._pruning_iteration='Tracking',0,1
            origin=torch.tensor([[0.,0.,0.],[0.,1.,0.],[0.,2.,0.],[0.,3.,0.]],requires_grad=True)
            direction=torch.tensor([[0.,0.,1.]]).repeat(4,1).requires_grad_()
            depth=torch.tensor([[2.],[3.],[0.],[1.]])
            color=torch.full((4,3),.5)
            m.config['sample_pruning']['enabled']=False
            torch.manual_seed(53)
            dense=m.forward(origin,direction,color,depth)
            calls=[]
            network=m.run_network
            m.run_network=lambda x: (calls.append(x.detach()),network(x))[1]
            m.config['sample_pruning']['enabled']=True
            torch.manual_seed(53)
            sparse=forward_supported(m,origin,direction,color,depth)
            for key in ('fs_loss','sdf_loss'):
                torch.testing.assert_close(sparse[key],dense[key],rtol=3e-5,atol=1e-6)
            ga=torch.autograd.grad(sparse['fs_loss']+sparse['sdf_loss'],(origin,direction,m.coefficients))
            gb=torch.autograd.grad(dense['fs_loss']+dense['sdf_loss'],(origin,direction,m.coefficients))
            for a,b in zip(ga,gb):
                torch.testing.assert_close(a,b,rtol=3e-5,atol=1e-6)
            self.assertEqual(len(calls),1)
            count=m.config['training']['n_samples_d']+m.config['training']['n_range_d']
            self.assertEqual(int((calls[0][:,1]==2).sum()),count)
            self.assertLess(len(calls[0]),4*count)
            self.assertEqual(m.pruning_diagnostics[-1]['queried_samples'],len(calls[0]))

    def test_supported_interval_identities_stay_fixed_but_points_update(self):
        m=self.model('support')
        m.config['training']['perturb']=1
        m.config['sample_pruning'].update(margin=.15,anchor_every=0)
        depth=torch.tensor([[2.],[3.]])
        rgb=torch.full((2,3),.5)
        direction=torch.tensor([[0.,0.,1.]]).repeat(2,1)
        calls=[]
        network=m.run_network
        m.run_network=lambda x,**kwargs:(calls.append(x.detach().clone()),network(x,**kwargs))[1]
        origin=torch.zeros(2,3)
        torch.manual_seed(1)
        forward_supported(m,origin,direction,rgb,depth)
        selection=m._support_selection
        torch.manual_seed(2)
        shifted=origin+.01
        sparse=forward_supported(m,shifted,direction,rgb,depth)
        self.assertIs(m._support_selection,selection)
        self.assertEqual(len(calls),2)
        self.assertEqual(calls[0].shape,calls[1].shape)
        self.assertFalse(torch.equal(calls[0],calls[1]))
        m.config['sample_pruning']['enabled']=False
        torch.manual_seed(2)
        dense=m.forward(shifted,direction,rgb,depth)
        for key in ('fs_loss','sdf_loss'):
            torch.testing.assert_close(sparse[key],dense[key],rtol=3e-5,atol=1e-6)

    def test_supported_periodic_full_lattice_schedule(self):
        m=self.model('support')
        m.config['sample_pruning']['anchor_every']=2
        actual=[]
        for iteration in range(m.config['tracking']['iter']):
            m._pruning_iteration=iteration
            actual.append(pruning_policy(m))
        self.assertEqual(actual,[None,'support',None,'support',None,'support',None,'support',None,None])
        m.config['sample_pruning']['anchor_every']=0
        actual=[]
        for iteration in range(m.config['tracking']['iter']):
            m._pruning_iteration=iteration
            actual.append(pruning_policy(m))
        self.assertEqual(actual,[None]+['support']*8+[None])

    def setUp(self):
        nvtx = patch('torch.cuda.nvtx.range', lambda *a, **k: nullcontext())
        nvtx.start()
        self.addCleanup(nvtx.stop)

    def model(self, policy='tail', constant=False):
        m = AnalyticField(constant=constant)
        m.config['sample_pruning'] = dict(enabled=True, policy=policy, margin=0., free_stride=2)
        m._pruning_stage, m._pruning_frame, m._pruning_iteration = 'Tracking', 3, 0
        return m

    def test_tail_matches_dense_losses_and_pose_map_gradients(self):
        for perturb in (0, 1):
            m = self.model()
            m.config['training']['perturb'] = perturb
            origin = torch.zeros(4, 3, requires_grad=True)
            direction = torch.tensor([[0., .1, 1.]]).repeat(4, 1)
            depth = torch.tensor([[2.], [2.5], [0.], [1.9]])
            rgb = torch.full((4, 3), .5)
            m.config['sample_pruning']['enabled'] = False
            torch.manual_seed(53)
            dense = m.forward(origin, direction, rgb, depth)
            m.config['sample_pruning']['enabled'] = True
            torch.manual_seed(53)
            sparse = forward_pruned(m, origin, direction, rgb, depth)
            keys = ('rgb_loss', 'depth_loss', 'sdf_loss', 'fs_loss')
            for key in (*keys, 'rgb', 'depth'):
                torch.testing.assert_close(sparse[key], dense[key], rtol=2e-5, atol=1e-6)
            a = torch.autograd.grad(sum(sparse[k] for k in keys), (origin, m.surface))
            b = torch.autograd.grad(sum(dense[k] for k in keys), (origin, m.surface))
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, rtol=3e-5, atol=1e-6)
            counts = m.pruning_diagnostics[-1]
            self.assertLess(counts['queried_samples'], counts['dense_samples'])

    def test_ht_geometry_is_unbiased_over_all_sampling_phases(self):
        z = torch.linspace(0, 5, 47).repeat(3, 1)
        depth = torch.tensor([[2.], [3.], [0.]])
        param = torch.tensor(.3, requires_grad=True)
        sdf = param * z.sin()
        expected = get_sdf_loss(z, depth, sdf, .1, 'l2')
        losses = []
        for phase in range(4):
            with patch('torch.randint', return_value=torch.full((3, 1), phase)):
                keep, prob = lattice_selection(z, depth, .1, .15, 4, 0)
            losses.append(weighted_geometry(z, depth, sdf, keep, prob, .1))
        actual = [torch.stack([a[k] for a in losses]).mean() for k in (0, 1)]
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
            ga = torch.autograd.grad(a, param, retain_graph=True)[0]
            gb = torch.autograd.grad(b, param, retain_graph=True)[0]
            torch.testing.assert_close(ga, gb)

    def test_tail_on_curved_multiple_crossing_and_late_surfaces(self):
        from test_ba_guarded_fields import CurvedField
        for mode in ('oscillating','late_surface','no_crossing','boundary_surface'):
            for scale in (.4, 1., 2.):
                m=CurvedField(mode)
                m.config['training']['perturb']=1
                m.config['data']['sc_factor']=scale
                m._pruning_stage,m._pruning_frame,m._pruning_iteration='Tracking',0,0
                origin=torch.zeros(4,3,requires_grad=True)
                direction=torch.tensor([[.1,-.1,1.]]).repeat(4,1).requires_grad_()
                depth=torch.tensor([[0.],[1.5],[2.],[2.1]])
                color=torch.full((4,3),.5)
                torch.manual_seed(67)
                dense=m.forward(origin,direction,color,depth)
                m.config['sample_pruning']=dict(enabled=True,policy='tail',margin=.05)
                torch.manual_seed(67)
                sparse=forward_pruned(m,origin,direction,color,depth)
                keys=('rgb_loss','depth_loss','sdf_loss','fs_loss')
                for key in (*keys,'rgb','depth'):
                    torch.testing.assert_close(dense[key],sparse[key],rtol=3e-5,atol=2e-6)
                ga=torch.autograd.grad(sum(sparse[k] for k in keys),(origin,direction,m.coefficients))
                gb=torch.autograd.grad(sum(dense[k] for k in keys),(origin,direction,m.coefficients))
                for a,b in zip(ga,gb):
                    torch.testing.assert_close(a,b,rtol=3e-5,atol=2e-5)

    def test_thinning_preserves_rng_surface_support_and_missing_depths(self):
        z = torch.linspace(0, 5, 101).repeat(3, 1)
        depth = torch.tensor([[2.], [0.], [3.]])
        before = torch.get_rng_state().clone()
        keep, prob = lattice_selection(z, depth, .1, .15, 4, 7)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertTrue(keep[1].all())
        self.assertTrue(keep[(z-depth).abs() <= .1].all())
        self.assertTrue((prob[:, :2] == 1).all())
        self.assertTrue((~keep[0]).any())

    def test_no_crossing_falls_back_without_duplicate_queries(self):
        m = self.model('adaptive', constant=True)
        out = forward_pruned(m, torch.zeros(2,3), torch.tensor([[0.,0.,1.]]).repeat(2,1),
                             torch.ones(2,3), torch.full((2,1),2.))
        self.assertTrue(torch.isfinite(out['rgb_loss']))
        row = m.pruning_diagnostics[-1]
        self.assertEqual(row['queried_samples'], row['dense_samples'])

    def test_feature_grid_indices_unique_and_in_bounds(self):
        image = torch.rand(1,64,80,3)
        index = feature_uniform_pixels(dict(rgb=image), 4, 4, 128)
        self.assertEqual(len(index.unique()), 128)
        self.assertGreaterEqual(int(index.min()), 0)
        self.assertLess(int(index.max()), 56*72)

    def test_surface_refinement_preserves_mass_and_first_moment(self):
        z=torch.linspace(0,4,101).repeat(3,1)
        depth=torch.tensor([[2.],[0.],[.01]])
        refined,mass=refine_surface_lattice(z,depth,.25)
        torch.testing.assert_close(mass.sum(1),torch.full((3,),101.))
        torch.testing.assert_close((mass*refined).sum(1),z.sum(1))
        self.assertTrue((mass>=0).all())
        self.assertEqual(int((mass[1]>0).sum()),101)
        front=(z<depth-.25).float()
        new_front=(refined<depth-.25).float()
        torch.testing.assert_close(front.sum(1),(new_front*mass).sum(1))

    def test_refinement_reduces_net_queries_and_preserves_planar_geometry(self):
        m=self.model('refine')
        depth=torch.full((2,1),2.25)
        m.config['sample_pruning']['enabled']=False
        dense=m.forward(torch.zeros(2,3),torch.tensor([[0.,0.,1.]]).repeat(2,1),torch.full((2,3),.5),depth)
        m.config['sample_pruning']['enabled']=True
        result=forward_pruned(m,torch.zeros(2,3),torch.tensor([[0.,0.,1.]]).repeat(2,1),
                             torch.full((2,3),.5),depth)
        self.assertTrue(all(torch.isfinite(result[k]).all() for k in ('rgb_loss','depth_loss','sdf_loss','fs_loss')))
        for key in ('sdf_loss','fs_loss'):
            torch.testing.assert_close(result[key],dense[key])
        counts=m.pruning_diagnostics[-1]
        self.assertLess(counts['queried_samples'],counts['dense_samples'])

    def test_pruned_refinement_matches_its_dense_quadrature_objective(self):
        from test_ba_guarded_fields import CurvedField
        for mode in ('oscillating','late_surface','no_crossing','boundary_surface'):
            m=CurvedField(mode)
            m.config['training']['perturb']=1
            m.config['sample_pruning']=dict(enabled=True,policy='refine_dense',margin=0.)
            m._pruning_stage,m._pruning_frame,m._pruning_iteration='Tracking',0,0
            origin=torch.zeros(4,3,requires_grad=True)
            direction=torch.tensor([[.1,-.1,1.]]).repeat(4,1).requires_grad_()
            depth=torch.tensor([[0.],[1.5],[2.],[2.1]])
            color=torch.full((4,3),.5)
            torch.manual_seed(71)
            dense=forward_pruned(m,origin,direction,color,depth)
            m.config['sample_pruning']['policy']='refine'
            torch.manual_seed(71)
            sparse=forward_pruned(m,origin,direction,color,depth)
            keys=('rgb_loss','depth_loss','sdf_loss','fs_loss')
            for key in (*keys,'rgb','depth'):
                torch.testing.assert_close(dense[key],sparse[key],rtol=3e-5,atol=2e-6)
            ga=torch.autograd.grad(sum(sparse[k] for k in keys),(origin,direction,m.coefficients))
            gb=torch.autograd.grad(sum(dense[k] for k in keys),(origin,direction,m.coefficients))
            for a,b in zip(ga,gb):
                torch.testing.assert_close(a,b,rtol=3e-5,atol=2e-5)

    def test_rejects_conflicting_sampling_and_invalid_settings(self):
        m = self.model()
        m.config['ba_bucket']['enabled'] = False
        m.config['tracking_depth_spatial']['enabled'] = False
        validate_pruning(m.config)
        m.config['sample_pruning']['free_stride'] = 0
        with self.assertRaises(ValueError):
            validate_pruning(m.config)


if __name__ == '__main__':
    unittest.main()
