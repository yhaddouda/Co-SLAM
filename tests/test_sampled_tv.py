import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch
from optimization.sample_pruning import sampled_tv_loss

class SampledTVTests(unittest.TestCase):
    def model(self):
        parameter=torch.tensor([.7,1.3],requires_grad=True)
        def query(p,embed=False):
            return torch.stack([parameter[0]*(p[...,0].square()+p[...,1].sin()),parameter[1]*(p[...,1]*p[...,2]+p[...,2].square())],-1)
        return SimpleNamespace(bounding_box=torch.tensor([[-2.,2.],[-2.,2.],[-2.,2.]]),
            config=dict(grid=dict(tcnn_encoding=True),sample_pruning=dict(seed=0)),
            model=SimpleNamespace(query_sdf=query,_pruning_frame=0,_pruning_iteration=1)),parameter

    def native(self,slam,n):
        root=Path(__file__).resolve().parents[1]
        util=ast.parse((root/'utils.py').read_text())
        coordinates=next(x for x in util.body if isinstance(x,ast.FunctionDef) and x.name=='coordinates')
        source=ast.parse((root/'coslam.py').read_text())
        cls=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=='CoSLAM')
        method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='smoothness')
        namespace={'torch':torch}
        exec(compile(ast.Module(body=[coordinates,method],type_ignores=[]),'native_smoothness','exec'),namespace)
        return namespace['smoothness'](slam,n,.1,.05)

    def test_initialization_stays_dense_and_full_fraction_restores_native(self):
        s,p=self.model()
        s.config['sample_pruning'].update(enabled=True,policy='support_tv',tv_fraction=.125)
        query=s.model.query_sdf;counts=[]
        def counted(points,**kw):
            counts.append(points.numel()//3)
            return query(points,**kw)
        s.model.query_sdf=counted
        s.model._pruning_stage='First_frame_mapping'
        torch.manual_seed(53);dense=self.native(s,32);state=torch.random.get_rng_state()
        self.assertEqual(counts[-1],31**3)
        s.model._pruning_stage='Bundle_Adjustment'
        torch.manual_seed(53);self.native(s,32)
        self.assertLess(counts[-1],31**3)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        s.config['sample_pruning']['tv_fraction']=1.
        torch.manual_seed(53);restored=self.native(s,32)
        self.assertEqual(counts[-1],31**3)
        torch.testing.assert_close(restored,dense)
        torch.testing.assert_close(torch.autograd.grad(restored,p)[0],torch.autograd.grad(dense,p)[0])

    def test_all_edges_match_original_loss_gradients_and_rng(self):
        s,p=self.model();n=5;population=(n-2)*(n-1)**2
        torch.manual_seed(53);expected=self.native(s,n);state=torch.random.get_rng_state()
        generator=SimpleNamespace(integers=lambda *a,**kw:np.tile(np.arange(population),(3,1)))
        with patch('optimization.sample_pruning.np.random.default_rng',return_value=generator):
            torch.manual_seed(53);actual=sampled_tv_loss(s,n,.1,.05,population)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual,p)[0],torch.autograd.grad(expected,p)[0])
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_sampled_edges_unbiased_over_all_edge_phases(self):
        s,p=self.model();n=4;population=(n-2)*(n-1)**2
        torch.manual_seed(53);expected=self.native(s,n)
        losses=[]
        for phase in range(population):
            generator=SimpleNamespace(integers=lambda *a,**kw:np.full((3,1),phase,dtype=np.int64))
            with patch('optimization.sample_pruning.np.random.default_rng',return_value=generator):
                torch.manual_seed(53);losses.append(sampled_tv_loss(s,n,.1,.05,1))
        actual=torch.stack(losses).mean()
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(torch.autograd.grad(actual,p)[0],torch.autograd.grad(expected,p)[0])

    def test_actual_feature_queries_are_pruned(self):
        s,p=self.model();sampled_tv_loss(s,32,.1,.05,128)
        r=s.model.smoothness_diagnostics[-1]
        self.assertEqual(r['queried_samples'],768)
        self.assertEqual(r['dense_samples'],31**3)
        self.assertLess(r['queried_samples'],r['dense_samples'])
