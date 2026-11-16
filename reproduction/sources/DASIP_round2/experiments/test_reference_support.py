"""CPU checks against the pinned public renderer/loss, not the modified baseline."""
import ast
import copy
import importlib.util
from pathlib import Path
import types
import unittest
import torch
import yaml
from round2_sampling import depth_lattice, supported_outputs, supported_render

RUNTIME=Path(__file__).parent/'coslam_reference_runtime_v1'


def parent_config(name):
    return yaml.safe_load((RUNTIME/'configs'/name).read_text())


class Field:
    def __init__(self,cfg):
        self.config=cfg;self._r2_stage='bundle_adjustment';self._r2_support=None
        self.weights=torch.tensor([.6,.2,.4,.8],requires_grad=True)
        source=ast.parse((RUNTIME/'model/scene_rep.py').read_text())
        cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='JointEncoding')
        ns={'torch':torch}
        nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('sdf2weights','raw2outputs','render_rays')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<public-renderer>','exec'),ns)
        for name in ('sdf2weights','raw2outputs','render_rays'):setattr(self,name,types.MethodType(ns[name],self))
    def run_network(self,points):
        return torch.cat([points*self.weights[:3],(2.-points[...,2:3])*self.weights[3]],-1)


class ReferenceSupportTests(unittest.TestCase):
    configs=('Replica/replica.yaml','Synthetic/synthetic.yaml','Tum/tum.yaml','ScanNet/scannet.yaml')
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('public_utils',RUNTIME/'model/utils.py')
        cls.native=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.native)
        torch.set_num_threads(1)
    def inputs(self):
        origin=torch.zeros((4,3),requires_grad=True)
        direction=torch.tensor([[.1,.05,1.]]*4,requires_grad=True)
        depth=torch.tensor([[.8],[1.2],[0.],[2.]])
        return origin,direction,depth
    def test_original_lattice_and_rng(self):
        for name in self.configs:
            with self.subTest(dataset=name):
                field=Field(parent_config(name));o,d,target=self.inputs()
                torch.manual_seed(41);native=field.render_rays(o,d,target);state=torch.get_rng_state()
                torch.manual_seed(41);z,lower=depth_lattice(field,o,target)
                torch.testing.assert_close(z,native['z_vals'],rtol=0,atol=0)
                self.assertTrue(torch.equal(state,torch.get_rng_state()))
                self.assertTrue(bool((z>=lower).all()))
    def test_full_support_renderer_and_gradients(self):
        for name in self.configs:
            with self.subTest(dataset=name):
                field=Field(parent_config(name));o,d,target=self.inputs()
                torch.manual_seed(8);reference=field.render_rays(o,d,target)
                raw=reference['raw'];z=reference['z_vals']
                restored=supported_outputs(field,raw,z,torch.ones_like(z,dtype=torch.bool))
                for key in ('rgb','depth','acc_map','depth_var'):
                    torch.testing.assert_close(restored[key],reference[key],rtol=0,atol=0)
                a=torch.autograd.grad(reference['rgb'].sum()+reference['depth'].sum(),(o,d,field.weights),retain_graph=True)
                b=torch.autograd.grad(restored['rgb'].sum()+restored['depth'].sum(),(o,d,field.weights))
                for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
    def test_pruning_preserves_public_geometry_losses_and_gradients(self):
        for name in self.configs:
            with self.subTest(dataset=name):
                cfg=parent_config(name);field=Field(cfg);o,d,target=self.inputs()
                tau=cfg['training']['trunc']*cfg['data']['sc_factor']
                torch.manual_seed(19);reference=field.render_rays(o,d,target)
                native=self.native.get_sdf_loss(reference['z_vals'],target,reference['raw'][...,-1],tau,'l2')
                a=torch.autograd.grad(sum(native),(o,d,field.weights))
                torch.manual_seed(19);selected=supported_render(field,o,d,target,.15)
                sparse=self.native.get_sdf_loss(selected['z_vals'],target,selected['raw'][...,-1],tau,'l2')
                b=torch.autograd.grad(sum(sparse),(o,d,field.weights))
                for x,y in zip(native,sparse):torch.testing.assert_close(x,y,rtol=0,atol=0)
                for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-7)
    def test_missing_depth_retains_the_complete_original_ray(self):
        for name in self.configs:
            with self.subTest(dataset=name):
                field=Field(parent_config(name));o,d,target=self.inputs()
                torch.manual_seed(2);reference=field.render_rays(o,d,target)
                torch.manual_seed(2);selected=supported_render(field,o,d,target,.15)
                torch.testing.assert_close(reference['raw'][2],selected['raw'][2],rtol=0,atol=0)
                self.assertTrue(bool((selected['raw'][0]==0).all(-1).any()))


if __name__=='__main__':
    result=unittest.main(exit=False,verbosity=2)
    assert not torch.cuda.is_initialized()
    raise SystemExit(not result.result.wasSuccessful())
