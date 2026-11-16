"""Exact CPU active-path comparisons against the original frozen V8 sources.

The original GPU evidence belongs to its archived sources. These checks are
supplemental cleanup evidence, not a new CUDA quality or timing experiment.
Set COSLAM_PRUNING_REFERENCE to an intact V8 source tree to replay the checks.
"""
import ast
import argparse
import copy
import importlib.util
import os
import pickle
import random
import sys
import unittest
from contextlib import contextmanager, nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import config
import optimization.sample_pruning as clean_pruning
import optimization.ray_selection as clean_rays
import optimization.cohort_selection as clean_cohorts
from optimization.ba_experiments import ba_bucket_mode, configure_ba_embed_eps
from optimization.utils import (at_to_transform_matrix, qt_to_transform_matrix,
    matrix_to_axis_angle, matrix_to_quaternion)
import pruning_cpu_fixtures as fixtures
import test_cohort_selection as ba_fixtures

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(os.environ.get('COSLAM_PRUNING_REFERENCE',
                               '/tmp/coslam_pruning8_v8/Co-SLAM'))
SCENES = ('Replica/office0.yaml', 'Synthetic/tg.yaml',
          'Tum/fr1_desk.yaml', 'ScanNet/scene0000.yaml')
RETAINED = dict(enabled=True, policy='ray_cohort', cohort_count=4,
    recent_keyframes=2, cohort_transition_keyframes=2, cohort_weighting='population',
    cohort_tracking='ray_cycle', tracking_cycle_groups=2, tracking_anchor_every=3,
    tracking_pose_selection='dense_anchors', tv_fraction=.125, seed=1)


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ast_method(name, root):
    source = root / 'coslam.py'
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'CoSLAM')
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(torch=torch, np=np, random=random,
        record_function=lambda *a: nullcontext(), nvtx_range=lambda *a: nullcontext(),
        ba_bucket_mode=ba_bucket_mode, configure_ba_embed_eps=configure_ba_embed_eps)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]



def benchmark_config(root, argv):
    source=root/'scripts/Validation/benchmark_ba_accuracy.py'
    tree=ast.parse(source.read_text())
    body=next(node.body for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    end=next(i for i,node in enumerate(body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='args' for t in node.targets))
    start=next(i for i,node in enumerate(body) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cfg' for t in node.targets))
    # The first mkdir is before cfg load. The second one follows cfg.data update.
    stop=next(i for i,node in enumerate(body[start:],start) if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='mkdir')
    variant=next(node for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VARIANTS' for t in node.targets))
    namespace=dict(argparse=argparse,Path=Path,config=config,__doc__=ast.get_docstring(tree))
    with patch('sys.argv',['benchmark_ba_accuracy.py',*argv]):
        exec(compile(ast.Module(body=[variant,*body[:end+1],*body[start:stop]],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['cfg']

class PrimitiveTrace(TorchDispatchMode):
    """Record executed tensor primitive order without examining tensor values."""
    def __init__(self):
        self.operations = []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(str(func))
        return func(*args, **(kwargs or {}))


def rng_state():
    return random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone()


def reset_rng():
    random.seed(197)
    np.random.seed(197)
    torch.manual_seed(197)


def tensor_copy(value):
    return value.detach().clone()


def native_pose_methods(instance, representation):
    instance.matrix_to_tensor = matrix_to_axis_angle if representation == 'axis_angle' else matrix_to_quaternion
    instance.matrix_from_tensor = at_to_transform_matrix if representation == 'axis_angle' else qt_to_transform_matrix


class RecordedField(fixtures.CurvedField):
    def __init__(self, slam):
        super().__init__('oscillating')
        self.slam = slam
        self.config = slam.config
        self.records = []
        self.evaluated = []
        self.feature_points = []
    def set_timing_context(self, frame, stage):
        self._pruning_frame, self._pruning_stage, self._pruning_iteration = frame, stage, 0
    def flush_scene_timing(self): pass
    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        self.records.append({k: tensor_copy(result[k]) for k in (*fixtures.LOSSES, 'rgb', 'depth')})
        if hasattr(self.slam, 'rotation'):
            self.evaluated.append(tensor_copy(self.slam.matrix_from_tensor(
                self.slam.rotation, self.slam.translation)))
        return result
    def query_sdf(self, points, embed=False):
        assert embed
        self.feature_points.append(tensor_copy(points))
        return torch.sin(points * self.coefficients[:3])


class ActualTracking(fixtures.TrackingSlam):
    def __init__(self, representation, best, scene, root):
        super().__init__(representation, best)
        tracking = self.config['tracking']
        self.config = config.load_config(str(ROOT / 'configs' / scene))
        self.config['tracking'].update(tracking, iter=10, lr_rot=.0005, lr_trans=.001)
        self.config['mapping']['keyframe_every'] = 5
        self.config['sample_pruning'] = copy.deepcopy(RETAINED)
        self.config['grid']['morton_sort'] = False
        self.config['training'].update(perturb=1, rgb_missing=.2)
        for name in ('tracking_bucket', 'tracking_naive_pruning', 'tracking_depth_spatial'):
            self.config.setdefault(name, {})['enabled'] = False
        self.model = RecordedField(self)
        native_pose_methods(self, representation)
        self.native_pose_optimizer = MethodType(ast_method('get_pose_param_optim', root), self)
        self.tracking_render = MethodType(fixtures.actual_tracking_method(root/'coslam.py'), self)
        self.gradient_records = []
    def get_pose_param_optim(self, poses, mapping=False):
        rotation, translation, optimizer = self.native_pose_optimizer(poses, mapping)
        self.rotation, self.translation, self.optimizer = rotation, translation, optimizer
        original_step = optimizer.step
        def record_step(*args, **kwargs):
            self.gradient_records.append((tensor_copy(rotation.grad), tensor_copy(translation.grad)))
            return original_step(*args, **kwargs)
        optimizer.step = record_step
        return rotation, translation, optimizer
    def get_loss_from_ret(self, ret):
        return sum(ret[key]*weight for key, weight in zip(fixtures.LOSSES, (5., .1, 10., 1000.)))


class ActualBA(ba_fixtures.CPUSlam):
    def __init__(self, representation, root, keep_empty=False, weighting='population'):
        super().__init__(n=10, count=4)
        mapping = self.config['mapping']
        self.config = config.load_config(str(ROOT/'configs/Replica/office0.yaml'))
        self.config['mapping'].update(mapping)
        self.config['sample_pruning'] = copy.deepcopy(RETAINED)
        self.config['sample_pruning'].update(cohort_keep_empty_pose_blocks=keep_empty,
                                             cohort_weighting=weighting)
        self.config['training'].update(perturb=1, rot_rep=representation, rgb_missing=.2)
        self.config['grid'].update(morton_sort=False)
        self.config['ba_bucket']['enabled'] = False
        self.config['ba_experiment'] = dict(variant='baseline')
        native_pose_methods(self, representation)
        # A nonzero initial angle avoids the existing native axis-angle
        # identity singularity; this test does not alter that implementation.
        initial = torch.tensor([[.1, -.05, .2]])
        rotation = at_to_transform_matrix(initial, torch.zeros(1, 3))[0, :3, :3]
        for pose in self.est_c2w_data.values():
            pose[:3, :3] = rotation
        self.model = RecordedField(self)
        self.map_optimizer = torch.optim.Adam([self.model.coefficients], lr=.0001)
        self.bounding_box = torch.tensor([[0., 4.], [0., 4.], [0., 4.]])
        self.native_pose_optimizer = MethodType(ast_method('get_pose_param_optim', root), self)
        for name in ('global_BA', 'global_BA_cohort', 'smoothness'):
            setattr(self, name, MethodType(ast_method(name, root), self))
        self.gradient_records = []
    def get_pose_param_optim(self, poses, mapping=True):
        self.pose_parameter_counts.append(len(poses))
        rotation, translation, optimizer = self.native_pose_optimizer(poses, mapping)
        self.rotation, self.translation, self.optimizer = rotation, translation, optimizer
        original_step = optimizer.step
        def record_step(*args, **kwargs):
            self.gradient_records.append((tensor_copy(rotation.grad), tensor_copy(translation.grad)))
            return original_step(*args, **kwargs)
        optimizer.step = record_step
        return rotation, translation, optimizer
    def get_loss_from_ret(self, ret, smooth=False):
        loss = sum(ret[key]*weight for key, weight in zip(fixtures.LOSSES, (5., .1, 10., 1000.)))
        return loss + .001*self.smoothness(sample_points=8, voxel_size=.1) if smooth else loss


@unittest.skipUnless((REFERENCE/'optimization/sample_pruning.py').exists(),
                     'Frozen V8 source is required for supplemental cleanup comparisons')
class CleanPruningEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen_pruning = load_file('_frozen_v8_pruning', REFERENCE/'optimization/sample_pruning.py')
        cls.frozen_rays = load_file('_frozen_v8_rays', REFERENCE/'optimization/ray_selection.py')
        cls.frozen_cohorts = load_file('_frozen_v8_cohorts', REFERENCE/'optimization/cohort_selection.py')
    def setUp(self):
        nvtx = patch('torch.cuda.nvtx.range', lambda *a, **kw: nullcontext())
        nvtx.start(); self.addCleanup(nvtx.stop)
    @contextmanager
    def source_modules(self, frozen):
        modules = {'optimization.sample_pruning': self.frozen_pruning if frozen else clean_pruning,
                   'optimization.ray_selection': self.frozen_rays if frozen else clean_rays,
                   'optimization.cohort_selection': self.frozen_cohorts if frozen else clean_cohorts}
        previous = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            yield
        finally:
            for name, module in previous.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
    def equal_tree(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0, equal_nan=False)
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(set(first), set(second))
            for key in first: self.equal_tree(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second): self.equal_tree(a, b)
        else:
            self.assertEqual(first, second)
    def model_and_pose_inputs(self, scene, representation, stage):
        helper = fixtures.RayFieldFixture()
        model = helper.model(stage=stage)
        model.config = config.load_config(str(ROOT/'configs'/scene))
        model.config['sample_pruning'] = copy.deepcopy(RETAINED)
        model.config['training'].update(perturb=1, rgb_missing=.2)
        model.config['tracking']['iter'] = 10
        inputs = helper.inputs((0., 1.2, 2.3, 6., 1.1, 2.5, 0., 3.1, 1.4, 2.9, 3.2, .8))
        axis = torch.tensor([[.1, -.2, .05]])
        rotation = (axis if representation=='axis_angle' else
                    matrix_to_quaternion(at_to_transform_matrix(axis, torch.zeros(1,3))[:,:3,:3])).requires_grad_()
        translation = torch.tensor([[.05, -.03, .1]], requires_grad=True)
        transform = at_to_transform_matrix if representation=='axis_angle' else qt_to_transform_matrix
        pose = transform(rotation, translation)
        origin = inputs[0]+pose[0,:3,3]
        direction = torch.sum(inputs[1][:,None,:]*pose[0,:3,:3], dim=-1)
        arguments = origin, direction, inputs[2], inputs[3]
        parameters = (model.coefficients, inputs[0], inputs[1], rotation, translation)
        model._ray_sampling_metadata = torch.cat((inputs[1].detach(), inputs[2], inputs[3]), dim=-1)
        if stage=='Bundle_Adjustment':
            model._ray_group_ids = torch.arange(-1, 11)
        return model, arguments, parameters
    def evaluate_forward(self, frozen, scene, representation, stage):
        with self.source_modules(frozen):
            model, args, parameters = self.model_and_pose_inputs(scene, representation, stage)
            reset_rng(); rows=[]; trace=PrimitiveTrace()
            with trace:
                for iteration in range(10 if stage=='Tracking' else 4):
                    if stage=='Bundle_Adjustment':
                        active, probability, _ = clean_cohorts.choose_pose_cohort(
                            11, 55, 5, 4, 2, iteration, transition_keyframes=2)
                        model._cohort_active_ids, model._cohort_probabilities = active, probability
                    result = model.forward(*args)
                    gradients = tuple(tuple(tensor_copy(g) for g in torch.autograd.grad(
                        result[key], parameters, retain_graph=True)) for key in fixtures.LOSSES)
                    rows.append(({key:tensor_copy(result[key]) for key in (*fixtures.LOSSES,'rgb','depth')},
                                 gradients, model._pruning_iteration))
            return rows, model.queried_points, getattr(model,'pruning_diagnostics',[]), rng_state(), trace.operations
    def test_all_four_native_lattices_exact_losses_map_pose_ray_gradients_rng_and_primitives(self):
        for scene in SCENES:
            for representation in ('axis_angle','quat'):
                for stage in ('Tracking','Bundle_Adjustment'):
                    with self.subTest(scene=scene, representation=representation, stage=stage):
                        frozen=self.evaluate_forward(True,scene,representation,stage)
                        clean=self.evaluate_forward(False,scene,representation,stage)
                        self.equal_tree(frozen,clean)
                        if stage=='Tracking':
                            count=clean[1][0].shape[0]//12
                            self.assertEqual(sum(len(points) for points in clean[1]),12*count*7)
    def tracking_result(self, frozen, representation, best, scene):
        with self.source_modules(frozen), redirect_stdout(StringIO()):
            slam=ActualTracking(representation,best,scene,REFERENCE if frozen else ROOT)
            reset_rng();trace=PrimitiveTrace()
            with trace:
                slam.tracking_render(fixtures.tracking_batch(),7)
            self.assertEqual(len(slam.gradient_records),10)
            self.assertEqual(len(slam.model.evaluated),10)
            return (slam.est_c2w_data,slam.est_c2w_data_rel,slam.model.records,
                slam.model.evaluated,slam.model.queried_points,slam.gradient_records,
                slam.model.coefficients.grad,slam.optimizer.state_dict(),
                slam.model.pruning_diagnostics,slam.coslam_timing.stages,rng_state(),trace.operations)
    def test_actual_tracking_native_pose_optimizer_handoff_is_bitwise_for_both_representations(self):
        for representation in ('axis_angle','quat'):
            for best in (False,True):
                for scene in (SCENES[0],SCENES[-1]):
                    with self.subTest(representation=representation,best=best,scene=scene):
                        self.equal_tree(self.tracking_result(True,representation,best,scene),
                                        self.tracking_result(False,representation,best,scene))
    def ba_result(self, frozen, representation, keep_empty, weighting):
        with self.source_modules(frozen), redirect_stdout(StringIO()):
            slam=ActualBA(representation,REFERENCE if frozen else ROOT,keep_empty,weighting)
            reset_rng();trace=PrimitiveTrace()
            with trace:
                slam.global_BA(ba_fixtures.CohortBATests().batch(),50)
            return (slam.est_c2w_data,slam.model.records,slam.model.queried_points,
                slam.model.feature_points,slam.gradient_records,slam.map_optimizer.state_dict(),
                slam.optimizer.state_dict(),slam.model.pruning_diagnostics,
                slam.model.smoothness_diagnostics,slam._cohort_pose_diagnostics,
                slam.sample_calls,slam.pose_parameter_counts,rng_state(),trace.operations)
    def test_actual_ba_cohort_pose_updates_map_updates_and_tv_match_with_attribution_controls(self):
        for representation in ('axis_angle','quat'):
            for keep_empty in (False,True):
                for weighting in ('population','native'):
                    with self.subTest(representation=representation,keep_empty=keep_empty,weighting=weighting):
                        self.equal_tree(self.ba_result(True,representation,keep_empty,weighting),
                                        self.ba_result(False,representation,keep_empty,weighting))
    def test_retained_functions_are_unchanged_ast(self):
        specifications={
            'optimization/sample_pruning.py': ('pruning_policy','lattice_selection',
                'refine_surface_lattice','weighted_geometry','render_weights','feature_uniform_pixels',
                'forward_supported','sampled_tv_loss','forward_cohort_ba'),
            'optimization/ray_selection.py': ('_image_depth_order','cyclic_ray_subsets'),
            'optimization/cohort_selection.py': ('choose_pose_cohort',),
            'coslam.py': ('get_pose_param_optim','global_BA_cohort')}
        for path,names in specifications.items():
            def selected(root):
                return {node.name:ast.dump(node,include_attributes=False) for node in ast.walk(
                    ast.parse((root/path).read_text())) if isinstance(node,ast.FunctionDef) and node.name in names}
            self.assertEqual(selected(ROOT),selected(REFERENCE),path)
    def test_preexisting_point_policies_are_numerically_unchanged(self):
        for policy in ('dense','tail','ht','adaptive','refine','refine_dense','support','support_tv'):
            for stage in ('Tracking','Bundle_Adjustment'):
                results=[]
                for frozen in (True,False):
                    with self.source_modules(frozen):
                        model,args,parameters=self.model_and_pose_inputs(SCENES[0],'axis_angle',stage)
                        model.config['sample_pruning'].update(policy=policy,margin=.15,free_stride=2)
                        model._pruning_iteration=1
                        reset_rng();trace=PrimitiveTrace()
                        with trace:
                            result=model.forward(*args)
                            gradients=torch.autograd.grad(sum(result[k] for k in fixtures.LOSSES),parameters)
                        results.append(({key:tensor_copy(result[key]) for key in (*fixtures.LOSSES,'rgb','depth')},
                            gradients,model.queried_points,getattr(model,'pruning_diagnostics',[]),
                            rng_state(),trace.operations))
                with self.subTest(policy=policy,stage=stage):self.equal_tree(*results)

    def test_clean_cli_changes_only_removed_inert_configuration_fields(self):
        active=['--cohort-count','4','--recent-keyframes','2',
            '--cohort-transition-keyframes','2','--cohort-weighting','population',
            '--cohort-tracking','ray_cycle','--tracking-cycle-count','2',
            '--tracking-anchor-every','3','--tracking-pose-selection','dense_anchors',
            '--tv-fraction','.125']
        removed={'tracking_ray_fraction','ba_ray_fraction','ray_strategy',
                 'ray_estimator','tracking_pose_average'}
        for scene in SCENES:
            dataset,name=scene.split('/')
            for policy in ('dense','ray_cohort'):
                args=['--dataset',dataset,'--scene',name[:-5],'--tracking','baseline',
                      '--variant','baseline','--frames','0','--warmup','20','--seed','1',
                      '--output','/tmp/not-created-cli-fixture','--pruning',policy]
                if policy=='ray_cohort':args+=active
                else:args+=['--tracking-pose-selection','native']
                frozen=benchmark_config(REFERENCE,args)
                clean=benchmark_config(ROOT,args)
                original_sampling=frozen.pop('sample_pruning')
                clean_sampling=clean.pop('sample_pruning')
                self.assertEqual(frozen,clean)
                self.assertEqual({k:v for k,v in original_sampling.items() if k not in removed},
                                 clean_sampling)
                self.assertFalse(removed & clean_sampling.keys())
                self.assertEqual(clean['grid']['hash_size'],16 if dataset=='ScanNet' else 19)

    def test_removed_public_policies_and_controls_fail_explicitly(self):
        for policy in ('ray_subset','ray_interp'):
            cfg=dict(training={},sample_pruning=dict(enabled=True,policy=policy))
            with self.assertRaises(ValueError):clean_pruning.validate_pruning(cfg)
        for key,value in (('cohort_tracking','ray_subset'),('tracking_pose_average',4)):
            cfg=dict(training={},sample_pruning=copy.deepcopy(RETAINED))
            cfg['sample_pruning'][key]=value
            with self.assertRaises(ValueError):clean_pruning.validate_pruning(cfg)


if __name__=='__main__':
    unittest.main(verbosity=2)
