"""Shared CPU analytic-field and actual tracking-loop fixtures.

These helpers exercise the native renderer and AST-extracted production loops;
they are not standalone experiments or test cases.
"""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
    matrix_to_quaternion, quaternion_to_matrix)

from test_ba_guarded_fields import CurvedField
from optimization.tracking_experiments import tracking_execution_flags

LOSSES=('rgb_loss','depth_loss','fs_loss','sdf_loss')


class RayFieldFixture:
    def model(self, mode='oscillating', stage='Tracking'):
        m=CurvedField(mode)
        m.config['training'].update(perturb=1,rgb_missing=.2)
        m.config['sample_pruning']=dict(enabled=True,policy='ray_cohort',seed=17)
        m._pruning_stage=stage;m._pruning_frame=23;m._pruning_iteration=0
        return m

    def inputs(self, depths=(0.,1.2,2.3,6.)):
        n=len(depths)
        origin=(torch.arange(n*3).reshape(n,3).float()*.013).requires_grad_()
        direction=torch.tensor([.11,-.07,1.]).repeat(n,1).requires_grad_()
        rgb=torch.linspace(.15,.85,n*3).reshape(n,3)
        depth=torch.tensor(depths)[:,None]
        return origin,direction,rgb,depth

    def dense(self,m,args,seed=197):
        enabled=m.config['sample_pruning']['enabled']
        m.config['sample_pruning']['enabled']=False
        torch.manual_seed(seed)
        out=m.forward(*args)
        m.config['sample_pruning']['enabled']=enabled
        return out


def make_pose(rotation, translation):
    top = torch.cat((rotation, translation[..., None]), dim=-1)
    bottom = torch.zeros_like(top[..., :1, :])
    bottom[..., 0, 3] = 1.
    return torch.cat((top, bottom), dim=-2)


class Timer:
    def __init__(self):
        self.stages = []
    def stage(self, name):
        self.stages.append(name)
        return nullcontext()
    def set_context(self, *args): pass
    def flush(self): pass


class ScriptedOptimizer:
    """Deterministic updates isolate actual tracking handoff from convergence."""
    def __init__(self, slam, rotation, translation):
        self.slam = slam
        self.rotation = rotation
        self.translation = translation
        self.steps = 0
    def zero_grad(self):
        self.rotation.grad = None
        self.translation.grad = None
    def step(self):
        if self.rotation.grad is None or self.translation.grad is None:
            raise AssertionError('The actual tracking backward pass was skipped')
        self.steps += 1
        target = self.slam.states[self.steps]
        with torch.no_grad():
            self.rotation.copy_(self.slam.to_rotation(target[:3, :3])[None])
            self.translation.copy_(target[:3, 3][None])


class TrackingField:
    def __init__(self, slam):
        self.slam = slam
        self.evaluated = []
        self.ray_counts = []
    def set_timing_context(self, *args): pass
    def flush_scene_timing(self): pass
    def forward(self, origins, directions, rgb, depth, **kwargs):
        self.evaluated.append(self.slam.matrix_from_tensor(
            self.slam.rotation, self.slam.translation).detach().clone()[0])
        self.ray_counts.append(len(depth))
        # Keep genuine graph dependencies for the actual backward/step path,
        # while scripting scalar losses to test native best-pose selection.
        loss = (origins.sum() + directions.sum()) * 0 + self.slam.losses[len(self.evaluated)-1]
        return dict(loss=loss)


class TrackingSlam:
    def __init__(self, representation='axis_angle', best=True):
        self.device = torch.device('cpu')
        self.dataset = SimpleNamespace(H=4, W=4)
        self.config = dict(tracking=dict(iter=6, iter_point=0, const_speed=True,
            ignore_edge_H=1, ignore_edge_W=1, sample=4, best=best, wait_iters=100),
            mapping=dict(keyframe_every=5), training=dict(rot_rep=representation),
            grid=dict(morton_sort=False), sample_pruning=dict(enabled=False, policy='dense'))
        angles = torch.arange(7, dtype=torch.float32) * .03 + .2
        vectors = torch.stack((angles * .2, angles * -.1, angles), dim=-1)
        translations = torch.arange(21, dtype=torch.float32).reshape(7, 3) * .01
        self.states = make_pose(axis_angle_to_matrix(vectors), translations)
        self.losses = [3., 2., 1., 5., 4., 6.]
        key = torch.eye(4)
        key[:3, 3] = torch.tensor([.1, -.2, .3])
        self.est_c2w_data = {5: key}
        self.est_c2w_data_rel = {}
        self.coslam_timing = Timer()
        self.model = TrackingField(self)
    def to_rotation(self, matrix):
        return (matrix_to_axis_angle(matrix) if self.config['training']['rot_rep'] == 'axis_angle'
                else matrix_to_quaternion(matrix))
    def matrix_from_tensor(self, rotation, translation):
        matrix = (axis_angle_to_matrix(rotation) if self.config['training']['rot_rep'] == 'axis_angle'
                  else quaternion_to_matrix(rotation))
        return make_pose(matrix, translation)
    def get_pose_param_optim(self, poses, mapping=False):
        self.rotation = torch.nn.Parameter(self.to_rotation(poses[:, :3, :3]))
        self.translation = torch.nn.Parameter(poses[:, :3, 3].clone())
        self.optimizer = ScriptedOptimizer(self, self.rotation, self.translation)
        return self.rotation, self.translation, self.optimizer
    def predict_current_pose(self, *args): return self.states[0]
    def tracking_naive_pruning_enabled(self): return False
    def select_samples(self, height, width, count): return torch.arange(count)
    def get_loss_from_ret(self, ret): return ret['loss']
    def start_profiling(self, **kwargs): pass
    def step_profiler(self): pass


def actual_tracking_method(source=None):
    source = Path(source) if source is not None else Path(__file__).resolve().parents[1] / 'coslam.py'
    cls = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == 'CoSLAM')
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'tracking_render')
    namespace = dict(torch=torch, F=F, nvtx_range=lambda *a: nullcontext(),
                     record_function=lambda *a: nullcontext(),
                     tracking_execution_flags=tracking_execution_flags)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['tracking_render']


TrackingSlam.tracking_render = actual_tracking_method()


def tracking_batch():
    return dict(c2w=torch.eye(4)[None], direction=torch.tensor([.1, -.2, 1.]).repeat(1, 4, 4, 1),
                rgb=torch.full((1, 4, 4, 3), .5), depth=torch.ones(1, 4, 4))
