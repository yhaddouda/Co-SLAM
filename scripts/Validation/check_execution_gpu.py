"""Frozen CUDA loss/gradient checks and repeated pose-execution microchecks.

Run in coslam or ec_slam according to --system. Paired checks share a frozen
field. EC's additional iterative probe first fits that field for 32 updates.
These checks do not infer whole-trajectory equivalence; full runs measure ATE.
"""
import argparse
import ast
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.Validation.execution_benchmark_common import device_metadata, statistics


def source_functions(path, names):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def errors(reference, actual):
    numerator, denominator, maximum = 0., 0., 0.
    for expected, observed in zip(reference, actual):
        if expected is None:
            if observed is not None:
                raise AssertionError('Changed gradient connectivity')
            continue
        if observed is None:
            raise AssertionError('Lost gradient connectivity')
        if expected.shape != observed.shape:
            raise AssertionError('Changed gradient shape')
        # Parameter-free encodings may expose an empty parameter/gradient.
        if not expected.numel():
            continue
        numerator += float((expected.float() - observed.float()).square().sum())
        denominator += float(expected.float().square().sum())
        maximum = max(maximum, float((expected.float() - observed.float()).abs().max()))
    return dict(relative_l2=(numerator / max(denominator, 1e-30)) ** .5, max_absolute=maximum)


def timed_repeated(function, repeats=30):
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        values.append(1000 * (time.perf_counter() - started))
    return statistics(values)


def coslam_checks(args):
    import config
    from datasets.dataset import get_dataset
    from model.scene_rep import JointEncoding
    from optimization.utils import at_to_transform_matrix, matrix_to_axis_angle
    from optimization.tracking_experiments import DevicePoseTransform
    cfg = config.load_config('configs/Replica/office1.yaml')
    cfg['timing']['mode'] = 'none'
    cfg['grid']['morton_sort'] = False
    model = JointEncoding(cfg, torch.tensor(cfg['mapping']['bound'], dtype=torch.float32, device='cuda')).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', mmap=True)['model'])
    dataset = get_dataset(cfg)
    records = []
    for frame in (0, 300, 900):
        sample = dataset[frame]
        indices = torch.randperm(dataset.H * dataset.W, generator=torch.Generator().manual_seed(17))[:512]
        directions = sample['direction'].reshape(-1, 3)[indices].cuda()
        colors = sample['rgb'].reshape(-1, 3)[indices].cuda()
        depths = sample['depth'].reshape(-1, 1)[indices].cuda()
        pose = sample['c2w'].cuda()
        rotation = matrix_to_axis_angle(pose[:3, :3])[None].detach().requires_grad_(True)
        translation = (pose[:3, 3][None] + .02).detach().requires_grad_(True)
        transforms = (at_to_transform_matrix, DevicePoseTransform(rotation))
        comparisons = []
        for transform in transforms:
            torch.manual_seed(73)
            torch.cuda.manual_seed_all(73)
            matrix = transform(rotation, translation)
            rays_d = torch.sum(directions[..., None, :] * matrix[:, :3, :3], -1)
            rays_o = matrix[..., :3, -1].repeat(len(directions), 1)
            result = model.forward(rays_o, rays_d, colors, depths)
            loss = sum(cfg['training'][weight] * result[key] for weight, key in
                       (('rgb_weight','rgb_loss'),('depth_weight','depth_loss'),
                        ('sdf_weight','sdf_loss'),('fs_weight','fs_loss')))
            gradient = torch.autograd.grad(loss, tuple(model.parameters()) + (rotation, translation), allow_unused=True)
            comparisons.append((matrix.detach(), float(loss), gradient))
        baseline, fast = comparisons
        torch.testing.assert_close(baseline[0], fast[0], rtol=1e-6, atol=1e-7)
        error = errors(baseline[2], fast[2])
        assert error['relative_l2'] < .001, error
        assert all(value is not None for value in (*baseline[2][-2:], *fast[2][-2:])), 'Disconnected Co pose gradient'
        pose_errors = {name:errors((baseline[2][index],),(fast[2][index],))
                       for name,index in (('rotation',-2),('translation',-1))}
        assert all(value['relative_l2'] < .001 or value['max_absolute'] < 1e-7
                   for value in pose_errors.values()), pose_errors
        assert abs(baseline[1] - fast[1]) < 1e-5 * max(1., abs(baseline[1]))
        records.append(dict(frame=frame, baseline_loss=baseline[1], fast_loss=fast[1],
                            pose_max_difference=float((baseline[0]-fast[0]).abs().max()),
                            map_rotation_translation_gradient_error=error))
        records[-1]['pose_gradient_errors'] = pose_errors
    rotation = torch.randn(1, 3, device='cuda') * .3
    translation = torch.randn_like(rotation)
    def singleton():
        for _ in range(10):
            at_to_transform_matrix(rotation, translation)
    def device():
        transform = DevicePoseTransform(rotation)
        for _ in range(10):
            transform(rotation, translation)
    micro = dict(scope='10 pose reconstructions; constant setup included; no network or backward',
                 baseline=timed_repeated(singleton), device_constants=timed_repeated(device))
    return records, micro


def ec_checks(args):
    workspace = ROOT / 'experiments/ec_slam_execution'
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    from src.utils import load_config, get_camera_from_tensor, get_tensor_from_camera
    from src.execution_experiments import camera_from_tensor_device
    from src.Model import Model
    from src.Datasets import get_dataset
    from src.Scene import cameraInfo
    cfg = load_config('configs/Replica/office1.yaml', 'configs/EC_SLAM.yaml')
    cfg['timing']['mode'] = 'none'
    cfg['HashGrid'].update(hash='CoherentPrime', morton_sort=False, hash_size=19)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = Model(cfg).cuda()
    data = get_dataset(cfg)
    dirs = cameraInfo('cuda', cfg['cam']['H'], cfg['cam']['W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy'])
    sample = data[10]
    indices = torch.randperm(cfg['cam']['H'] * cfg['cam']['W'], generator=torch.Generator().manual_seed(17))[:512]
    colors = torch.as_tensor(sample['image'], dtype=torch.float32, device='cuda').reshape(-1, 3)[indices]
    depths = torch.as_tensor(sample['depth'], dtype=torch.float32, device='cuda').reshape(-1, 1)[indices]
    directions = dirs.rays_dir_cache.reshape(-1, 3)[indices]
    camera = get_tensor_from_camera(torch.tensor(sample['T'], dtype=torch.float32)).cuda().detach().requires_grad_(True)
    records = []
    for tracking in (True, False):
        comparisons = []
        for transform in (get_camera_from_tensor, camera_from_tensor_device):
            torch.manual_seed(73)
            torch.cuda.manual_seed_all(73)
            matrix = transform(camera)
            rays_d = (matrix[None, :, :3] @ directions[..., None]).squeeze(-1)
            rays_o = matrix[:, 3].repeat(len(directions), 1)
            loss = model.forward(rays_o, rays_d, colors, depths, tracker=tracking, smooth=False)
            gradient = torch.autograd.grad(loss, tuple(model.parameters()) + (camera,), allow_unused=True)
            comparisons.append((matrix.detach(), float(loss), gradient))
        baseline, fast = comparisons
        torch.testing.assert_close(baseline[0], fast[0], rtol=0, atol=0)
        error = errors(baseline[2], fast[2])
        assert error['relative_l2'] < .001, error
        pose_error = errors((baseline[2][-1],),(fast[2][-1],))
        assert baseline[2][-1] is not None and fast[2][-1] is not None, 'Disconnected EC camera gradient'
        assert pose_error['relative_l2'] < .001 or pose_error['max_absolute'] < 1e-7, pose_error
        assert abs(baseline[1] - fast[1]) < 1e-5 * max(1., abs(baseline[1]))
        records.append(dict(stage='tracking' if tracking else 'mapping', frozen_map='fresh initialized real EC model',
            baseline_loss=baseline[1], fast_loss=fast[1],
            pose_max_difference=float((baseline[0]-fast[0]).abs().max()),
            map_quaternion_translation_gradient_error=error))
        records[-1]['pose_gradient_error'] = pose_error
    # Validate device-side mapping reference replacement on actual quaternion
    # transforms, including quaternion gradients through ray transformation.
    from pytorch3d.transforms import quaternion_to_matrix
    reference = torch.eye(4, device='cuda').repeat(20, 1, 1)
    parameters = torch.randn(3, 7, device='cuda', requires_grad=True)
    indices = [3, 5, 12]
    gradients, values = [], []
    for fast in (False, True):
        matrix = torch.eye(4, device='cuda').repeat(3, 1, 1)
        matrix[:, :3, :3] = quaternion_to_matrix(parameters[:, :4])
        matrix[:, :3, 3] = parameters[:, 4:]
        if fast:
            current = reference.index_copy(0, torch.tensor(indices, device='cuda'), matrix)
        else:
            current = reference.clone()
            current[indices] = matrix
        values.append(current.detach())
        gradients.append(torch.autograd.grad(current.square().sum(), parameters))
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0][0], gradients[1][0], rtol=0, atol=0)
    records.append(dict(stage='mapping_pose_reference', values_and_pose_gradients='exact match'))
    from scripts.Validation.check_ec_tracking_invocation_gpu import iterative_tracking_checks
    records.extend(iterative_tracking_checks(cfg, model, colors, depths, directions, camera))
    camera = camera.detach()
    def singleton():
        for _ in range(8):
            get_camera_from_tensor(camera)
    def device():
        for _ in range(8):
            camera_from_tensor_device(camera)
    micro = dict(scope='8 quaternion pose reconstructions; no network or backward',
                 baseline=timed_repeated(singleton), device_allocation=timed_repeated(device))
    return records, micro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=('coslam', 'ec_slam'), required=True)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT / 'output/Replica/office1/run_1/baseline_Size19_Run1/checkpoint1999.pt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output, args.checkpoint = args.output.resolve(), args.checkpoint.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    records, micro = coslam_checks(args) if args.system == 'coslam' else ec_checks(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(system=args.system, checks=records, pose_microcheck=micro, **device_metadata())
    args.output.write_text(json.dumps(summary, indent=2))
    print('PASS ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
