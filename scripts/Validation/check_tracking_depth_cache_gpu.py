"""Validate dense tracking-depth caching against a trained Co-SLAM field on CUDA."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import config
from datasets.dataset import get_dataset
from model.scene_rep import JointEncoding
from optimization.utils import at_to_transform_matrix, matrix_to_axis_angle
from optimization.tracking_experiments import DenseDepthLattice
from scripts.Validation.check_execution_gpu import errors
from scripts.Validation.execution_benchmark_common import device_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'output/Replica/office1/run_1/baseline_Size19_Run1/checkpoint1999.pt')
    args = parser.parse_args()
    cfg = config.load_config('configs/Replica/office1.yaml')
    cfg['timing']['mode'] = 'none'
    cfg['grid'].update(hash='CoherentPrime', hash_size=19, morton_sort=False)
    model = JointEncoding(cfg, torch.tensor(cfg['mapping']['bound'], dtype=torch.float32, device='cuda')).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', mmap=True)['model'])
    dataset = get_dataset(cfg)
    original_render = model.render_rays
    sampled = []
    def observed(*inputs, **kwargs):
        result = original_render(*inputs, **kwargs)
        sampled.append(result['z_vals'].detach().clone())
        return result
    model.render_rays = observed
    records = []
    for frame in (0, 300, 900):
        sample = dataset[frame]
        ids = torch.randperm(dataset.H * dataset.W, generator=torch.Generator().manual_seed(17))[:512]
        directions = sample['direction'].reshape(-1, 3)[ids].cuda()
        colors = sample['rgb'].reshape(-1, 3)[ids].cuda()
        depths = sample['depth'].reshape(-1, 1)[ids].cuda()
        pose = sample['c2w'].cuda()
        rotation = matrix_to_axis_angle(pose[:3, :3])[None].detach().requires_grad_(True)
        translation = (pose[:3, 3][None] + .02).detach().requires_grad_(True)
        comparisons, rng_states = [], []
        for cached in (False, True):
            cache = DenseDepthLattice(cfg) if cached else None
            torch.manual_seed(73)
            torch.cuda.manual_seed_all(73)
            iterations = []
            for iteration in range(3):
                matrix = at_to_transform_matrix(rotation, translation + .005 * iteration)
                rays_d = torch.sum(directions[..., None, :] * matrix[:, :3, :3], -1)
                rays_o = matrix[..., :3, -1].repeat(len(directions), 1)
                result = model.forward(rays_o, rays_d, colors, depths, depth_sampling_cache=cache)
                loss = sum(cfg['training'][weight] * result[key] for weight, key in
                           (('rgb_weight','rgb_loss'),('depth_weight','depth_loss'),
                            ('sdf_weight','sdf_loss'),('fs_weight','fs_loss')))
                gradient = torch.autograd.grad(loss, tuple(model.parameters()) + (rotation, translation), allow_unused=True)
                assert torch.isfinite(loss)
                iterations.append((result, float(loss), gradient, sampled.pop()))
            comparisons.append(iterations)
            rng_states.append((torch.get_rng_state(), torch.cuda.get_rng_state()))
        for expected, actual in zip(rng_states[0], rng_states[1]):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for iteration, (baseline, cached) in enumerate(zip(*comparisons)):
            torch.testing.assert_close(baseline[3], cached[3], rtol=0, atol=0)
            for name in baseline[0]:
                torch.testing.assert_close(baseline[0][name], cached[0][name], rtol=0, atol=0)
            error = errors(baseline[2], cached[2])
            assert error['relative_l2'] < .001, error
            assert all(value is not None for value in (*baseline[2][-2:], *cached[2][-2:])), 'Disconnected cached pose gradient'
            pose_errors = {name:errors((baseline[2][index],),(cached[2][index],))
                           for name,index in (('rotation',-2),('translation',-1))}
            assert all(value['relative_l2'] < .001 or value['max_absolute'] < 1e-7
                       for value in pose_errors.values()), pose_errors
            records.append(dict(frame=frame,iteration=iteration,baseline_loss=baseline[1],cached_loss=cached[1],
                                samples_and_forward_values='exact match',
                                map_rotation_translation_gradient_error=error))
            records[-1]['pose_gradient_errors'] = pose_errors
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(system='coslam',depth_cache_checks=records,
        cpu_and_cuda_rng_states='exact match after each three-iteration comparison',
        scope='Frozen trained field, 512 rays, three changing poses on each of three real frames; not whole-trajectory equivalence',
        **device_metadata()), indent=2) + '\n')
    print('DEPTH_CACHE_CHECKS_PASSED ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
