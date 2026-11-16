"""Compare dense and guarded BA losses/gradients on an actual frozen CUDA map."""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import config
from datasets.dataset import get_dataset
from model.scene_rep import JointEncoding
from optimization.ba_experiments import forward_ba_lattice
from optimization.ba_experiments import commit_ba_poses_fast
from optimization.utils import at_to_transform_matrix
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', default='office1')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=1024)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('This check requires CUDA')
    cfg = config.load_config(f'configs/Replica/{args.scene}.yaml')
    cfg['timing']['mode'] = 'none'
    cfg['grid']['morton_sort'] = False
    cfg['ba_experiment'] = dict(collect_samples=True)
    bounds = torch.tensor(cfg['mapping']['bound'], dtype=torch.float32, device='cuda')
    model = JointEncoding(cfg, bounds).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', mmap=True)['model'])
    data = get_dataset(cfg)
    results = []
    for frame, offset in ((0, 0.), (300, .02), (900, .2)):
        batch = data[frame]
        indices = torch.randperm(data.H * data.W, generator=torch.Generator().manual_seed(17))[:args.samples]
        directions = batch['direction'].reshape(-1, 3)[indices].cuda()
        color = batch['rgb'].reshape(-1, 3)[indices].cuda()
        depth = batch['depth'].reshape(-1, 1)[indices].cuda()
        pose = batch['c2w'].cuda()
        rays_d = directions @ pose[:3, :3].T
        translation = (pose[:3, 3] + offset).detach().requires_grad_(True)
        rays_o = translation[None, :].expand_as(rays_d)
        parameters = tuple(model.parameters()) + (translation,)
        comparisons = []
        for guarded in (False, True):
            torch.manual_seed(73)
            torch.cuda.manual_seed_all(73)
            ret = forward_ba_lattice(model, rays_o, rays_d, color, depth) if guarded else model.forward(rays_o, rays_d, color, depth)
            total = sum(cfg['training'][weight] * ret[loss] for weight, loss in
                        (('rgb_weight', 'rgb_loss'), ('depth_weight', 'depth_loss'),
                         ('sdf_weight', 'sdf_loss'), ('fs_weight', 'fs_loss')))
            gradients = torch.autograd.grad(total, parameters, allow_unused=True)
            comparisons.append((ret, tuple(g.detach() if g is not None else None for g in gradients)))
        dense, sparse = comparisons
        for key in ('rgb', 'depth', 'rgb_loss', 'depth_loss', 'sdf_loss', 'fs_loss'):
            torch.testing.assert_close(dense[0][key], sparse[0][key], rtol=2e-4, atol=2e-6)
        differences, norms = [], []
        for d, s in zip(dense[1], sparse[1]):
            if d is not None:
                differences.append((d.float() - s.float()).square().sum())
                norms.append(d.float().square().sum())
        relative_error = float((sum(differences) / sum(norms).clamp_min(1e-20)).sqrt())
        if relative_error > .003:
            raise AssertionError(f'Gradient relative error {relative_error} exceeds tolerance')
        results.append(dict(frame=frame, translation_offset_m=offset,
                            losses_dense={k:float(dense[0][k]) for k in ('rgb_loss','depth_loss','sdf_loss','fs_loss')},
                            losses_guarded={k:float(sparse[0][k]) for k in ('rgb_loss','depth_loss','sdf_loss','fs_loss')},
                            map_and_translation_gradient_relative_error=relative_error,
                            pose_gradient_dense=dense[1][-1].cpu().tolist(),
                            pose_gradient_guarded=sparse[1][-1].cpu().tolist(),
                            **model.ba_experiment_sample_counts[-1]))
    rotations = torch.randn(400, 3, device='cuda') * .3
    translations = torch.randn_like(rotations)
    batched = at_to_transform_matrix(rotations, translations)
    torch.cuda.synchronize()
    start = time.perf_counter()
    singleton = torch.stack([at_to_transform_matrix(rotations[i:i+1], translations[i:i+1]).detach().clone()[0]
                             for i in range(400)])
    torch.cuda.synchronize()
    original_ms = (time.perf_counter() - start) * 1000
    controller = SimpleNamespace(config=dict(mapping=dict(optim_cur=True)), est_c2w_data={})
    frame_ids = torch.arange(0, 2000, 5)
    torch.cuda.synchronize()
    start = time.perf_counter()
    commit_ba_poses_fast(controller, batched, frame_ids, 2000)
    torch.cuda.synchronize()
    batched_ms = (time.perf_counter() - start) * 1000
    maximum_difference = float((singleton - batched).abs().max())
    torch.testing.assert_close(singleton, batched, rtol=1e-5, atol=1e-6)
    pose_commit = dict(cameras=400, maximum_matrix_difference=maximum_difference,
                       original_singleton_commit_ms=original_ms, cached_batch_commit_ms=batched_ms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(scene=args.scene, checkpoint=str(args.checkpoint), results=results,
                                          pose_commit_equivalence=pose_commit), indent=2))
    print('PASS', json.dumps(results), flush=True)
    print('POSE_COMMIT_PASS', json.dumps(pose_commit), flush=True)


if __name__ == '__main__':
    main()
