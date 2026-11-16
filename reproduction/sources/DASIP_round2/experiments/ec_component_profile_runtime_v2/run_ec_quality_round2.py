#!/usr/bin/env python3
"""Native EC-SLAM quality runner. This is NOT a whole-call timing benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rigid_rmse(estimated, truth):
    import numpy as np
    x, y = estimated[:, :3, 3], truth[:, :3, 3]
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    return float(np.sqrt(np.mean(np.sum((xc @ rotation - yc)**2, axis=1))))


def export_mesh(model, cfg, directory):
    """Query native WORLD-space SDF after SLAM; never normalize it twice."""
    import numpy as np
    import torch
    import trimesh
    import marching_cubes
    bounds = np.asarray(cfg['mapping']['marching_cubes_bound'], dtype=np.float64)
    voxel = float(cfg['mesh']['final_voxel_size'])
    axes = [np.linspace(a, b, int(np.ceil((b-a)/voxel))+1) for a, b in bounds]
    shape = tuple(len(a) for a in axes)
    total = int(np.prod(shape))
    values = np.empty(total, np.float32)
    with torch.no_grad():
        for start in range(0, total, 65536):
            flat = np.arange(start, min(start+65536, total))
            indices = np.unravel_index(flat, shape)
            points = np.stack([axes[i][indices[i]] for i in range(3)], -1)
            points = torch.as_tensor(points, device=cfg['device'], dtype=torch.float32)
            values[start:start+len(flat)] = model.query_sdf(points).reshape(-1).cpu().numpy()
    if not np.isfinite(values).all() or not (values.min() < 0 < values.max()):
        raise RuntimeError('Final field has no finite zero crossing')
    spacing = tuple(a[1]-a[0] for a in axes)
    vertices, faces = marching_cubes.marching_cubes(values.reshape(shape), 0., truncation=3.)
    vertices = vertices*np.asarray(spacing)+bounds[:, 0]
    colors = []
    with torch.no_grad():
        for start in range(0, len(vertices), 65536):
            points = torch.as_tensor(vertices[start:start+65536], device=cfg['device'], dtype=torch.float32)
            colors.append(model.query_color(points).cpu().numpy())
    mesh = trimesh.Trimesh(vertices, faces, vertex_colors=np.clip(np.concatenate(colors)*255, 0, 255).astype(np.uint8), process=False)
    target = directory/'final_mesh.ply'
    mesh.export(target)
    (directory/'mesh_export.json').write_text(json.dumps(dict(path=str(target), sha256=digest(target), vertices=len(vertices), faces=len(faces), requested_voxel_m=voxel, actual_spacing_m=spacing, world_bounds=bounds.tolist(), after_timing=True), indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--datadir', required=True, type=Path)
    parser.add_argument('--condition', choices=['BASE', 'MC'], required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--hash-size', choices=[16, 19], type=int, required=True)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--export-mesh', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    manifest = json.loads((root/'RUNTIME_MANIFEST.json').read_text())
    for name, expected in manifest['runtime_hashes'].items():
        if digest(root/name) != expected:
            raise RuntimeError('Frozen runtime file changed: '+name)
    import numpy as np
    import torch
    import yaml
    import tinycudann as tcnn
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    from src.utils import load_config
    cfg = load_config('configs/Replica/'+args.scene+'.yaml', 'configs/EC_SLAM.yaml')
    cfg['data']['input_folder'] = str(args.datadir.resolve())
    cfg['data']['output'] = str(args.output.resolve())
    cfg['HashGrid']['hash_size'] = args.hash_size
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=True))
    (args.output/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    import src.Datasets as datasets
    original_len = datasets.Replica.__len__
    reader = datasets.Replica(cfg)
    full_length = original_len(reader)
    if args.frames:
        datasets.Replica.__len__ = lambda self: min(args.frames, original_len(self))
    import src.Model as model_module
    original_encoder = model_module.get_encoder
    if args.condition == 'MC':
        # Same native encoder request, with only the explicit hash changed.
        def encoder(encoding, input_dim=3, n_bins=16, n_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=512):
            if 'hash' not in encoding.lower() and 'tiled' not in encoding.lower():
                return original_encoder(encoding, input_dim, n_bins, n_levels, level_dim, base_resolution, log2_hashmap_size, desired_resolution)
            instance = tcnn.Encoding(n_input_dims=input_dim, encoding_config=dict(otype='HashGrid', n_levels=n_levels, n_features_per_level=level_dim, log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution, per_level_scale=np.exp2(np.log2(desired_resolution/base_resolution)/(n_levels-1)), hash='Morton'), dtype=torch.float)
            return instance, instance.n_output_dims
        model_module.get_encoder = encoder
        from torch.utils.cpp_extension import load
        major, minor = torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST'] = str(major)+'.'+str(minor)
        extension = load(name='dasip_round2_ec_morton_'+str(major)+str(minor), sources=[str(root/'morton_cuda.cu')], extra_cuda_cflags=['-O3'], verbose=False)
        original_forward = model_module.HashGrid.forward
        order_check = []
        def forward(self, points, onlySdf=False):
            normalized = (points-self.bounding_box[:, 0])/(self.bounding_box[:, 1]-self.bounding_box[:, 0])
            with torch.no_grad():
                perm = torch.argsort(extension.morton3d_keys(normalized.float().contiguous(), 128), stable=False).long()
            values = original_forward(self, points[perm], onlySdf)
            output = torch.empty_like(values)
            output[perm] = values
            # One diagnostic, before the first optimization update. No RNG draw.
            if not order_check:
                with torch.no_grad():
                    reference = original_forward(self, points, onlySdf)
                    error = float((reference-output.detach()).abs().max().item())
                    if not torch.allclose(reference, output.detach(), atol=2e-5, rtol=2e-5):
                        raise RuntimeError('Morton inverse permutation failed: '+str(error))
                    order_check.append(error)
            return output
        model_module.HashGrid.forward = forward
    else:
        order_check = []
    from src.system import EC_SLAM
    # Native workers remain concurrent; fail visibly instead of hanging forever
    # if a worker throws and its peer waits at a native handshake.
    def worker_failure(info):
        traceback.print_exception(info.exc_type, info.exc_value, info.exc_traceback)
        sys.stderr.flush()
        os._exit(2)
    threading.excepthook = worker_failure
    from argparse import Namespace
    slam = EC_SLAM(cfg, Namespace(output=str(args.output)))
    torch.cuda.synchronize()
    begin = time.perf_counter()
    slam.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-begin
    estimate = slam.estimate_c2w_list.detach().cpu().numpy()
    gt = slam.gt_c2w_list.detach().cpu().numpy()
    relative = slam.estimate_relative_c2w_list.detach().cpu().numpy()
    corrected = relative.copy()
    every = int(cfg['mapping']['every_frame'])
    for i in range(len(corrected)):
        if i % every:
            corrected[i] = relative[(i//every)*every] @ relative[i]
    field = slam.vis_dict[0].vMapModel
    finite = all(np.isfinite(a).all() for a in (estimate, gt, relative, corrected)) and all(bool(torch.isfinite(p).all()) for p in field.parameters())
    if not finite or np.any(np.abs(estimate[:, 3, 3]-1)>1e-5) or int(slam.mapping_idx.item()) != len(estimate)-1:
        raise RuntimeError('Incomplete/nonfinite native EC sequence')
    np.savez(args.output/'trajectories.npz', estimated=estimate, corrected=corrected, ground_truth=gt, relative=relative)
    # Save the final learned field, which upstream pose-only checkpoints omit.
    torch.save(dict(state_dict=field.state_dict(), config=cfg), args.output/'final_field.pt')
    summary = dict(system='EC-SLAM', purpose='native_quality_not_whole_call_timing', scene=args.scene, condition=args.condition, hash_size=args.hash_size, seed=args.seed, frames=len(estimate), available_frames=full_length, full_sequence=len(estimate)==full_length, finite_parameters_and_poses=finite, ate_rmse_m=rigid_rmse(estimate, gt), corrected_ate_rmse_m=rigid_rmse(corrected, gt), native_scene_elapsed_seconds=elapsed, includes_native_thread_waits_and_pose_evaluation=True, permutation_check_max_abs=order_check, precision_request='native FP32 HashGrid and OneBlob', public_revision=manifest['public_revision'])
    if args.export_mesh:
        export_mesh(field, cfg, args.output)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
