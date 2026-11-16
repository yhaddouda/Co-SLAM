"""Fresh-process EC-SLAM tracking/mapping execution experiment on Replica.

Uses the isolated source workspace, paired phase seeds and synchronized whole
invocation/frame-work boundaries. Mapping and tracking optimizer work does not
overlap; tracking waits before frame transfers in every benchmark condition.
"""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.Validation.execution_benchmark_common import (
    device_metadata, rigid_ate, seed_phase, snapshot, statistics, write_csv,
)

CONDITIONS = {
    'baseline': ('baseline', 'baseline'),
    'tracking_fast': ('baseline_fast', 'baseline'),
    'mapping_fast': ('baseline', 'baseline_fast'),
    'both_fast': ('baseline_fast', 'baseline_fast'),
    'tracking_reuse_pose': ('reuse_pose', 'baseline'),
    'tracking_device_pose': ('device_pose', 'baseline'),
    'tracking_defer_best': ('defer_best', 'baseline'),
    'mapping_cache_poses': ('baseline', 'cache_poses'),
    'mapping_reuse_inverse': ('baseline', 'reuse_inverse'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--variant', choices=tuple(CONDITIONS), required=True)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, default=ROOT / 'experiments/ec_slam_execution')
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    args.workspace, args.output = args.workspace.resolve(), args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'Refusing to overwrite existing output: {args.output}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    os.chdir(args.workspace)
    sys.path.insert(0, str(args.workspace))
    from src.utils import load_config
    from src.system import EC_SLAM
    from src.Scene import Scene
    from src.execution_experiments import tracking_flags, mapping_flags

    args.output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(f'configs/Replica/{args.scene}.yaml', 'configs/EC_SLAM.yaml')
    tracking_variant, mapping_variant = CONDITIONS[args.variant]
    cfg['tracking_experiment'] = dict(variant=tracking_variant)
    cfg['mapping_experiment'] = dict(variant=mapping_variant)
    cfg['HashGrid'].update(hash='CoherentPrime', morton_sort=False, hash_size=19)
    cfg.setdefault('timing', {}).update(mode='none', disable_eval=True, max_frames=args.frames or None)
    cfg['verbose'] = False
    cfg['debug'].update(flag=False, checkpoint=False, showTrackerLoss=False, showMapperLoss=False)
    cfg['data']['input_folder'] = str((args.workspace / cfg['data']['input_folder']).resolve())
    cfg['data']['output'] = str(args.output / 'artifacts')
    (args.output / 'config.json').write_text(json.dumps(cfg, indent=2))
    snapshot(args.output, args.workspace, (
        'src/Tracker.py', 'src/Mapper.py', 'src/Model.py', 'src/Scene.py',
        'src/utils.py', 'src/system.py', 'src/Datasets.py', 'src/profiling.py',
        'src/execution_experiments.py', 'src/tools/eval_ate.py',
    ))
    snapshot(args.output, ROOT, ('scripts/Validation/benchmark_ec_replica_execution.py',
                               'scripts/Validation/execution_benchmark_common.py'))
    # Merge source manifests from the two roots, rather than dropping EC hashes.
    hashes = {}
    import hashlib
    for path in (args.output / 'source').rglob('*.py'):
        hashes[str(path.relative_to(args.output / 'source'))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    seed_phase(args.seed)
    tracked, mapped, tracking_calls, mapping_calls = {}, {}, {}, {}
    counts = dict(tracking=0, mapping=0)

    original_init, original_append = Scene.__init__, Scene.append_keyframe
    def seeded_init(scene, config, obj_id, n_img, dirs, rgb, depth, frame_id):
        seed_phase(args.seed, int(frame_id), 3)
        original_init(scene, config, obj_id, n_img, dirs, rgb, depth, frame_id)
        if args.diagnostic:
            forward = scene.vMapModel.forward
            def checked_forward(*inputs, **kwargs):
                loss = forward(*inputs, **kwargs)
                kind = 'tracking' if kwargs.get('tracker', False) else 'mapping'
                counts[kind] += 1
                if not torch.isfinite(loss).all():
                    raise RuntimeError(f'Nonfinite EC-SLAM {kind} loss')
                return loss
            scene.vMapModel.forward = checked_forward
    def seeded_append(scene, rgb, depth, features, frame_id):
        seed_phase(args.seed, int(frame_id), 3)
        return original_append(scene, rgb, depth, features, frame_id)
    Scene.__init__, Scene.append_keyframe = seeded_init, seeded_append
    slam = EC_SLAM(cfg, SimpleNamespace(output=None, input_folder=None))
    # Wait before copies without enabling event/profiler instrumentation. This
    # removes next-frame transfer contamination from mapping wall measurements.
    slam.tracker.component_timing_enabled = True

    def install_frame_timer(worker, phase, records):
        starts = {}
        def begin(frame):
            if phase == 1:
                seed_phase(args.seed, frame, phase)
            torch.cuda.synchronize()
            starts[frame] = time.perf_counter()
        def end(frame):
            torch.cuda.synchronize()
            records[frame] = 1000 * (time.perf_counter() - starts.pop(frame))
            if phase == 1 and frame % 200 == 0:
                print(f'BENCHMARK_PROGRESS system=ec_slam scene={args.scene} variant={args.variant} frame={frame}', flush=True)
        worker._experiment_begin_frame, worker._experiment_end_frame = begin, end
    install_frame_timer(slam.tracker, 1, tracked)
    install_frame_timer(slam.mapper, 2, mapped)

    tracking_method = slam.tracker.optimize_cam_in_batch
    def tracking_timed(color, depth, features, optimizer, gt_camera, rotation, translation, frame):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = tracking_method(color, depth, features, optimizer, gt_camera, rotation, translation, frame)
        torch.cuda.synchronize()
        tracking_calls[int(frame)] = 1000 * (time.perf_counter() - started)
        return result
    slam.tracker.optimize_cam_in_batch = tracking_timed
    mapping_method = slam.mapper.optimize_map
    def mapping_timed(frame):
        seed_phase(args.seed, int(frame), 2)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = mapping_method(frame)
        torch.cuda.synchronize()
        mapping_calls[int(frame)] = 1000 * (time.perf_counter() - started)
        return result
    slam.mapper.optimize_map = mapping_timed

    # Propagate exceptions from worker threads to this process. A failed worker
    # otherwise leaves its peer waiting forever and can produce incomplete ATE.
    errors = []
    def guarded(call):
        def run(rank):
            try:
                call(rank)
            except BaseException as error:
                import traceback
                traceback.print_exc()
                errors.append(error)
        return run
    started = time.perf_counter()
    workers = [threading.Thread(target=guarded(slam.mapping), args=(1,), daemon=True),
               threading.Thread(target=guarded(slam.tracking), args=(0,), daemon=True)]
    for worker in workers:
        worker.start()
    while any(worker.is_alive() for worker in workers):
        if errors:
            raise RuntimeError('EC-SLAM worker failed') from errors[0]
        for worker in workers:
            worker.join(timeout=.2)
    elapsed = time.perf_counter() - started
    torch.cuda.synchronize()
    if errors:
        raise RuntimeError('EC-SLAM worker failed') from errors[0]
    expected = args.frames or 2000
    assert len(tracked) == expected and sorted(tracked) == list(range(expected))
    assert int(slam.mapping_idx[0]) == expected - 1
    model = slam.vis_dict[0].vMapModel
    assert all(torch.isfinite(p).all() for p in model.parameters())
    ground_truth = slam.gt_c2w_list.detach().numpy()
    estimated = slam.estimate_c2w_list.detach().numpy()
    relative = slam.estimate_relative_c2w_list.clone()
    for frame in range(expected):
        if frame % cfg['mapping']['every_frame']:
            keyframe = frame // cfg['mapping']['every_frame'] * cfg['mapping']['every_frame']
            relative[frame] = slam.estimate_relative_c2w_list[keyframe] @ slam.estimate_relative_c2w_list[frame]
    relative = relative.numpy()
    assert np.isfinite(estimated).all() and np.isfinite(relative).all()
    np.savez(args.output / 'trajectories.npz', ground_truth=ground_truth, estimated=estimated, relative=relative)
    records = [dict(frame_id=frame, tracking_frame_wall_ms=tracked[frame],
        tracking_wall_ms=tracking_calls.get(frame, 0.), has_mapping=int(frame in mapped),
        mapping_frame_wall_ms=mapped.get(frame, 0.), mapping_wall_ms=mapping_calls.get(frame, 0.))
        for frame in sorted(tracked)]
    write_csv(args.output / 'frames.csv', records)
    measured = [r for r in records if r['frame_id'] >= args.warmup]
    summary = dict(system='ec_slam', scene=args.scene, variant=args.variant, seed=args.seed,
        frames=expected, warmup_frames=args.warmup, diagnostic=args.diagnostic,
        tracking_variant=tracking_variant, mapping_variant=mapping_variant,
        tracking_execution_flags=tracking_flags(cfg), mapping_execution_flags=mapping_flags(cfg),
        ate_rmse_m=rigid_ate(ground_truth, estimated) / cfg['scale'],
        ate_relative_rmse_m=rigid_ate(ground_truth, relative) / cfg['scale'],
        ate_evaluator='float64 Horn rigid alignment; no scale fitting',
        tracking_invocations=statistics([r['tracking_wall_ms'] for r in measured]),
        tracking_frames=statistics([r['tracking_frame_wall_ms'] for r in measured]),
        mapping_invocations=statistics([r['mapping_wall_ms'] for r in measured if r['has_mapping']]),
        mapping_frames=statistics([r['mapping_frame_wall_ms'] for r in measured if r['has_mapping']]),
        run_wall_seconds=elapsed, finite_parameters_and_poses=True,
        checked_forwards=counts, rng_policy='paired frame/phase seeds, keyframe seeds',
        timing_policy='synchronized invocation and frame-work times; loader/wait excluded from tracking frame; loader included in mapping frame',
        scheduling_policy='tracking waits for mapping before frame transfer; optimizer schedules unchanged',
        **device_metadata())
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('BENCHMARK_RESULT ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
