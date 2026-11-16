"""Four-dataset BA experiment, final aligned ATE and synchronized wall timings.

Run each condition in a fresh process. Tracking and keyframe sampling have
frame-specific seeds shared by every condition, preventing BA RNG consumption
from changing subsequent tracking ray batches. Timed BA invocations include
all preprocessing; BA-frame wall time also includes tracking and data loading.
"""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import config
import coslam
from scripts.Validation.rank_replica_ba_scenes import trajectory_rmse

VARIANTS = ('baseline', 'bucket', 'guarded', 'stable', 'bucket_dense', 'bucket_reference', 'lattice', 'lattice_guarded',
            'lattice_guarded_preload', 'baseline_fast', 'lattice_guarded_fast')


def write_csv(path, records):
    if not records:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def statistics(values):
    x = np.asarray(values, dtype=float)
    return dict(count=len(x), mean_ms=float(x.mean()), median_ms=float(np.median(x)),
                p95_ms=float(np.percentile(x, 95)), std_ms=float(x.std())) if len(x) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('Tum', 'ScanNet', 'Synthetic', 'Replica'), required=True)
    parser.add_argument('--tracking', choices=('baseline', 'depth_spatial', 'balanced_dense'), default='depth_spatial')
    parser.add_argument('--scene', required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--frames', type=int, default=0, help='0 = full scene')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--margin', type=float, default=0.0, help='Extra distance beyond depth + truncation')
    parser.add_argument('--ba-embed-eps', type=float, help='Encoding Adam epsilon from the first BA only')
    parser.add_argument('--pruning', choices=('dense', 'tail', 'ht', 'adaptive', 'refine', 'refine_dense', 'support', 'support_tv', 'ray_cohort'))
    parser.add_argument('--tracking-pose-selection',choices=('native','dense_anchors'),default='native')
    parser.add_argument('--cohort-count',type=int,default=4)
    parser.add_argument('--cohort-transition-keyframes',type=int,default=0)
    parser.add_argument('--cohort-weighting',choices=('population','native'),default='native')
    parser.add_argument('--tracking-cycle-count',dest='tracking_cycle_groups',type=int,default=4)
    parser.add_argument('--cohort-tracking',choices=('support','ray_cycle'),default='support')
    parser.add_argument('--cohort-keep-empty-pose-blocks',action='store_true')
    parser.add_argument('--recent-keyframes',type=int,default=4)
    parser.add_argument('--tv-fraction',type=float,default=.125)
    parser.add_argument('--tracking-anchor-every',type=int,default=3)
    parser.add_argument('--anchor-every', type=int, default=0)
    parser.add_argument('--pruning-margin', type=float, default=.15)
    parser.add_argument('--free-stride', type=int, default=2)
    parser.add_argument('--residual-threshold', type=float, default=.35)
    parser.add_argument('--pruning-stages', nargs='+', choices=('Tracking', 'Bundle_Adjustment'),
                        default=['Tracking', 'Bundle_Adjustment'])
    parser.add_argument('--pixels', choices=('random', 'features'), default='random')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic', action='store_true', help='Check every loss; do not compare diagnostic timings')
    parser.add_argument('--export-mesh', action='store_true',
                        help='Export the final mesh after recording ATE/timing, preserving the experiment RNG protocol')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f'Refusing to overwrite experiment directory: {args.output}')
    if not torch.cuda.is_available():
        raise RuntimeError('ATE benchmarking requires the host CUDA device')
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = config.load_config(f'configs/{args.dataset}/{args.scene}.yaml')
    for name in ('tracking_bucket', 'tracking_naive_pruning', 'tracking_depth_spatial'):
        cfg.setdefault(name, {})['enabled'] = False
    cfg['tracking_depth_spatial'].update(collect_sample_count_stats=False, collect_fallback_stats=False, stats_only=False)
    cfg['tracking_depth_spatial']['enabled'] = args.tracking != 'baseline'
    cfg['tracking_depth_spatial']['mode'] = 'balanced_dense' if args.tracking == 'balanced_dense' else 'pruned'
    cfg['tracking_experiment'] = dict(variant='baseline')
    cfg['sample_pruning'] = dict(enabled=args.pruning is not None,
        policy=args.pruning or 'dense', margin=args.pruning_margin,
        free_stride=args.free_stride, residual_threshold=args.residual_threshold,
        stages=args.pruning_stages, pixel_strategy=args.pixels, seed=args.seed,
        tracking_pose_selection=args.tracking_pose_selection,
        tracking_anchor_every=args.tracking_anchor_every)
    if args.pruning == 'ray_cohort':
        cfg['sample_pruning'].update(cohort_count=args.cohort_count,recent_keyframes=args.recent_keyframes,
            cohort_transition_keyframes=args.cohort_transition_keyframes,tv_fraction=args.tv_fraction,
            cohort_weighting=args.cohort_weighting,cohort_tracking=args.cohort_tracking,tracking_cycle_groups=args.tracking_cycle_groups,
            cohort_keep_empty_pose_blocks=args.cohort_keep_empty_pose_blocks,
            tracking_anchor_every=args.tracking_anchor_every)
    if args.pruning == 'support_tv':
        cfg['sample_pruning'].update(anchor_every=args.anchor_every,tracking_anchor_every=args.tracking_anchor_every,tv_fraction=args.tv_fraction)
    if args.pruning == 'support':
        cfg['sample_pruning'].update(anchor_every=args.anchor_every)
    cfg['training']['uniform_samples_until_depth'] = False
    cfg['ba_bucket']['enabled'] = args.variant in ('bucket', 'guarded', 'stable', 'bucket_dense', 'bucket_reference')
    cfg['ba_bucket'].update(debug_log=False, morton_inside_bucket=False,
                            mode=args.variant if args.variant in ('guarded', 'stable') else 'legacy',
                            depth_margin=args.margin)
    cfg['ba_bucket'].pop('embed_eps', None)
    if args.ba_embed_eps is not None:
        cfg['ba_bucket']['embed_eps'] = args.ba_embed_eps
    cfg['ba_experiment'] = dict(variant='baseline' if args.variant in ('guarded', 'stable') else args.variant,
                               margin=args.margin, collect_samples=True)
    # Preserve each scene's inherited hash-table capacity in both conditions.
    cfg['grid'].update(hash='CoherentPrime', morton_sort=False)
    cfg.setdefault('timing', {}).update(mode='none', max_frames=args.frames or None, disable_eval=True)
    cfg.setdefault('profiling', {})['enabled'] = False
    cfg['mapping']['first_mesh'] = False
    cfg['mesh'].update(visualisation=False, vis=1000000000)
    cfg['data'].update(output=str(args.output), exp_name='artifacts')
    (args.output / 'artifacts').mkdir()
    (args.output / 'config.json').write_text(json.dumps(cfg, indent=2))
    source_dir = args.output / 'source'
    source_dir.mkdir()
    hashes = {}
    for name in ('coslam.py', 'model/scene_rep.py', 'model/utils.py', 'optimization/ba_experiments.py',
                 'optimization/utils.py', 'model/keyframe.py', 'scripts/Validation/benchmark_ba_accuracy.py',
                 'scripts/Validation/rank_replica_ba_scenes.py', 'datasets/dataset.py',
                 'optimization/tracking_experiments.py', 'model/layer_timing.py',
                 'scripts/Validation/check_guarded_objective.py', 'optimization/sample_pruning.py', 'optimization/ray_selection.py', 'optimization/cohort_selection.py'):
        path = Path(name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        target = source_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (args.output / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    coslam.cfg = cfg
    slam = coslam.CoSLAM(cfg)
    frames, ba_times, tracking_times = [], {}, {}
    state = dict(phase='', frame=0, checked_forwards=0)
    query_counts = {}
    feature_queries={}
    query_sdf=slam.model.query_sdf
    def counted_sdf(points,*inputs,**kwargs):
        if kwargs.get('embed',False):
            stage=getattr(slam.model,'_pruning_stage','unknown')
            feature_queries[stage]=feature_queries.get(stage,0)+points.numel()//3
        return query_sdf(points,*inputs,**kwargs)
    slam.model.query_sdf=counted_sdf
    network = slam.model.run_network
    def counted_network(points, *inputs, **kwargs):
        stage = getattr(slam.model, '_pruning_stage', 'unknown')
        row = query_counts.setdefault(stage, dict(queried_samples=0, dense_samples=0, forwards=0))
        row['queried_samples'] += points.numel() // 3
        return network(points, *inputs, **kwargs)
    slam.model.run_network = counted_network
    original_forward = slam.model.forward
    def counted_forward(rays_o, *inputs, **kwargs):
        stage = getattr(slam.model, '_pruning_stage', 'unknown')
        row = query_counts.setdefault(stage, dict(queried_samples=0, dense_samples=0, forwards=0))
        row['dense_samples'] += len(rays_o) * (cfg['training']['n_samples_d'] + cfg['training']['n_range_d'])
        row['forwards'] += 1
        return original_forward(rays_o, *inputs, **kwargs)
    slam.model.forward = counted_forward

    def phase_seed(frame, phase):
        seed = args.seed * 1000003 + frame * 17 + phase
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if phase == 1:
            slam._tracking_depth_spatial_rng = np.random.default_rng(seed)

    def wrap_phase(method, phase, name):
        def wrapped(batch, frame, *inputs, **kwargs):
            phase_seed(frame, phase)
            state.update(phase=name, frame=int(frame))
            if name in ('ba', 'tracking'):
                torch.cuda.synchronize()
                start = time.perf_counter()
            result = method(batch, frame, *inputs, **kwargs)
            if name in ('ba', 'tracking'):
                torch.cuda.synchronize()
                times = ba_times if name == 'ba' else tracking_times
                times[int(frame)] = (time.perf_counter() - start) * 1000
            state['phase'] = ''
            return result
        return wrapped

    slam.current_frame_mapping = wrap_phase(slam.current_frame_mapping, 4, 'current_mapping')
    slam.tracking_render = wrap_phase(slam.tracking_render, 1, 'tracking')
    slam.global_BA = wrap_phase(slam.global_BA, 2, 'ba')
    add_keyframe = slam.keyframeDatabase.add_keyframe
    def add_seeded(batch, *inputs, **kwargs):
        phase_seed(int(batch['frame_id']), 3)
        return add_keyframe(batch, *inputs, **kwargs)
    slam.keyframeDatabase.add_keyframe = add_seeded
    if args.diagnostic:
        original_loss = slam.get_loss_from_ret
        def checked_loss(ret, *inputs, **kwargs):
            state['checked_forwards'] += 1
            for key in ('rgb_loss', 'depth_loss', 'sdf_loss', 'fs_loss'):
                if not torch.isfinite(ret[key]).all():
                    raise RuntimeError(f'Nonfinite {key} in {state}')
            result = original_loss(ret, *inputs, **kwargs)
            if not torch.isfinite(result).all():
                raise RuntimeError(f'Nonfinite total loss in {state}')
            return result
        slam.get_loss_from_ret = checked_loss

    finish = slam.coslam_timing.finish_frame
    last_end = None
    def finish_timed(frame):
        nonlocal last_end
        finish(frame)
        torch.cuda.synchronize()
        end = time.perf_counter()
        if last_end is not None:
            frames.append(dict(frame_id=int(frame), wall_ms=(end - last_end) * 1000,
                               has_ba=int(frame in ba_times), ba_wall_ms=ba_times.get(frame, 0.),
                               tracking_wall_ms=tracking_times.get(frame, 0.)))
        last_end = time.perf_counter()
        if frame % 200 == 0:
            print(f'BENCHMARK_PROGRESS scene={args.scene} variant={args.variant} frame={frame}', flush=True)
    slam.coslam_timing.finish_frame = finish_timed
    # Trajectories suffice for this ATE experiment; no mesh extraction or large map checkpoint.
    slam.save_ckpt = lambda *inputs, **kwargs: None
    start = time.perf_counter()
    slam.run()
    elapsed = time.perf_counter() - start
    assert all(torch.isfinite(p).all() for p in slam.model.parameters())
    assert all(torch.isfinite(p).all() for p in slam.est_c2w_data.values())
    relative = slam.convert_relative_pose()
    gt = np.stack([slam.pose_gt[i].cpu().numpy() for i in sorted(slam.est_c2w_data)])
    scale = cfg['data']['sc_factor']
    # Match the standard evaluator: exclude nonfinite ScanNet GT poses.
    valid_ids = [i for i in sorted(slam.est_c2w_data) if np.isfinite(gt[i]).all()]
    if len(valid_ids) < 3:
        raise ValueError('Fewer than three valid GT poses for ATE')
    ate = trajectory_rmse(gt, {i: slam.est_c2w_data[i] for i in valid_ids}) / scale
    ate_relative = trajectory_rmse(gt, {i: relative[i] for i in valid_ids}) / scale
    np.savez(args.output / 'trajectories.npz', ground_truth=gt,
             estimated=np.stack([p.detach().cpu().numpy() for p in slam.est_c2w_data.values()]),
             relative=np.stack([relative[i].detach().cpu().numpy() for i in sorted(relative)]))
    write_csv(args.output / 'frames.csv', frames)
    measured = [r for r in frames if r['frame_id'] >= args.warmup]
    summary = dict(dataset=args.dataset, valid_gt_frames=len(valid_ids), scene=args.scene, variant=args.variant, seed=args.seed, margin=args.margin,
        ba_embed_eps=args.ba_embed_eps,
        frames=len(slam.est_c2w_data), diagnostic=args.diagnostic, tracking=args.tracking,
        ate_rmse_m=ate, ate_relative_rmse_m=ate_relative, warmup_frames=args.warmup,
        ba_invocations=statistics([r['ba_wall_ms'] for r in measured if r['has_ba']]),
        tracking_invocations=statistics([r['tracking_wall_ms'] for r in measured]),
        ba_frames=statistics([r['wall_ms'] for r in measured if r['has_ba']]),
        tracking_frames=statistics([r['wall_ms'] for r in measured if not r['has_ba']]),
        run_wall_seconds=elapsed, finite_parameters_and_poses=True,
        checked_forwards=state['checked_forwards'], rng_policy='paired frame/phase seeds',
        device=torch.cuda.get_device_name(0), torch_version=torch.__version__)
    summary['python_version']=sys.version.split()[0]
    summary['ray_database']=dict(height=slam.dataset.H,width=slam.dataset.W,
        rays_per_keyframe=slam.keyframeDatabase.num_rays_to_save)
    summary['sample_pruning'] = cfg['sample_pruning']
    tv=getattr(slam.model,'smoothness_diagnostics',[])
    write_csv(args.output/'smoothness_diagnostics.csv',tv)
    if tv:
        summary['smoothness_queries']={k:sum(r[k] for r in tv) for k in ('dense_samples','queried_samples','sampled_edges')}
    for row in query_counts.values():
        row['reduction_pct'] = 100 * (1 - row['queried_samples'] / row['dense_samples']) if row['dense_samples'] else None
    summary['stage_queries'] = query_counts
    summary['stage_feature_queries']=feature_queries
    diagnostics = getattr(slam.model, 'pruning_diagnostics', [])
    write_csv(args.output / 'pruning_diagnostics.csv', diagnostics)
    pose_diagnostics=getattr(slam,'_cohort_pose_diagnostics',[])
    if pose_diagnostics:
        (args.output/'cohort_pose_diagnostics.json').write_text(json.dumps(pose_diagnostics,indent=2))
    summary['map_optimizer_eps'] = [group['eps'] for group in slam.map_optimizer.param_groups]
    sample_counts = getattr(slam.model, 'ba_experiment_sample_counts', [])
    if sample_counts:
        summary['network_samples'] = {key: sum(r[key] for r in sample_counts) for key in sample_counts[0]}
        summary['network_samples']['reduction_pct'] = 100 * (1 - summary['network_samples']['queried_samples'] / summary['network_samples']['dense_samples'])
    if args.variant == 'guarded':
        from scripts.Validation.check_guarded_objective import check_guarded_objective
        summary['frozen_objective_check'] = check_guarded_objective(slam)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('BENCHMARK_RESULT ' + json.dumps(summary), flush=True)
    if args.export_mesh:
        mesh_start = time.perf_counter()
        frame = max(slam.est_c2w_data)
        slam.save_mesh(frame, voxel_size=cfg['mesh']['voxel_final'])
        mesh = args.output / 'artifacts' / f'mesh_track{frame}.ply'
        if not mesh.exists():
            raise RuntimeError(f'Mesh export did not create {mesh}')
        (args.output / 'mesh_export.json').write_text(json.dumps(dict(
            path=str(mesh), voxel_size=cfg['mesh']['voxel_final'],
            export_wall_seconds=time.perf_counter()-mesh_start,
            excluded_from_training_timings=True), indent=2))
        print('MESH_EXPORTED', mesh, flush=True)


if __name__ == '__main__':
    main()
