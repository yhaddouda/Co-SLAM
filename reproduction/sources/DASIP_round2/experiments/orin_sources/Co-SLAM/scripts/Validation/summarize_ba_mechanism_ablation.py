"""Audit and compare the four dense BA execution conditions, keeping profiles separate."""
import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from optimization.ba_experiments import ba_execution_flags
from scripts.Validation.execution_benchmark_common import rigid_ate
from scripts.Validation.summarize_ba_breakdown import profile_frames, write_csv

CONDITIONS = dict(baseline=(False, False), staging_only=(True, False),
                  pose_only=(False, True), baseline_fast=(True, True))


def normalized_config(config):
    result = copy.deepcopy(config)
    result['data'].pop('output')
    for key in ('variant', 'preload_rays', 'fast_pose_commit'):
        result['ba_experiment'].pop(key, None)
    for key in ('mode', 'warmup_frames', 'ba_output_csv'):
        result['timing'].pop(key, None)
    return result


def summarize(root):
    suite = json.loads((root / 'suite.json').read_text())
    groups, audit, summaries = {}, [], []
    reference_config, reference_hashes = None, None
    for job in suite['jobs']:
        directory = root / job['mode'] / job['scene'] / f"{job['variant']}_seed{job['seed']}_repeat{job['repeat']}"
        record = json.loads((directory / 'summary.json').read_text())
        config = json.loads((directory / 'config.json').read_text())
        if record['diagnostic'] or record['variant'] != job['variant']:
            raise ValueError(f'Invalid condition in {directory}')
        if record['scene'] != job['scene'] or record['seed'] != job['seed']:
            raise ValueError(f'Unmatched scene/seed in {directory}')
        if config['ba_experiment']['variant'] not in {'baseline', 'baseline_fast'} or config['ba_bucket']['enabled']:
            raise ValueError(f'Non-dense BA objective in {directory}')
        if record['ba_variant'] != config['ba_experiment']['variant']:
            raise ValueError(f'Variant/config disagreement in {directory}')
        if config['tracking_experiment']['variant'] != 'baseline' or any(
            config[name]['enabled'] for name in ('tracking_bucket', 'tracking_naive_pruning', 'tracking_depth_spatial')
        ) or config['training']['uniform_samples_until_depth']:
            raise ValueError(f'Non-baseline tracking in {directory}')
        expected_flags = dict(zip(('preload_rays', 'fast_pose_commit'), CONDITIONS[record['variant']]))
        if ba_execution_flags(config) != expected_flags or record['ba_execution'] != expected_flags:
            raise ValueError(f'Execution switches disagree in {directory}')
        expected_frames = job['frames'] or 2000
        if record['frames'] != expected_frames or record['warmup_frames'] != job['warmup']:
            raise ValueError(f'Incomplete frame horizon in {directory}')
        expected_mode = 'ba_breakdown' if job['mode'] == 'breakdown' else 'none'
        if record['timing_mode'] != expected_mode or not record['finite_parameters_and_poses']:
            raise ValueError(f'Invalid timing mode or model state in {directory}')
        normalized = normalized_config(config)
        hashes = json.loads((directory / 'source_hashes.json').read_text())
        for name, expected in hashes.items():
            if hashlib.sha256((directory / 'source' / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Source snapshot changed: {directory / name}')
        if reference_config is None:
            reference_config, reference_hashes = normalized, hashes
        if normalized != reference_config or hashes != reference_hashes:
            raise ValueError(f'Unmatched configuration/source in {directory}')
        with (directory / 'frames.csv').open() as handle:
            frames = list(csv.DictReader(handle))
        if [int(r['frame_id']) for r in frames] != list(range(1, expected_frames)):
            raise ValueError(f'Missing frame timings in {directory}')
        for row in frames:
            if bool(int(row['has_ba'])) != (int(row['frame_id']) % config['mapping']['map_every'] == 0):
                raise ValueError(f'Incorrect BA schedule in {directory}')
            if not all(np.isfinite(float(row[k])) and float(row[k]) >= 0 for k in ('wall_ms', 'ba_wall_ms')):
                raise ValueError(f'Invalid wall timing in {directory}')
        measured = [r for r in frames if int(r['frame_id']) >= record['warmup_frames']]
        ba_frames = [r for r in measured if int(r['has_ba'])]
        for metric, selected, column in (
            ('ba_invocations', ba_frames, 'ba_wall_ms'), ('ba_frames', ba_frames, 'wall_ms'),
            ('tracking_frames', [r for r in measured if not int(r['has_ba'])], 'wall_ms'),
        ):
            values = np.array([float(r[column]) for r in selected])
            expected = dict(count=len(values), mean_ms=values.mean(), median_ms=np.median(values),
                            p95_ms=np.percentile(values, 95), std_ms=values.std())
            if any(not np.isclose(record[metric][k], v, rtol=1e-12, atol=1e-12) for k, v in expected.items()):
                raise ValueError(f'Timing summary mismatch in {directory}')
        with np.load(directory / 'trajectories.npz') as trajectories:
            for name in ('ground_truth', 'estimated', 'relative'):
                if trajectories[name].shape != (expected_frames, 4, 4) or not np.isfinite(trajectories[name]).all():
                    raise ValueError(f'Invalid trajectory in {directory}')
            for name, field in (('estimated', 'ate_rmse_m'), ('relative', 'ate_relative_rmse_m')):
                actual = rigid_ate(trajectories['ground_truth'], trajectories[name]) / config['data']['sc_factor']
                if not np.isclose(actual, record[field], rtol=1e-10, atol=1e-12):
                    raise ValueError(f'ATE disagreement in {directory}')
        profile = None
        if job['mode'] == 'breakdown':
            profile, stages = profile_frames(directory / 'ba_breakdown.csv')
            if set(profile) != {int(r['frame_id']) for r in ba_frames}:
                raise ValueError(f'Unmatched profile horizon in {directory}')
            for stage, active in (
                ('BA_RAY_PRELOAD', expected_flags['preload_rays']),
                ('BA_COMMIT_BATCH_COPY', expected_flags['fast_pose_commit']),
                ('BA_COMMIT_REBUILD_STORE', not expected_flags['fast_pose_commit']),
            ):
                observed = {r['frame_id'] for r in stages if r['stage'] == stage}
                eligible = set(profile) if stage == 'BA_RAY_PRELOAD' else {
                    frame for frame, values in profile.items() if values['optimized_poses'] > 0
                }
                if observed != (eligible if active else set()):
                    raise ValueError(f'Execution block {stage} disagrees with switches in {directory}')
        group = groups.setdefault((job['scene'], job['seed'], job['repeat'], job['mode']), {})
        if record['variant'] in group:
            raise ValueError('Duplicate condition')
        group[record['variant']] = (record, profile)
        audit.append(dict(run=str(directory.relative_to(root)), frames=expected_frames, ba_calls=len(ba_frames),
                          source_hashes_verified=True, configuration_matched=True, ate_recomputed=True,
                          timing_statistics_recomputed=True, execution_flags=expected_flags,
                          profile_mechanisms_verified=True if profile is not None else None))
        summaries.append(dict(run=str(directory.relative_to(root)), mode=job['mode'], repeat=job['repeat'], **record))
    comparisons, effects = [], []
    for (scene, seed, repeat, mode), conditions in groups.items():
        if set(conditions) != set(CONDITIONS):
            raise ValueError('Each pass requires all four mechanism conditions')
        baseline = conditions['baseline'][0]
        for variant in CONDITIONS:
            record, profile = conditions[variant]
            row = dict(scene=scene, seed=seed, repeat=repeat, mode=mode, condition=variant,
                       frames=record['frames'], ba_calls=record['ba_invocations']['count'],
                       preload_rays=record['ba_execution']['preload_rays'],
                       fast_pose_commit=record['ba_execution']['fast_pose_commit'],
                       ate_mm=record['ate_rmse_m'] * 1000,
                       ate_change_pct=100 * (record['ate_rmse_m'] / baseline['ate_rmse_m'] - 1),
                       ba_mean_ms=record['ba_invocations']['mean_ms'],
                       ba_median_ms=record['ba_invocations']['median_ms'],
                       ba_p95_ms=record['ba_invocations']['p95_ms'],
                       ba_std_ms=record['ba_invocations']['std_ms'],
                       ba_reduction_pct=100 * (1 - record['ba_invocations']['mean_ms'] / baseline['ba_invocations']['mean_ms']),
                       ba_frame_mean_ms=record['ba_frames']['mean_ms'],
                       ba_frame_reduction_pct=100 * (1 - record['ba_frames']['mean_ms'] / baseline['ba_frames']['mean_ms']),
                       tracking_frame_mean_ms=record['tracking_frames']['mean_ms'])
            for key in ('cuda_total_ms', 'ray_prepare_cuda_ms', 'pose_commit_cuda_ms',
                        'setup_cuda_ms', 'loop_compute_cuda_ms', 'unattributed_cuda_ms'):
                row[key] = float(np.mean([r[key] for r in profile.values()])) if profile else ''
            comparisons.append(row)
        for label, control, treatment in (
            ('staging_without_fast_commit', 'baseline', 'staging_only'),
            ('staging_with_fast_commit', 'pose_only', 'baseline_fast'),
            ('fast_commit_without_staging', 'baseline', 'pose_only'),
            ('fast_commit_with_staging', 'staging_only', 'baseline_fast'),
        ):
            before = conditions[control][0]['ba_invocations']['mean_ms']
            after = conditions[treatment][0]['ba_invocations']['mean_ms']
            effects.append(dict(scene=scene, seed=seed, repeat=repeat, mode=mode, effect=label,
                                control=control, treatment=treatment, added_ba_ms=after - before,
                                time_reduction_pct=100 * (1 - after / before)))
    write_csv(root / 'comparison.csv', comparisons)
    write_csv(root / 'mechanism_effects.csv', effects)
    (root / 'summaries.json').write_text(json.dumps(summaries, indent=2))
    (root / 'AUDIT.json').write_text(json.dumps(dict(passed=True, runs=audit), indent=2))
    full_scene = all(record['frames'] == 2000 for record in summaries)
    horizon = '2,000 frames' if full_scene else ', '.join(str(n) for n in sorted({r['frames'] for r in summaries})) + ' frames'
    lines = [f'# Office1 dense BA mechanism ablation ({horizon})', '',
             'Fresh processes in conda `coslam`, serial GPU runs on Orin. Power/clocks are recorded in '
             '`hardware.json` and `hardware_telemetry.csv`. Four conditions share dense sampling, '
             'baseline tracking, CoherentPrime hash size 19, paired frame/phase seed(s), ray counts and '
             'optimizer schedules. Evaluation, meshes, checkpoints and active profilers are disabled.', '',
             f"Timing excludes the first {suite['arguments']['warmup']} frames; all frames remain in final ATE.", '',
             'Wall passes are the headline performance measurements. BA time includes preparation, '
             'all iterations and final pose commit, with CUDA synchronization at both boundaries. '
             'Complete BA-frame time also includes loading, tracking and keyframe insertion. '
             'ATE is the final saved trajectory, rigid-aligned in float64 without scale fitting. '
             'Positive time reduction means faster; positive ATE change means worse.', '',
             '| Scene / seed / repeat | Condition | Frames / BA calls | BA mean ms | Median / p95 ms | BA saved | BA-frame mean ms | BA-frame saved | ATE mm | ATE change |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        if r['mode'] == 'wall':
            lines.append(f"| {r['scene']} / {r['seed']} / {r['repeat']} | {r['condition']} | {r['frames']} / {r['ba_calls']} | "
                         f"{r['ba_mean_ms']:.3f} | {r['ba_median_ms']:.3f} / {r['ba_p95_ms']:.3f} | {r['ba_reduction_pct']:+.2f}% | "
                         f"{r['ba_frame_mean_ms']:.3f} | {r['ba_frame_reduction_pct']:+.2f}% | {r['ate_mm']:.3f} | {r['ate_change_pct']:+.2f}% |")
    lines += ['', '| Wall-pass mechanism contrast | Control → treatment | Added BA ms | Time saved |',
              '|---|---|---:|---:|']
    for r in effects:
        if r['mode'] == 'wall':
            lines.append(f"| {r['scene']} s{r['seed']} r{r['repeat']}: {r['effect']} | {r['control']} → {r['treatment']} | "
                         f"{r['added_ba_ms']:+.3f} | {r['time_reduction_pct']:+.2f}% |")
    lines += ['', 'Separate instrumented passes below explain where time went. Ray preparation sums '
              'preload + per-iteration ray selection + CPU-to-GPU transfer, counting moved work once. '
              'CUDA-event intervals include stream idle time caused by host submission; they are not '
              'kernel-only time. Their instrumented wall times do not enter the headline table.', '',
              '| Scene / seed / repeat | Condition | Stream BA ms | Ray preparation ms | Pose commit ms | Loop compute ms |',
              '|---|---|---:|---:|---:|---:|']
    for r in comparisons:
        if r['mode'] == 'breakdown':
            lines.append(f"| {r['scene']} / {r['seed']} / {r['repeat']} | {r['condition']} | "
                         f"{r['cuda_total_ms']:.3f} | {r['ray_prepare_cuda_ms']:.3f} | {r['pose_commit_cuda_ms']:.3f} | "
                         f"{r['loop_compute_cuda_ms']:.3f} |")
    lines += ['', 'Ray staging also puts the pose-index tensors on the GPU ahead of the loop. '
              'The baseline keeps those indices on CPU during per-iteration pose indexing in '
              '`resize_tensors_BA`. Consequently, the ray-preparation rows alone do not include '
              'every downstream consequence of staging; use complete BA wall time for its net effect.']
    scope = ('This full-scene test has one 2,000-frame trajectory per condition/pass. '
             'It measures the full sequence, including the late calls with roughly 400 keyframes. '
             'It does not establish repeat uncertainty, accuracy equivalence or effects in other power modes.'
             if full_scene else
             'This quick prefix test has one trajectory per condition/pass. It does not establish '
             'repeat uncertainty, accuracy equivalence, full-sequence gains or effects in other power modes. '
             'Pose finalization cost grows with keyframe count; this prefix has fewer keyframes than '
             'the earlier 2,000-frame report.')
    lines += ['', scope + ' Frame timings are correlated observations, not independent runs.', '',
              '`AUDIT.json` verifies all requested conditions, complete frame horizons, BA schedules, '
              'finite trajectories, independently recomputed raw/relative ATE, wall statistics, matching '
              'configurations and source snapshot hashes. Profile block presence also verifies actual '
              'staging and pose-commit activation. Detailed profiles and disjoint early/late '
              'windows are in [BA_BREAKDOWN_REPORT.md](BA_BREAKDOWN_REPORT.md). '
              'Exact runs and commands are in `suite.json`; marginal effects are in `mechanism_effects.csv`.', '']
    if (root / 'hardware.json').exists():
        hardware = json.loads((root / 'hardware.json').read_text())
        query = hardware.get('power_mode_query', {})
        if isinstance(query, dict):
            power = query.get('output', '').strip().replace('\n', '; ')
            lines += [f"Captured power query: `{power}`. See `hardware_summary.json` for telemetry ranges.", '']
    if (root / 'office1_ba_wall_curves.png').exists():
        lines += ['[Normal wall-time curves](office1_ba_wall_curves.png) '
                  '([PDF](office1_ba_wall_curves.pdf)) show the keyframe-dependent cost over the full run. '
                  '[Separate instrumented block curves](office1_ba_block_curves.png) '
                  '([PDF](office1_ba_block_curves.pdf)) show preparation and finalization intervals.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    return comparisons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summarize(args.root)
    print(args.root / 'REPORT.md')
