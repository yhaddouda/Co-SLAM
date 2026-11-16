"""Independently audit the fixed eight-scene, three-seed pruning study.

Expected layout: controls/<Dataset>_<scene>_seed<N> and
<candidate>/<Dataset>_<scene>_seed<N>. Performance gates apply to ratios of
scene means, never individual seeds or means of per-seed percentages. The
10% tracking / 30% BA targets have five percentage points of tolerance.
No acceptance artifact is written for partial or nonqualifying studies.

The candidate jointly prunes historical ray constraints and historical pose
blocks. This audit does not label the resulting speedup neural-only. Tracking
can preserve all rays with interval-support pruning and dense anchors, or use
selected-batch ratio normalization. BA can use native selected-batch losses or
population cycle weights. Counts verify sampling schedules, not unbiased
optimizer trajectories or reconstruction quality. Saved diagnostics do not
contain per-ray depths/masks, so support-mask identities cannot be replayed.
V8 also audits dense-anchor pose selection and its balanced complementary-half
schedule. The seven-appearances result is a structural coverage proof, not a
claim that saved diagnostics identify each queried ray or the winning pose.
Historical BA groups can pass through a two-phase transition band before the
longer mature cycle. Native inherited hash sizes are retained per scene.

Accept exactly either the frozen 17-source manifest or the cleaned 15-source
manifest, with one identical hash mapping across every run of a study. Installed
source provenance and optional CPU cleanup evidence are supplemental; original
GPU measurements are always attributed to their own archived source bytes.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np


# Chosen from data availability and scene diversity before new outcomes.
SCENES = (
    ('Replica', 'office0', 2000), ('Synthetic', 'tg', 395),
    ('Tum', 'fr1_desk', 592), ('ScanNet', 'scene0106', 2324),
    ('Replica', 'room0', 2000), ('Synthetic', 'ck', 1211),
    ('Tum', 'fr3_office', 2515), ('ScanNet', 'scene0000', 5578),
)
NATIVE_HASH_SIZES = {('ScanNet', 'scene0106'): 14, ('ScanNet', 'scene0000'): 16}
STAGES = (
    ('Tracking', 'tracking_invocations', 'tracking', 'tracking_wall_ms'),
    ('Bundle_Adjustment', 'ba_invocations', 'ba', 'ba_wall_ms'),
)
FROZEN_REQUIRED_SOURCES = {
    'coslam.py', 'model/scene_rep.py', 'model/utils.py',
    'optimization/ba_experiments.py', 'optimization/utils.py',
    'model/keyframe.py', 'scripts/Validation/benchmark_ba_accuracy.py',
    'scripts/Validation/rank_replica_ba_scenes.py', 'datasets/dataset.py',
    'optimization/tracking_experiments.py', 'model/layer_timing.py',
    'scripts/Validation/check_guarded_objective.py',
    'optimization/sample_pruning.py', 'optimization/ray_selection.py',
    'optimization/ray_interpolation.py', 'optimization/cohort_selection.py',
    'optimization/pose_averaging.py',
}

CLEAN_REQUIRED_SOURCES = FROZEN_REQUIRED_SOURCES - {
    'optimization/ray_interpolation.py', 'optimization/pose_averaging.py',
}
SOURCE_MANIFESTS = {
    'frozen_v8_17': frozenset(FROZEN_REQUIRED_SOURCES),
    'clean_15': frozenset(CLEAN_REQUIRED_SOURCES),
}
# Backward-compatible name for callers auditing the original frozen study.
REQUIRED_SOURCES = FROZEN_REQUIRED_SOURCES


def check(condition, message):
    """Do not make integrity validation disappear under python -O."""
    if not condition:
        raise ValueError(message)


def close(actual, expected, message, atol=1e-8, rtol=1e-10):
    check(math.isfinite(actual) and math.isfinite(expected)
          and math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
          f'{message}: got {actual}, expected {expected}')


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def integer(value, message):
    result = int(value)
    check(float(value) == result, message)
    return result


def rigid_rmse(prediction, truth):
    """Independent NumPy rigid alignment: translation + rotation, no scale."""
    x, y = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    check(x.shape == y.shape and x.ndim == 2 and x.shape[1] == 3,
          'Trajectory translation shapes differ')
    check(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite trajectory')
    x, y = x - x.mean(0), y - y.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1
    rotation = vt.T @ correction @ u.T
    return float(np.sqrt(np.mean(np.sum((x @ rotation.T - y) ** 2, axis=1))))


def nonsampling_config(config):
    copied = json.loads(json.dumps(config))
    copied.pop('sample_pruning')
    copied['data'].pop('output')
    return copied


def reference_schedule(config, frames, stage):
    """Native forward populations, independent of sampler diagnostics."""
    if stage == 'Tracking':
        return {frame: (int(config['tracking']['iter']), int(config['tracking']['sample']))
                for frame in range(1, frames)}
    sample = int(config['mapping']['sample'])
    minimum = int(config['mapping']['min_pixels_cur'])
    keyframe_every = int(config['mapping']['keyframe_every'])
    iterations = int(config['mapping']['iters'])
    # A frame becomes a keyframe after its BA call. Frame zero already exists.
    return {frame: (iterations, sample + max(sample // ((frame - 1) // keyframe_every + 1), minimum))
            for frame in range(int(config['mapping']['map_every']), frames,
                               int(config['mapping']['map_every']))}


def reference_cohort(config, frame, seed):
    """Independent integer definition; never call the implementation helper."""
    mapping, sampling = config['mapping'], config['sample_pruning']
    count = integer(sampling['cohort_count'], 'Noninteger cohort count')
    recent = integer(sampling['recent_keyframes'], 'Noninteger recent-keyframe count')
    transition = integer(sampling.get('cohort_transition_keyframes', 0),
                         'Noninteger transition-keyframe count')
    period = integer(sampling.get('cohort_transition_period', 2),
                     'Noninteger transition period')
    check(count >= 1 and recent >= 0 and transition >= 0, 'Invalid cohort configuration')
    check(period == 2, 'The maturation transition period must be two')
    check(not transition or (count >= 4 and count % period == 0),
          'A transition band requires an even mature cohort count of at least four')
    history = len(range(0, frame, int(mapping['keyframe_every'])))
    phase = (frame // int(mapping['map_every']) + seed) % count
    anchors = {0} | set(range(max(1, history - recent), history))
    transition_groups = {group for group in range(history)
                         if group not in anchors and history - group <= recent + transition}
    active = [group for group in range(history)
              if group in anchors or
              (group % period == phase % period if group in transition_groups
               else group % count == phase)]
    keep_empty = sampling.get('cohort_keep_empty_pose_blocks', False)
    check(isinstance(keep_empty, bool), 'cohort_keep_empty_pose_blocks must be a boolean')
    optimized = (list(range(1, history)) if keep_empty
                 else [group for group in active if group != 0]) if history >= 2 else []
    current = bool(history >= 2 and mapping['optim_cur'])
    return dict(frame=frame, numhistory=history, phase=phase,
                active_history_indices=active, optimized_history_indices=optimized,
                optim_cur=current, cohort_count=count, recent_keyframes=recent,
                cohort_transition_keyframes=transition, cohort_transition_period=period,
                optimized_pose_count=len(optimized) + int(current),
                cohort_keep_empty_pose_blocks=keep_empty), anchors


def tracking_anchor_iterations(config):
    """Dense tracking iterations; cyclic complements are unanchored at zero."""
    sampling = config['sample_pruning']
    policy = sampling.get('cohort_tracking', 'support')
    total = integer(config['tracking']['iter'], 'Noninteger tracking iterations')
    every = integer(sampling.get('tracking_anchor_every', 0), 'Noninteger tracking anchor interval')
    check(total >= 1 and every >= 0, 'Invalid tracking anchor schedule')
    if policy == 'ray_subset' or (policy == 'ray_cycle' and every == 0):
        return set()
    anchors = {0, total - 1}
    if every:
        anchors.update(range(0, total, every))
    return anchors


def cyclic_retained_count(population, groups, seed, frame, iteration):
    """Replay complement cardinality, independently of spatial/depth ordering."""
    groups = integer(groups, 'Noninteger tracking cycle group count')
    check(groups >= 2 and population >= 0, 'Invalid cyclic tracking population')
    if population < 2:
        return population
    count = min(groups, population)
    sampling_seed = seed * 1000003 + frame * 1009 + 71
    initial_phase = int(np.random.default_rng(sampling_seed).integers(count))
    omitted_group = (initial_phase + iteration) % count
    omitted_count = population // count + int(omitted_group < population % count)
    return population - omitted_count


def inspect_tracking_handoff(config, control=False):
    """Audit pose-output options without importing the runtime implementation."""
    sampling, tracking = config['sample_pruning'], config['tracking']
    average = sampling.get('tracking_pose_average', 0)
    check(not isinstance(average, bool), 'Tracking pose average must not be boolean')
    average = integer(average, 'Noninteger tracking pose average')
    check(average == 0 or 2 <= average <= int(tracking['iter']),
          'Tracking pose average must be zero or between two and tracking.iter')
    selection = sampling.get('tracking_pose_selection', 'native')
    check(selection in ('native', 'dense_anchors'), 'Unknown tracking pose selection')
    check(isinstance(tracking['best'], bool), 'tracking.best must retain its native boolean setting')
    if control:
        check(selection == 'native' and average == 0,
              'Primary dense controls must retain native pose selection without averaging')
    eligible = set(range(int(tracking['iter'])))
    if selection == 'dense_anchors':
        check(average == 0, 'Dense-anchor pose selection and pose averaging cannot be combined')
        interval = sampling.get('tracking_anchor_every', 0)
        check(not isinstance(interval, bool), 'Tracking anchor interval must not be boolean')
        check(integer(interval, 'Noninteger tracking anchor interval') > 0,
              'Dense-anchor pose selection requires a positive anchor interval')
        check(sampling['policy'] == 'ray_cohort'
              and sampling.get('cohort_tracking') == 'ray_cycle',
              'Pruned dense-anchor pose selection requires cyclic ray complements')
        eligible = tracking_anchor_iterations(config)
        check({0, int(tracking['iter']) - 1}.issubset(eligible),
              'Dense-anchor pose selection must evaluate first and last poses densely')
    return dict(tracking_pose_average=average, tracking_pose_selection=selection,
                tracking_native_best=tracking['best'],
                tracking_eligible_best_iterations=sorted(eligible),
                tracking_saved_pose_rule=('tail_average' if average else
                    'best_eligible_pre_step' if tracking['best'] else 'last_pre_step'))


def cyclic_exposure_proof(population, groups, iterations, anchors):
    """Prove coverage for every initial phase, independent of spatial ordering.

    Each original ray occupies exactly one ordered rank and therefore one
    residue group. Dense anchors add one appearance to every group. Each
    non-anchor update omits exactly one residue group. Group cardinalities
    affect query totals but not the appearance count of an individual ray.
    For q=2 and ten updates, removing anchors 0,3,6,9 leaves three odd and
    three even iteration numbers; each ray consequently appears 4+3=7 times,
    including when the two groups have unequal cardinality for odd N.
    """
    population = integer(population, 'Noninteger tracking population')
    groups = integer(groups, 'Noninteger tracking cycle group count')
    iterations = integer(iterations, 'Noninteger tracking iteration count')
    check(population >= 2 and groups >= 2 and iterations >= 1,
          'A complementary-group coverage proof requires at least two rays')
    check(set(anchors).issubset(range(iterations)), 'Anchor iteration outside tracking schedule')
    groups = min(groups, population)
    sizes = [population // groups + int(group < population % groups)
             for group in range(groups)]
    by_phase = [[sum(iteration in anchors or (phase + iteration) % groups != group
                     for iteration in range(iterations)) for group in range(groups)]
                for phase in range(groups)]
    minimum = min(min(counts) for counts in by_phase)
    maximum = max(max(counts) for counts in by_phase)
    seven = groups == 2 and iterations == 10 and set(anchors) == {0, 3, 6, 9}
    if seven:
        check(minimum == maximum == 7, 'Complementary-half anchor schedule lost its exact seven-exposure property')
    return dict(population=population, effective_groups=groups,
                group_sizes=sizes, dense_anchor_iterations=sorted(anchors),
                group_appearances_by_initial_phase=by_phase,
                minimum_appearances_per_ray=minimum, maximum_appearances_per_ray=maximum,
                queries_in_ray_units_by_initial_phase=[sum(size * count for size, count in zip(sizes, counts))
                                                       for counts in by_phase],
                equal_exposure_for_every_initial_phase=minimum == maximum,
                q2_anchor3_exactly_seven_appearances=seven)


def inspect_cohorts(path, config, summary, frames, seed):
    """Replay native Python ray draws and the active pose schedule, on CPU."""
    check(summary['python_version'] == sys.version.split()[0],
          f'{path}: exact random.sample replay requires Python {summary["python_version"]}, '
          f'not {sys.version.split()[0]}')
    database = summary['ray_database']
    check(set(database) == {'height', 'width', 'rays_per_keyframe'}, f'{path}: invalid ray database metadata')
    check(config['data']['downsample'] == config['data']['trainskip'] == 1,
          f'{path}: this fixed full-sequence study requires downsample=trainskip=1')
    edge = int(config['cam'].get('crop_edge', 0))
    original_h, original_w = int(config['cam']['H']), int(config['cam']['W'])
    # BaseDataset computes the saved-ray population before TUM's image resize.
    expected_saved = int((original_h - 2 * edge) * (original_w - 2 * edge)
                         * config['mapping']['n_pixels'])
    image_h, image_w = (config['cam']['crop_size'] if config['dataset'] == 'tum'
                        else (original_h, original_w))
    expected_database = dict(height=int(image_h) - 2 * edge,
                             width=int(image_w) - 2 * edge,
                             rays_per_keyframe=expected_saved)
    check(database == expected_database, f'{path}: ray database differs from native dataset settings')
    saved = integer(database['rays_per_keyframe'], 'Noninteger saved-ray count')
    image_pixels = int(database['height']) * int(database['width'])
    check(saved > 0 and image_pixels > 0, f'{path}: empty ray database')
    pose_rows = read_json(path / 'cohort_pose_diagnostics.json')
    schedule = reference_schedule(config, frames, 'Bundle_Adjustment')
    check([r['frame'] for r in pose_rows] == list(schedule),
          f'{path}: missing, duplicate, or out-of-order cohort pose records')
    replay, dense_pose_blocks, selected_pose_blocks = {}, 0, 0
    phases = []
    mapping, sampling = config['mapping'], config['sample_pruning']
    for row in pose_rows:
        frame = integer(row['frame'], 'Noninteger BA frame')
        expected, anchors = reference_cohort(config, frame, seed)
        check(set(row) == set(expected), f'{path}: cohort pose diagnostic schema differs')
        check(row == expected, f'{path}: cohort phase, active poses, or optimized poses differ at frame {frame}')
        history, count = expected['numhistory'], expected['cohort_count']
        period = expected['cohort_transition_period']
        transition_groups = {group for group in range(history) if group not in anchors
            and history - group <= expected['recent_keyframes'] + expected['cohort_transition_keyframes']}
        active = set(expected['active_history_indices'])
        phases.append(expected['phase'])
        # Fixed age-class coverage: anchors every phase, transition groups
        # every second phase, mature groups once. This does not claim the
        # same inclusion frequencies while groups age during online SLAM.
        for group in range(history):
            coverage = sum(group in anchors or
                (group % period == phase % period if group in transition_groups
                 else group % count == phase) for phase in range(count))
            expected_coverage = count if group in anchors else count // period if group in transition_groups else 1
            check(coverage == expected_coverage, f'{path}: invalid cycle coverage')
        dense_pose_blocks += (history - 1 + int(mapping['optim_cur'])) if history >= 2 else 0
        selected_pose_blocks += expected['optimized_pose_count']
        generator = random.Random(seed * 1000003 + frame * 17 + 2)
        current_rays = max(int(mapping['sample']) // history, int(mapping['min_pixels_cur']))
        check(current_rays <= image_pixels, f'{path}: invalid native current-ray count')
        for iteration in range(int(mapping['iters'])):
            draws = generator.sample(range(history * saved), int(mapping['sample']))
            # This second draw must be consumed even though every current ray
            # is retained, otherwise subsequent iteration replays are wrong.
            generator.sample(range(image_pixels), current_rays)
            groups = [index // saved for index in draws]
            retained_groups = [group for group in groups if group in active]
            old = sum(group not in anchors for group in retained_groups) if count > 1 else 0
            transitional = sum(group in transition_groups for group in retained_groups)
            mature = old - transitional
            anchor = len(retained_groups) - old + current_rays
            weighting = sampling.get('cohort_weighting', 'population')
            check(weighting in ('population', 'native'), f'{path}: unknown cohort loss weighting')
            mass = count * mature + period * transitional + anchor if weighting == 'population' else old + anchor
            replay[(frame, iteration)] = dict(selected_rays=anchor + old,
                cohort_old_rays=old, anchor_rays=anchor, weight_sum=mass,
                cohort_transition_rays=transitional, cohort_mature_rays=mature)
    check(all(right == (left + 1) % int(sampling['cohort_count'])
              for left, right in zip(phases, phases[1:])), f'{path}: incomplete rotation schedule')
    return replay, dict(dense_pose_blocks=dense_pose_blocks,
                        selected_pose_blocks=selected_pose_blocks,
                        pose_blocks_pruned_pct=100 * (1 - selected_pose_blocks / dense_pose_blocks),
                        ba_transition_rays=sum(r['cohort_transition_rays'] for r in replay.values()),
                        ba_mature_rays=sum(r['cohort_mature_rays'] for r in replay.values()),
                        ba_anchor_rays=sum(r['anchor_rays'] for r in replay.values()),
                        ba_total_weight_mass=sum(r['weight_sum'] for r in replay.values()),
                        cohort_ba_calls=len(pose_rows))


def source_manifest_name(hashes):
    """Only the two exact source sets are recognized; no subsets/supersets."""
    check(isinstance(hashes, dict), 'Source hashes must be a mapping')
    for name, required in SOURCE_MANIFESTS.items():
        if set(hashes) == required:
            return name
    raise ValueError('Runtime manifest must be exactly frozen V8 (17 files) '
                     'or clean (15 files); unknown, missing, or extra paths found')


def inspect_sources(path):
    """Validate every archived byte against that run's own exact manifest."""
    hashes = read_json(path / 'source_hashes.json')
    source_manifest_name(hashes)
    for name, digest in hashes.items():
        relative = Path(name)
        check(not relative.is_absolute() and '..' not in relative.parts,
              f'{path}: invalid source path')
        check(isinstance(digest, str) and len(digest) == 64
              and all(character in '0123456789abcdef' for character in digest),
              f'{path}: invalid SHA-256 digest for {name}')
        check(hashlib.sha256((path / 'source' / relative).read_bytes()).hexdigest() == digest,
              f'{path}: archived source hash differs for {name}')
    return hashes


def installed_source_provenance():
    """Report installed bytes separately; they never validate an old archive."""
    root = Path(__file__).resolve().parents[2]
    present = {name for name in FROZEN_REQUIRED_SOURCES if (root / name).is_file()}
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in sorted(present)}
    manifest = next((name for name, required in SOURCE_MANIFESTS.items()
                     if present == required), None)
    return dict(root=str(root), manifest=manifest, source_hashes=hashes)


def cleanup_equivalence_metadata(path, archive_hashes, installed):
    """Bind supplemental CPU evidence to both source sets without a GPU claim."""
    if path is None:
        return None
    path = Path(path)
    evidence = read_json(path)
    check(evidence.get('schema_version') == 1, 'Unknown cleanup-equivalence schema')
    frozen, clean = evidence.get('frozen_source_hashes'), evidence.get('clean_source_hashes')
    check(source_manifest_name(frozen) == 'frozen_v8_17'
          and source_manifest_name(clean) == 'clean_15', 'Cleanup evidence manifest mismatch')
    check(archive_hashes in (frozen, clean),
          'Cleanup evidence does not describe these archived runtime hashes')
    check(installed['source_hashes'] in (frozen, clean),
          'Cleanup evidence does not describe the currently installed runtime hashes')
    check(evidence.get('cpu_equivalence_passed') is True,
          'Cleanup evidence does not report passing CPU equivalence checks')
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                cpu_equivalence_reported=True, source_hash_bindings_verified=True,
                supplemental_only=True, replaces_archived_gpu_sources=False,
                clean_gpu_quality_or_timing_rerun=False,
                regression=evidence.get('regression'),
                limitations=evidence.get('limitations', []))


def inspect_run(path, dataset, scene, frames, seed):
    summary, config = read_json(path / 'summary.json'), read_json(path / 'config.json')
    identity = (dataset, scene, frames, seed)
    check(tuple(summary[k] for k in ('dataset', 'scene', 'frames', 'seed')) == identity,
          f'{path}: run identity or full sequence length differs')
    check(summary['finite_parameters_and_poses'] and not summary['diagnostic'],
          f'{path}: invalid parameters or diagnostic timing')
    check(summary['valid_gt_frames'] == frames, f'{path}: incomplete finite GT coverage')
    check(summary['warmup_frames'] == 20, f'{path}: timing warmup differs')
    check(summary['rng_policy'] == 'paired frame/phase seeds', f'{path}: RNG protocol differs')
    check(summary['sample_pruning'] == config['sample_pruning'],
          f'{path}: summary and saved sampling configuration differ')
    check(config['sample_pruning']['seed'] == seed, f'{path}: sampler seed differs')
    check(summary['run_wall_seconds'] > 0 and math.isfinite(summary['run_wall_seconds']),
          f'{path}: invalid total time')

    rows = read_csv(path / 'frames.csv')
    check([int(r['frame_id']) for r in rows] == list(range(1, frames)),
          f'{path}: missing, duplicate, or out-of-order timed frames')
    ba_frames = set(reference_schedule(config, frames, 'Bundle_Adjustment'))
    for row in rows:
        frame = int(row['frame_id'])
        check(int(row['has_ba']) == int(frame in ba_frames), f'{path}: BA schedule differs at {frame}')
        for key in ('wall_ms', 'tracking_wall_ms', 'ba_wall_ms'):
            value = float(row[key])
            check(math.isfinite(value) and value >= 0, f'{path}: invalid {key} at {frame}')
        check(float(row['tracking_wall_ms']) > 0, f'{path}: unmeasured tracking at {frame}')
        check((float(row['ba_wall_ms']) > 0) == (frame in ba_frames), f'{path}: unmeasured BA')
    measured = [r for r in rows if int(r['frame_id']) >= 20]
    for _, metric, label, column in STAGES:
        selected = measured if label == 'tracking' else [r for r in measured if int(r['has_ba'])]
        values = np.asarray([float(r[column]) for r in selected])
        check(len(values) == summary[metric]['count'], f'{path}: {metric} count differs')
        for field, expected in (
            ('mean_ms', float(values.mean())), ('median_ms', float(np.median(values))),
            ('p95_ms', float(np.percentile(values, 95))), ('std_ms', float(values.std())),
        ):
            close(summary[metric][field], expected, f'{path}: {metric}.{field}')

    with np.load(path / 'trajectories.npz') as trajectory:
        gt = trajectory['ground_truth']
        check(gt.shape == (frames, 4, 4) and np.isfinite(gt).all(), f'{path}: invalid GT')
        scale = float(config['data']['sc_factor'])
        check(math.isfinite(scale) and scale > 0, f'{path}: invalid scale')
        for field, metric in (('estimated', 'ate_rmse_m'), ('relative', 'ate_relative_rmse_m')):
            estimated = trajectory[field]
            check(estimated.shape == gt.shape and np.isfinite(estimated).all(), f'{path}: invalid {field}')
            value = rigid_rmse(estimated[:, :3, 3], gt[:, :3, 3]) / scale
            close(value, summary[metric], f'{path}: independently recomputed {metric}')

    hashes = inspect_sources(path)
    return summary, config, hashes


def inspect_pair(dense_path, selected_path, dataset, scene, frames, seed):
    dense, dc, dh = inspect_run(dense_path, dataset, scene, frames, seed)
    selected, sc, sh = inspect_run(selected_path, dataset, scene, frames, seed)
    check(dh == sh, f'{selected_path}: dense and candidate source differ')
    check(nonsampling_config(dc) == nonsampling_config(sc),
          f'{selected_path}: only sample_pruning and data.output may differ')
    for key in ('device', 'torch_version', 'python_version', 'map_optimizer_eps', 'ray_database'):
        check(dense[key] == selected[key], f'{selected_path}: {key} differs')
    for config in (dc, sc):
        for flag in ('ba_bucket', 'tracking_bucket', 'tracking_naive_pruning', 'tracking_depth_spatial'):
            check(not config.get(flag, {}).get('enabled', False), f'{selected_path}: legacy sampler {flag} enabled')
        check(config['ba_experiment']['variant'] == config['tracking_experiment']['variant'] == 'baseline',
              f'{selected_path}: execution experiment enabled')
        check(config['grid']['hash_size'] == NATIVE_HASH_SIZES.get((dataset, scene), 19)
              and config['grid']['hash'] == 'CoherentPrime' and not config['grid']['morton_sort'],
              f'{selected_path}: inherited scene hash capacity or shared encoding settings differ')
        check(config['training'].get('n_importance', 0) == 0, f'{selected_path}: importance stage differs')
    check(dc['sample_pruning']['policy'] == 'dense', f'{dense_path}: control is not dense')
    inspect_tracking_handoff(dc, control=True)
    handoff = inspect_tracking_handoff(sc)
    sampling = {k: v for k, v in sc['sample_pruning'].items() if k != 'seed'}
    check(sampling['enabled'] and sampling['policy'] == 'ray_cohort', f'{selected_path}: wrong pruning method')
    check(sampling['stages'] == ['Tracking', 'Bundle_Adjustment'], f'{selected_path}: stages differ')
    archive_manifest = source_manifest_name(sh)
    if archive_manifest == 'frozen_v8_17':
        check(sampling['ba_ray_fraction'] == 1., f'{selected_path}: BA must retain complete selected groups')
    else:
        removed = {'tracking_ray_fraction', 'ba_ray_fraction', 'ray_strategy',
                   'ray_estimator', 'interp_fraction', 'tracking_pose_average'}
        check(not (removed & sampling.keys()),
              f'{selected_path}: clean configurations must omit removed experimental options')
        check(sampling.get('cohort_tracking', 'support') in ('support', 'ray_cycle'),
              f'{selected_path}: clean runtime does not implement fixed ray subsets')
    tracking_policy = sampling.get('cohort_tracking', 'support')
    check(tracking_policy in ('support', 'ray_subset', 'ray_cycle'), f'{selected_path}: unknown cohort tracking policy')
    if tracking_policy == 'ray_subset':
        tracking_fraction = float(sampling['tracking_ray_fraction'])
        check(sampling['ray_strategy'] == 'systematic' and sampling['ray_estimator'] == 'self_normalized',
              f'{selected_path}: expected systematic selected-batch tracking')
        check(math.isfinite(tracking_fraction) and 0 < tracking_fraction < 1,
              f'{selected_path}: invalid tracking pruning budget')
    elif tracking_policy == 'support':
        margin = float(sampling['margin'])
        check(math.isfinite(margin) and margin >= 0, f'{selected_path}: invalid support margin')
    else:
        check(integer(sampling['tracking_cycle_groups'], 'Noninteger tracking cycle group count') >= 2,
              f'{selected_path}: invalid tracking cycle group count')
    tracking_anchors = tracking_anchor_iterations(sc)
    exposure = (cyclic_exposure_proof(sc['tracking']['sample'], sampling['tracking_cycle_groups'],
                                    sc['tracking']['iter'], tracking_anchors)
                if tracking_policy == 'ray_cycle' else None)
    check(not (dense_path / 'cohort_pose_diagnostics.json').exists(),
          f'{dense_path}: dense control must use native full-pose BA')
    cohort_replay, pose_counts = inspect_cohorts(selected_path, sc, selected, frames, seed)
    diagnostics = read_csv(selected_path / 'pruning_diagnostics.csv')
    check(all(r['stage'] in {s[0] for s in STAGES} for r in diagnostics),
          f'{selected_path}: unexpected pruning stage')
    row = dict(dataset=dataset, scene=scene, seed=seed, frames=frames,
               native_hash_size=sc['grid']['hash_size'],
               dense_ate_cm=100 * dense['ate_rmse_m'], selected_ate_cm=100 * selected['ate_rmse_m'],
               ate_change_pct=100 * (selected['ate_rmse_m'] / dense['ate_rmse_m'] - 1),
               dense_relative_ate_cm=100 * dense['ate_relative_rmse_m'],
               selected_relative_ate_cm=100 * selected['ate_relative_rmse_m'],
               dense_run_s=dense['run_wall_seconds'], selected_run_s=selected['run_wall_seconds'],
               **handoff, tracking_cyclic_exposure_proof=exposure,
               **pose_counts)
    lattice = int(sc['training']['n_samples_d'] + sc['training']['n_range_d'])
    for stage, metric, label, _ in STAGES:
        a, b = dense['stage_queries'][stage], selected['stage_queries'][stage]
        schedule = reference_schedule(sc, frames, stage)
        expected_forwards = sum(iters for iters, _ in schedule.values())
        expected_dense = sum(iters * rays * lattice for iters, rays in schedule.values())
        check(a['forwards'] == b['forwards'] == expected_forwards, f'{selected_path}: {stage} iterations differ')
        check(a['dense_samples'] == b['dense_samples'] == a['queried_samples'] == expected_dense,
              f'{selected_path}: {stage} original ray population differs')
        check(0 < b['queried_samples'] < expected_dense, f'{selected_path}: no actual {stage} sample pruning')
        check(dense[metric]['count'] == selected[metric]['count'], f'{selected_path}: stage timing counts differ')
        stage_rows = [r for r in diagnostics if r['stage'] == stage]
        anchors = tracking_anchors if stage == 'Tracking' else set()
        expected_ids = Counter((frame, iteration) for frame, (iters, _) in schedule.items()
                               for iteration in range(iters) if iteration not in anchors)
        actual_ids = Counter((int(r['frame']), int(r['iteration'])) for r in stage_rows)
        check(actual_ids == expected_ids,
              f'{selected_path}: {stage} diagnostics do not cover each non-anchor forward exactly once')
        anchor_queries = sum(len(anchors) * rays * lattice for _, rays in schedule.values())
        retained_per_frame = {}
        for r in stage_rows:
            frame = int(r['frame'])
            rays = integer(r['rays'], 'Noninteger original ray count')
            retained = integer(r['selected_rays'], 'Noninteger selected ray count')
            queried = integer(r['queried_samples'], 'Noninteger query count')
            check(rays == schedule[frame][1], f'{selected_path}: ray population differs at {frame}')
            if stage == 'Tracking' and tracking_policy == 'support':
                check(retained == rays and min(2, lattice) * rays <= queried <= rays * lattice,
                      f'{selected_path}: support pruning must retain every ray and at least its first two points')
                if frame in retained_per_frame:
                    check(queried == retained_per_frame[frame],
                          f'{selected_path}: fixed tracking support count changed within frame {frame}')
                retained_per_frame[frame] = queried
            else:
                check(0 < retained <= rays and queried == retained * lattice,
                      f'{selected_path}: impossible selected ray/query counts')
            check(integer(r['dense_samples'], 'Noninteger dense count') == rays * lattice,
                  f'{selected_path}: dense diagnostic reference differs')
            check(int(r.get('recovery_samples', 0)) == int(r.get('recovered_rays', 0)) == 0,
                  f'{selected_path}: unaccounted recovery queries')
            if stage == 'Tracking':
                if tracking_policy == 'ray_subset':
                    check(retained == math.ceil(tracking_fraction * rays),
                          f'{selected_path}: tracking retained budget differs')
                elif tracking_policy == 'ray_cycle':
                    expected_retained = cyclic_retained_count(rays, sampling['tracking_cycle_groups'],
                                                              seed, frame, int(r['iteration']))
                    check(retained == expected_retained,
                          f'{selected_path}: tracking cyclic complement count differs at frame {frame}')
                check(int(r['cohort_old_rays']) == 0 and int(r['anchor_rays']) == retained,
                      f'{selected_path}: tracking diagnostic categories differ')
                check(int(r['cohort_transition_rays']) == int(r['cohort_mature_rays']) == 0,
                      f'{selected_path}: tracking diagnostics contain BA age categories')
                # N is reference-population bookkeeping. The interval variant
                # keeps every ray; the ray-subset variant is a ratio estimate.
                close(float(r['weight_sum']), rays, f'{selected_path}: tracking reference population differs',
                      atol=1e-3, rtol=1e-6)
            else:
                expected = cohort_replay[(frame, int(r['iteration']))]
                for key in ('selected_rays', 'cohort_old_rays', 'anchor_rays',
                            'cohort_transition_rays', 'cohort_mature_rays'):
                    check(int(r[key]) == expected[key],
                          f'{selected_path}: {key} differs from independent native-ray replay at frame {frame}')
                close(float(r['weight_sum']), expected['weight_sum'],
                      f'{selected_path}: cohort loss weight total differs', atol=1e-3, rtol=1e-6)
        check(sum(int(r['dense_samples']) for r in stage_rows) + anchor_queries == expected_dense,
              f'{selected_path}: {stage} diagnostic dense totals differ')
        check(sum(int(r['queried_samples']) for r in stage_rows) + anchor_queries == b['queried_samples'],
              f'{selected_path}: {stage} actual queries differ from diagnostics plus dense anchors')
        if stage == 'Tracking' and exposure and exposure['equal_exposure_for_every_initial_phase']:
            expected_queries = (exposure['minimum_appearances_per_ray'] * int(sc['tracking']['sample'])
                                * lattice * len(schedule))
            check(b['queried_samples'] == expected_queries,
                  f'{selected_path}: tracking query total differs from independent per-ray coverage proof')
        reduction = 100 * (1 - b['queried_samples'] / expected_dense)
        close(reduction, b['reduction_pct'], f'{selected_path}: {stage} pruning percentage')
        close(a['reduction_pct'], 0., f'{dense_path}: dense control prunes queries')
        row.update({label + '_pruning_pct': reduction,
                    'dense_' + label + '_points': expected_dense,
                    'selected_' + label + '_points': b['queried_samples'],
                    'dense_' + label + '_ms': dense[metric]['mean_ms'],
                    'selected_' + label + '_ms': selected[metric]['mean_ms'],
                    label + '_dense_anchor_forwards': len(anchors) * len(schedule),
                    label + '_time_saved_pct': 100 * (1 - selected[metric]['mean_ms'] / dense[metric]['mean_ms'])})

    for stage in set(dense['stage_queries']) | set(selected['stage_queries']):
        if stage not in {s[0] for s in STAGES}:
            check(dense['stage_queries'].get(stage) == selected['stage_queries'].get(stage),
                  f'{selected_path}: initialization or other field work changed')
            counts = dense['stage_queries'][stage]
            check(counts['queried_samples'] == counts['dense_samples'], f'{selected_path}: initialization is pruned')
    initial = dense['stage_queries']['First_frame_mapping']
    check(initial['forwards'] == sc['mapping']['first_iters'], f'{selected_path}: initialization iterations changed')
    check(initial['dense_samples'] == sc['mapping']['first_iters'] * sc['mapping']['sample'] * lattice,
          f'{selected_path}: initialization ray population changed')

    fraction = float(sampling['tv_fraction'])
    check(0 < fraction <= 1 and math.isfinite(fraction), f'{selected_path}: invalid feature sampling budget')
    vertices = (int(sc['training']['smooth_pts']) - 1) ** 3
    forwards = selected['stage_queries']['Bundle_Adjustment']['forwards']
    expected_features = vertices * forwards
    dense_features, selected_features = dense['stage_feature_queries'], selected['stage_feature_queries']
    check(dense_features['Bundle_Adjustment'] == expected_features, f'{dense_path}: feature query population differs')
    tv_path = selected_path / 'smoothness_diagnostics.csv'
    tv_rows = read_csv(tv_path) if tv_path.exists() else []
    if fraction < 1:
        edges = max(1, int(vertices * fraction / 6))
        schedule = reference_schedule(sc, frames, 'Bundle_Adjustment')
        expected_ids = Counter((frame, iteration) for frame, (iters, _) in schedule.items()
                               for iteration in range(iters))
        check(Counter((int(r['frame']), int(r['iteration'])) for r in tv_rows) == expected_ids,
              f'{selected_path}: smoothness diagnostics do not cover every BA update')
        check(all(int(r['dense_samples']) == vertices and int(r['queried_samples']) == 6 * edges
                  and int(r['sampled_edges']) == 3 * edges for r in tv_rows),
              f'{selected_path}: sampled feature budget differs')
        check(selected_features['Bundle_Adjustment'] == 6 * edges * forwards,
              f'{selected_path}: actual feature queries differ')
        for key in ('dense_samples', 'queried_samples', 'sampled_edges'):
            check(selected['smoothness_queries'][key] == sum(int(r[key]) for r in tv_rows),
                  f'{selected_path}: feature summary and diagnostics differ')
    else:
        check(not tv_rows and selected_features['Bundle_Adjustment'] == expected_features,
              f'{selected_path}: full feature budget must restore native smoothness')
    for stage in set(dense_features) | set(selected_features):
        if stage != 'Bundle_Adjustment':
            check(dense_features.get(stage) == selected_features.get(stage),
                  f'{selected_path}: non-BA feature work changed')
    row.update(dense_feature_points=expected_features,
               selected_feature_points=selected_features['Bundle_Adjustment'],
               feature_pruning_pct=100 * (1 - selected_features['Bundle_Adjustment'] / expected_features),
               sample_pruning_verified=True)
    return row, sampling, sh, nonsampling_config(sc), len(dh) + len(sh), dense


def evaluate(root, candidate='selected', seeds=(0, 1, 2), require_all=True, cleanup_equivalence=None):
    root, seeds = Path(root), tuple(seeds)
    check(len(seeds) > 0 and len(set(seeds)) == len(seeds), 'Seeds must be distinct')
    check(Path(candidate).name == candidate and candidate not in ('controls', '.', '..'),
          'Candidate must be a sibling family directory name')
    cases, missing, source_hashes, method, hardware = [], [], None, None, None
    scene_configs, verified = {}, 0
    for dataset, scene, frames in SCENES:
        for seed in seeds:
            name = f'{dataset}_{scene}_seed{seed}'
            dense_path, selected_path = root / 'controls' / name, root / candidate / name
            absent = [str(p.relative_to(root)) for p in (dense_path, selected_path) if not (p / 'summary.json').exists()]
            if absent:
                missing.extend(absent)
                continue
            row, sampling, hashes, config, count, summary = inspect_pair(
                dense_path, selected_path, dataset, scene, frames, seed)
            if source_hashes is None:
                source_hashes, method = hashes, sampling
                hardware = {k: summary[k] for k in ('device', 'torch_version', 'python_version')}
            check(hashes == source_hashes, 'Runtime source must be identical across every fresh run')
            check(sampling == method, 'One common sampling method is required across every scene and seed')
            check({k: summary[k] for k in hardware} == hardware, 'Hardware/software differs between runs')
            key = (dataset, scene)
            if key in scene_configs:
                check(config == scene_configs[key], f'{key}: non-sampling settings changed between seeds')
            scene_configs[key] = config
            cases.append(row)
            verified += count
    scenes = []
    for dataset, scene, frames in SCENES:
        values = [r for r in cases if (r['dataset'], r['scene']) == (dataset, scene)]
        if not values:
            continue
        mean = lambda key: statistics.mean(r[key] for r in values)
        result = dict(dataset=dataset, scene=scene, frames=frames,
                      seeds=[r['seed'] for r in values], complete=len(values) == len(seeds),
                      dense_ate_cm=mean('dense_ate_cm'), selected_ate_cm=mean('selected_ate_cm'),
                      ate_change_pct=100 * (mean('selected_ate_cm') / mean('dense_ate_cm') - 1))
        for label in ('tracking', 'ba'):
            result['dense_' + label + '_ms'] = mean('dense_' + label + '_ms')
            result['selected_' + label + '_ms'] = mean('selected_' + label + '_ms')
            result[label + '_time_saved_pct'] = 100 * (1 - mean('selected_' + label + '_ms') / mean('dense_' + label + '_ms'))
        result['ate_gate'] = result['ate_change_pct'] <= 15
        result['tracking_gate'] = result['tracking_time_saved_pct'] >= 5
        result['ba_gate'] = result['ba_time_saved_pct'] >= 25
        result['tracking_target'] = result['tracking_time_saved_pct'] >= 10
        result['ba_target'] = result['ba_time_saved_pct'] >= 30
        result['passed'] = all(result[k] for k in ('ate_gate', 'tracking_gate', 'ba_gate'))
        scenes.append(result)
    complete = not missing and len(cases) == len(SCENES) * len(seeds)
    passed = bool(scenes) and all(s['passed'] for s in scenes) and (complete or not require_all)
    attribution_control = bool(method and method.get('cohort_keep_empty_pose_blocks', False))
    qualified = passed and complete and set(seeds) == {0, 1, 2} and not attribution_control
    installed = installed_source_provenance()
    provenance = dict(
        archived_manifest=source_manifest_name(source_hashes) if source_hashes is not None else None,
        archived_source_hashes=source_hashes,
        installed_runtime=installed,
        installed_matches_archived_runtime=source_hashes == installed['source_hashes'],
        validation_basis='Every run archive is checked against its own exact manifest; one identical hash mapping is required throughout the study.',
        installed_files_used_to_validate_archives=False,
        cleanup_equivalence=cleanup_equivalence_metadata(cleanup_equivalence, source_hashes, installed)
            if cleanup_equivalence is not None else None)
    return dict(passed=passed, qualified=qualified, complete=complete,
                candidate=candidate, seeds=list(seeds), require_all=require_all,
                cases=cases, scenes=scenes, missing_runs=missing,
                manifest=[list(s) for s in SCENES], source_files_verified=verified,
                source_hashes=source_hashes, source_provenance=provenance,
                method_config=method, hardware=hardware,
                identical_fresh_runtime_sources=True, independent_direct_and_relative_ate=True,
                native_inherited_hash_capacities_verified=True,
                wall_times_recomputed=True, original_ray_populations_and_iterations_verified=True,
                actual_queries_verified=True, unchanged_dense_initialization=True,
                joint_constraint_and_pose_block_pruning=True,
                attribution_control=attribution_control,
                cohort_phase_cycles_and_active_poses_verified=True,
                ba_group_counts_replayed_independently=True,
                ba_transition_and_mature_counts_and_weights_replayed=True,
                dense_tracking_anchor_query_counts_verified=True,
                native_tracking_best_setting_unchanged=True,
                tracking_pose_handoff_configuration_verified=True,
                tracking_cyclic_exposure_proved_from_schedule=bool(
                    method and method.get('cohort_tracking') == 'ray_cycle'),
                tracking_pose_selection_replayed_from_records=False,
                tracking_point_masks_replayed=False,
                tracking_ray_identity_cycles_replayed=False,
                inactive_pose_values_verified_from_trajectories=False,
                criterion='Every scene: three-seed mean ATE increase <=15%; complete tracking wall time reduction >=5% and BA >=25% (targets 10%/30%, tolerance 5 percentage points).')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--candidate', default='selected')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--partial', action='store_true', help='Audit available pairs without accepting a partial study')
    parser.add_argument('--write', action='store_true', help='Write acceptance.json only for a fully qualified study')
    parser.add_argument('--cleanup-equivalence', type=Path,
                        help='Optional supplemental CPU cleanup proof; never replaces original GPU archives')
    args = parser.parse_args()
    result = evaluate(args.root, args.candidate, tuple(args.seeds), not args.partial, args.cleanup_equivalence)
    if args.write:
        check(result['qualified'], 'Study is incomplete or does not qualify; no acceptance artifact written')
        (args.root / 'acceptance.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('passed', 'qualified', 'complete', 'scenes', 'missing_runs')}, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
