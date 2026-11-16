#!/usr/bin/env python3
"""Write a scratch Markdown report only after independent V8 qualification.

This tool does not read a cached acceptance verdict and never runs SLAM or CUDA.
It calls validate_pruning8.evaluate(root, candidate, seeds=(0, 1, 2),
require_all=True) before creating any output. Only this exact V8 method is
described. Outputs must resolve below /tmp; promotion is a separate operation.

Optional --clean-proof supplies cleanup metadata, never replacement benchmark
evidence. --artifact-prefix can specify the relative location of the run root
when preparing a report that will later be promoted with its artifacts.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
from urllib.parse import quote


DEFAULT_BASE = Path('/tmp/coslam_pruning8_v8')
INSTALLED_VERIFIER = Path(__file__).resolve().parents[2] / 'scripts/Validation/validate_pruning8.py'
DEFAULT_VERIFIER = (INSTALLED_VERIFIER if INSTALLED_VERIFIER.is_file() else
                    DEFAULT_BASE / 'Co-SLAM/scripts/Validation/validate_pruning8.py')
SEEDS = (0, 1, 2)
EXPECTED_SCENES = (
    ('Replica', 'office0', 2000), ('Synthetic', 'tg', 395),
    ('Tum', 'fr1_desk', 592), ('ScanNet', 'scene0106', 2324),
    ('Replica', 'room0', 2000), ('Synthetic', 'ck', 1211),
    ('Tum', 'fr3_office', 2515), ('ScanNet', 'scene0000', 5578),
)
EXPECTED_METHOD = {
    'enabled': True, 'policy': 'ray_cohort',
    'stages': ['Tracking', 'Bundle_Adjustment'],
    'pixel_strategy': 'random', 'tracking_pose_average': 0,
    'tracking_pose_selection': 'dense_anchors', 'tracking_anchor_every': 3,
    'tracking_ray_fraction': 0.5, 'ba_ray_fraction': 1.0,
    'ray_strategy': 'systematic', 'ray_estimator': 'self_normalized',
    'cohort_count': 4, 'recent_keyframes': 2,
    'cohort_transition_keyframes': 2, 'tv_fraction': 0.125,
    'cohort_weighting': 'population', 'cohort_tracking': 'ray_cycle',
    'tracking_cycle_groups': 2, 'cohort_keep_empty_pose_blocks': False,
}
# These values remain part of the original archive-integrity check. They do
# not control the selected ray_cycle/cohort path and the cleaned CLI omits them.
ARCHIVE_ONLY_FIELDS = frozenset({
    'tracking_ray_fraction', 'ba_ray_fraction', 'ray_strategy',
    'ray_estimator', 'tracking_pose_average',
})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_verifier(path):
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('_independent_pruning8_report_audit', path)
    require(spec is not None and spec.loader is not None, 'Cannot load independent verifier')
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    require(tuple(tuple(s) for s in module.SCENES) == EXPECTED_SCENES,
            'Verifier does not cover the fixed eight-scene study')
    return module


def checked_result(result):
    require(result.get('qualified') is True and result.get('complete') is True,
            'Study is incomplete or unqualified; no report will be written')
    require(result.get('passed') is True and result.get('require_all') is True,
            'Independent acceptance must require every scene')
    require(tuple(result.get('seeds', ())) == SEEDS and not result.get('missing_runs'),
            'Exactly the complete paired seeds 0, 1 and 2 are required')
    require(tuple(tuple(s) for s in result['manifest']) == EXPECTED_SCENES,
            'Scene manifest differs from this report specification')
    require(len(result['cases']) == 24 and len(result['scenes']) == 8,
            'Report requires 24 pairs and eight complete scenes')
    expected = {(dataset, scene, seed) for dataset, scene, _ in EXPECTED_SCENES for seed in SEEDS}
    actual = [(r['dataset'], r['scene'], r['seed']) for r in result['cases']]
    require(len(set(actual)) == 24 and set(actual) == expected, 'Missing or duplicate paired cases')
    for key, value in EXPECTED_METHOD.items():
        if key == 'tv_fraction':
            continue
        require(result['method_config'].get(key) == value,
                f'This writer describes V8 only; sampling setting differs: {key}')
    rho = float(result['method_config']['tv_fraction'])
    require(math.isfinite(rho) and 0 < rho < 1, 'A real feature-sampling budget is required')
    require(len(result['source_hashes']) == 17,
            'Expected the original 17-file runtime archive, not a cleaned-source replacement')
    require(result['source_files_verified'] == 48 * 17,
            'Expected independent verification of all 48 original runtime archives')
    require('RTX 4070' in result['hardware']['device'], 'Unexpected GPU for this protocol')
    for scene in result['scenes']:
        require(scene['complete'] and scene['passed'], 'Every scene must independently pass')
        rows = group_cases(result, scene['dataset'], scene['scene'])
        for key in ('dense_ate_cm', 'selected_ate_cm', 'dense_tracking_ms',
                    'selected_tracking_ms', 'dense_ba_ms', 'selected_ba_ms'):
            values = [r[key] for r in rows]
            require(all(math.isfinite(v) and v > 0 for v in values), f'Invalid metric: {key}')
            require(math.isclose(statistics.mean(values), scene[key], rel_tol=1e-10, abs_tol=1e-10),
                    f'Case and scene means disagree: {key}')
        require(scene['ate_change_pct'] <= 15 and scene['tracking_time_saved_pct'] >= 5
                and scene['ba_time_saved_pct'] >= 25, 'A scene does not meet the acceptance floors')
    return result


def group_cases(result, dataset, scene):
    return sorted((r for r in result['cases'] if (r['dataset'], r['scene']) == (dataset, scene)),
                  key=lambda r: r['seed'])


def mean_sd(rows, key, digits=2):
    values = [r[key] for r in rows]
    return f'{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}'


def reduction(rows, dense_key, selected_key):
    return 100 * (1 - sum(r[selected_key] for r in rows) / sum(r[dense_key] for r in rows))


def all_call_metrics(root, candidate, dataset, scene):
    """Describe every recorded stage call, including the timing warmup."""
    result = []
    for seed in SEEDS:
        row = {}
        for label, family in (('dense', 'controls'), ('selected', candidate)):
            path = root / family / f'{dataset}_{scene}_seed{seed}/frames.csv'
            with path.open(newline='') as handle:
                frames = list(csv.DictReader(handle))
            row[label + '_tracking_ms'] = statistics.mean(float(r['tracking_wall_ms']) for r in frames)
            row[label + '_ba_ms'] = statistics.mean(float(r['ba_wall_ms']) for r in frames if int(r['has_ba']))
        result.append(row)
    return result


def label(dataset, scene):
    return f'{"TUM" if dataset == "Tum" else dataset} / {scene}'


def markdown_link(title, path, output):
    relative = os.path.relpath(Path(path).resolve(), output.parent)
    return f'[{title}]({quote(relative, safe="/._-")})'


def checked_configs(root, candidate):
    configs = {}
    for dataset, scene, _ in EXPECTED_SCENES:
        config = read_json(root / candidate / f'{dataset}_{scene}_seed0/config.json')
        mapping, tracking = config['mapping'], config['tracking']
        expected_first = 1000 if dataset == 'Tum' else 500 if dataset == 'ScanNet' else 200
        expected_ba = 20 if dataset == 'Tum' else 10
        expected_hash = {('ScanNet', 'scene0106'): 14, ('ScanNet', 'scene0000'): 16}.get((dataset, scene), 19)
        require(mapping['first_iters'] == expected_first, 'Native initialization length differs')
        require(mapping['iters'] == expected_ba and tracking['iter'] == 10, 'Native iteration count differs')
        require(mapping['map_every'] == mapping['keyframe_every'] == 5, 'Native frame schedule differs')
        require(mapping['sample'] == 2048 and tracking['sample'] == 1024, 'Native ray budget differs')
        require(tracking['best'] is (dataset != 'Tum'), 'Native tracking pose rule differs')
        require(tracking['wait_iters'] >= tracking['iter'], 'Early stopping could change the ten-update schedule')
        require(config['grid']['hash_size'] == expected_hash, 'Native inherited hash capacity differs')
        require(config['training']['perturb'] == 1, 'This report assumes native fresh depth jitter')
        configs[(dataset, scene)] = config
    return configs


def cleanup_note(path, result, output):
    if path is None:
        return ('No clean-source equivalence record was supplied to this report writer. '
                'All performance measurements refer to the original archived runtime source.')
    metadata = read_json(path)
    frozen, clean = metadata['frozen_source_hashes'], metadata['clean_source_hashes']
    require(frozen == result['source_hashes'] and len(frozen) == 17,
            'Cleanup proof does not identify the measured 17-file source archive')
    require(len(clean) == 15, 'Expected a separate 15-file cleanup manifest')
    clean_root = Path(metadata['clean_source_root'])
    for name, digest in clean.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid cleanup source path')
        require(hashlib.sha256((clean_root / relative).read_bytes()).hexdigest() == digest,
                f'Clean source differs from its supplied supplemental manifest: {name}')
    return (f'A separate {markdown_link("cleanup record", path, output)} identifies the measured '
            '17-file source archive and a 15-file cleaned runtime. The writer checked the original '
            'manifest match and the cleaned files against their supplied hashes. Any CPU equivalence '
            'claims are limited to the scope and limitations recorded there. This supplemental record '
            'does not replace the original GPU archives, constitute a new ATE/timing experiment, or '
            'establish bitwise-equivalent CUDA optimization trajectories.')


def regression_note(path, output):
    if path is None:
        return ('No regression-log metadata was supplied to this writer. The independent saved-run '
                'audit below is distinct from CPU regression testing.')
    text = Path(path).read_text(encoding='utf-8')
    counts = re.findall(r'^Ran (\d+) tests? in [^\n]+$', text, flags=re.MULTILINE)
    outcomes = re.findall(r'^(OK(?: \([^\n]*\))?|FAILED[^\n]*)$', text, flags=re.MULTILINE)
    require(counts and outcomes and outcomes[-1].startswith('OK'),
            'Supplied regression log lacks a completed successful unittest summary')
    skipped = re.search(r'skipped=(\d+)', outcomes[-1])
    suffix = f', including {skipped.group(1)} skipped' if skipped else ''
    return (f'The supplied {markdown_link("regression log", path, output)} records {counts[-1]} '
            f'tests with status `{outcomes[-1]}`{suffix}. These CPU checks complement the '
            'saved-run audit; they do not establish GPU trajectory equivalence or mesh quality.')


def build_report(result, root, output, configs, verifier_path, artifact_prefix=None,
                 clean_proof=None, regression_log=None):
    checked_result(result)
    candidate = result['candidate']
    root_href = (artifact_prefix.rstrip('/') if artifact_prefix is not None
                 else os.path.relpath(root, output.parent))
    link = lambda title, relative: f'[{title}]({quote(root_href + "/" + relative, safe="/._-")})'
    nominal_tracking = sum(s['tracking_target'] for s in result['scenes'])
    nominal_ba = sum(s['ba_target'] for s in result['scenes'])
    hardware = result['hardware']
    rho = result['method_config']['tv_fraction']
    lines = [
        '# Joint ray and historical-constraint pruning across eight scenes', '',
        '**Qualified method: complementary tracking rays with full-ray checkpoints, graduated '
        'historical BA constraints, and sampled quadratic feature smoothness.** Every scene passes '
        'the agreed three-seed mean criteria: ATE increase at most 15%, complete tracking wall-time '
        'reduction at least 5%, and complete BA reduction at least 25%. These are the permitted '
        'floors around nominal targets of 10% tracking and 30% BA reduction.', '',
        f'The nominal tracking target is reached on **{nominal_tracking}/8 scenes** and the nominal '
        f'BA target on **{nominal_ba}/8 scenes**. Passing the tolerance floors is not described as '
        'meeting a nominal target where the measured value falls below it. All 24 paired seeds '
        '(48 complete runs) are included; no failing seed is discarded.', '',
        '## Criteria and protocol', '',
        'Acceptance uses the ratio of arithmetic scene means over paired seeds 0, 1 and 2, not '
        'the mean of per-seed percentage changes and not a per-seed accuracy limit. ATE change '
        'is `100 × (candidate mean / dense mean − 1)`; a negative change means improvement. '
        'Time reduction is `100 × (1 − candidate mean / dense mean)`. The ±5% speed allowance '
        'is interpreted as five percentage points: the tracking/BA floors are 5%/25%. A '
        '30% wall-time reduction corresponds to a speed factor of approximately 1.43.', '',
        'All sequences are complete. Scene-specific inherited initialization lengths and hash '
        'capacities are retained; they are not homogenized to obtain a speedup. The table reports '
        '`log2_hashmap_size`, not the number of populated hash entries.', '',
        '| Scene | Frames | Dense initialization iterations | Tracking / BA iterations | Native hash exponent | Native best-pose setting |',
        '| --- | ---: | ---: | ---: | ---: | --- |',
    ]
    for dataset, scene, frames in EXPECTED_SCENES:
        config = configs[(dataset, scene)]
        lines.append(f'| {label(dataset, scene)} | {frames} | {config["mapping"]["first_iters"]} | '
                     f'{config["tracking"]["iter"]} / {config["mapping"]["iters"]} | '
                     f'{config["grid"]["hash_size"]} | {config["tracking"]["best"]} |')
    lines += ['',
        f'Runs execute serially on **{hardware["device"]}** in conda **`coslam`**, using '
        f'Python {hardware["python_version"]} and PyTorch {hardware["torch_version"]}. '
        'Dense and candidate runs use identical frozen runtime source; only sampling configuration '
        'and output path differ. Both retain the native representation, optimizer settings, '
        'initialization, iteration counts, ray-database draws and frame cadence. The shared grid '
        'uses `CoherentPrime` with Morton sorting disabled; execution experiment variants remain '
        '`baseline`. The pruning method includes the reduced pose blocks induced by removed '
        'constraints, so its measured gain is not labeled neural-query-only.', '',
        'The benchmark resets Python, NumPy and Torch random streams by run seed, frame and phase '
        'using `seed × 1000003 + frame × 17 + phase`, with phases 1/2/3 for tracking/BA/keyframe '
        'storage. The reset occurs before the timed invocation in both conditions. Private sampling '
        'generators do not consume these shared streams. This prevents one stage’s different RNG '
        'consumption from shifting a later stage’s starting stream; it does not guarantee identical '
        'within-stage jitter draws or bitwise reproducibility of CUDA hash-grid training.', '',
        'CUDA-synchronized wall timing encloses each complete tracking or BA invocation, including '
        'selection, preprocessing, transfers, rendering, regularization, losses, backward, optimizer '
        'and pose commits. Acceptance stage means omit frames 0–19, following the existing '
        '20-frame warmup protocol explicitly confirmed by the user. ATE and query counts cover the full '
        'sequence, including all full-ray checkpoints. Total SLAM wall time includes dense '
        'initialization and data loading during SLAM, but excludes model/dataset construction, '
        'post-run evaluation and mesh export. Frame samples are correlated; the reported SD is '
        'the sample SD across three seed-level means, not the frame-level SD.', '',
        '## Accuracy', '',
        '| Scene | Dense ATE (cm), mean ± SD | Candidate ATE (cm), mean ± SD | Ratio change | Limit |',
        '| --- | ---: | ---: | ---: | --- |',
    ]
    for scene in result['scenes']:
        rows = group_cases(result, scene['dataset'], scene['scene'])
        lines.append(f'| {label(scene["dataset"], scene["scene"])} | {mean_sd(rows, "dense_ate_cm", 4)} | '
                     f'{mean_sd(rows, "selected_ate_cm", 4)} | {scene["ate_change_pct"]:+.2f}% | ≤15%: pass |')
    lines += ['',
        'ATE is the rigidly aligned translational RMSE without scale fitting. Alignment is '
        'recomputed independently from each saved trajectory, including the native dataset scale. '
        'Direct ATE is the acceptance metric; relative-trajectory ATE is independently checked '
        'for artifact integrity. Three-seed means can pass while individual seeds degrade by more '
        'than 15%; all such cases remain visible below. Passing a mean gate does not prove per-run '
        'stability or statistical significance.', '',
        '## Secondary corrected-trajectory ATE', '',
        'Co-SLAM also propagates final keyframe pose corrections to non-keyframes through '
        '`convert_relative_pose`. This secondary ATE is shown for completeness; the '
        'predefined acceptance gate continues to use direct-trajectory ATE.', '',
        '| Scene | Dense corrected ATE (cm), mean ± SD | Candidate corrected ATE (cm), mean ± SD | Change of means |',
        '| --- | ---: | ---: | ---: |',
    ]
    for scene in result['scenes']:
        rows = group_cases(result, scene['dataset'], scene['scene'])
        dense_relative = statistics.mean(r['dense_relative_ate_cm'] for r in rows)
        selected_relative = statistics.mean(r['selected_relative_ate_cm'] for r in rows)
        change = 100 * (selected_relative / dense_relative - 1)
        lines.append(f'| {label(scene["dataset"], scene["scene"])} | '
                     f'{mean_sd(rows, "dense_relative_ate_cm", 4)} | '
                     f'{mean_sd(rows, "selected_relative_ate_cm", 4)} | {change:+.2f}% |')
    lines += ['',
        '## Complete wall time', '',
        '| Scene | Tracking dense → candidate (ms), mean ± SD | Tracking reduction | 10% target | BA dense → candidate (ms), mean ± SD | BA reduction | 30% target | Total SLAM dense → candidate (s), mean ± SD |',
        '| --- | ---: | ---: | --- | ---: | ---: | --- | ---: |',
    ]
    for scene in result['scenes']:
        rows = group_cases(result, scene['dataset'], scene['scene'])
        lines.append(f'| {label(scene["dataset"], scene["scene"])} | '
                     f'{mean_sd(rows, "dense_tracking_ms")} → {mean_sd(rows, "selected_tracking_ms")} | '
                     f'{scene["tracking_time_saved_pct"]:.2f}% | {"met" if scene["tracking_target"] else "floor only"} | '
                     f'{mean_sd(rows, "dense_ba_ms")} → {mean_sd(rows, "selected_ba_ms")} | '
                     f'{scene["ba_time_saved_pct"]:.2f}% | {"met" if scene["ba_target"] else "floor only"} | '
                     f'{mean_sd(rows, "dense_run_s")} → {mean_sd(rows, "selected_run_s")} |')
    lines += ['',
        'Every row meets the 5% tracking and 25% BA acceptance floors. These measured complete '
        'invocation reductions include any sampling overhead; query-count reduction is not used '
        'as a substitute for wall time. Total SLAM time is descriptive and has no separate '
        'acceptance threshold.', '',
        '## Stage timings including the warmup calls', '',
        'The following secondary table includes every call; acceptance above uses the existing '
        '20-frame warmup exclusion confirmed by the user.', '',
        '| Scene | Tracking dense → candidate (ms), mean ± SD | Reduction | BA dense → candidate (ms), mean ± SD | Reduction |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for scene in result['scenes']:
        rows = all_call_metrics(root, candidate, scene['dataset'], scene['scene'])
        tracking_change = reduction(rows, 'dense_tracking_ms', 'selected_tracking_ms')
        ba_change = reduction(rows, 'dense_ba_ms', 'selected_ba_ms')
        lines.append(f'| {label(scene["dataset"], scene["scene"])} | '
                     f'{mean_sd(rows, "dense_tracking_ms")} → {mean_sd(rows, "selected_tracking_ms")} | '
                     f'{tracking_change:.2f}% | '
                     f'{mean_sd(rows, "dense_ba_ms")} → {mean_sd(rows, "selected_ba_ms")} | '
                     f'{ba_change:.2f}% |')
    lines += ['',
        '## Actual pruning and active pose blocks', '',
        '| Scene | Tracking joint RGB/SDF queries pruned | BA joint RGB/SDF queries pruned | BA feature-only queries pruned | BA pose blocks omitted |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for dataset, scene, _ in EXPECTED_SCENES:
        rows = group_cases(result, dataset, scene)
        values = [reduction(rows, 'dense_tracking_points', 'selected_tracking_points'),
                  reduction(rows, 'dense_ba_points', 'selected_ba_points'),
                  reduction(rows, 'dense_feature_points', 'selected_feature_points'),
                  reduction(rows, 'dense_pose_blocks', 'selected_pose_blocks')]
        lines.append(f'| {label(dataset, scene)} | {values[0]:.2f}% | {values[1]:.2f}% | '
                     f'{values[2]:.4f}% | {values[3]:.2f}% |')
    lines += ['',
        'Percentages are computed from summed actual/reference counts across the three complete '
        'runs, including all checkpoint forwards. Feature endpoints and joint RGB/SDF points have '
        'different computational costs and are reported separately. Pose-block counts sum the '
        'number of optimized poses over BA calls, including the current pose when enabled and '
        'excluding the fixed first pose; they are neither neural tokens nor counts of individual '
        'Adam updates. Initialization remains dense and is outside these tracking/BA pruning '
        'percentages. None of these counts establishes how much wall-time benefit comes from '
        'each component.', '',
        '## Why the wall-time reduction varies across scenes', '',
        'The tracking query reduction is exactly 30% in every scene, while whole-stage '
        'savings vary because the native along-ray populations differ (43, 85 or 117 points), '
        'and pose, optimizer, transfer and launch costs remain. BA also inherits different '
        'iteration counts, feature-grid sizes, current-ray minimums and encoding capacities. '
        'In particular, ScanNet retains its native hash capacities rather than the larger '
        'capacity used by the other datasets. These settings are identical within each '
        'dense/candidate pair, but prevent a cross-dataset speed difference from being '
        'attributed to frame count alone.', '',
        'Sequence length changes the population of historical constraints. Early BA calls '
        'have few mature groups and a larger native current-frame ray budget; later calls '
        'offer more pruning opportunities. Historical pose work also grows with the number '
        'of keyframes. The method omits selected-out pose blocks, while still stacking the '
        'full pose history and performing the original mixed-ray sampling, transfers and '
        'ray transforms. Those remaining costs limit the conversion of query reductions '
        'into wall-time savings. Dense initialization is excluded from the stage means '
        'but included in total SLAM time.', '',
        'This explains the relevant cost structure from the code and measured budgets. '
        'It is not a measured decomposition of the gain: matched component ablations or '
        'profiles would be needed to assign causal fractions to neural work, regularization '
        'and pose work.', '',
        '## Every paired seed', '',
        '| Scene | Seed | Dense ATE (cm) | Candidate ATE (cm) | ATE change | Tracking dense → candidate (ms) | Tracking reduction | BA dense → candidate (ms) | BA reduction |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for dataset, scene, _ in EXPECTED_SCENES:
        for row in group_cases(result, dataset, scene):
            lines.append(f'| {label(dataset, scene)} | {row["seed"]} | {row["dense_ate_cm"]:.4f} | '
                         f'{row["selected_ate_cm"]:.4f} | {row["ate_change_pct"]:+.2f}% | '
                         f'{row["dense_tracking_ms"]:.2f} → {row["selected_tracking_ms"]:.2f} | '
                         f'{row["tracking_time_saved_pct"]:.2f}% | '
                         f'{row["dense_ba_ms"]:.2f} → {row["selected_ba_ms"]:.2f} | '
                         f'{row["ba_time_saved_pct"]:.2f}% |')
    lines += ['', '## Method and mathematical scope', '',
        '**Difference from the original depth/bucket sampling.** In the original '
        '`tracking_depth_spatial` pruned path, candidate pixels must fit a low depth-derived '
        'sample-count ceiling while satisfying spatial coverage quotas. Legacy `ba_bucket` '
        'sorts a larger ray pool by that count into depth-homogeneous update batches. Their '
        'depth-limited renderer reduces the uniform point count and redistributes those '
        'points from near to measured depth, retaining the separate depth-centered points. '
        'This changes longitudinal sample locations and the lattice on which native geometry '
        'loss masks, means and balancing are calculated; it is not simply deletion of '
        'unchanged original sample nodes.', '',
        'The present method starts with native ray draws and prunes whole observations: '
        'complementary tracking halves recover every original ray over time, and BA '
        'selects historical keyframe constraints by age. It preserves native longitudinal '
        'sampling for each retained ray. In BA, period weights 1/2/4 and original population '
        'normalizers compensate the fixed-state selection frequencies. Tracking half-updates '
        'still use selected-batch ratio losses; full-population normalization is not claimed '
        'for them. These code-level differences motivate the new hypothesis but do not '
        'isolate the cause of the original quality degradation or prove which new component '
        'improves accuracy.', '',
        '**Whole-ray queries.** Every retained ray uses the complete native Co-SLAM '
        'near-to-far uniform and depth-centered sample counts, native jitter law, neural '
        'RGB/SDF evaluation and renderer. Tracking draws jitter for the selected batch, so '
        'its realized points need not match a corresponding dense-control draw. BA constructs '
        'the original full-batch lattice before selecting rows. There is no interpolated neural '
        'output, cached feature prediction, auxiliary importance network or omitted interval '
        'within a retained ray. The scientific approximation is which rays and historical '
        'constraints participate in an update, together with sampled feature regularization.', '',
        '**Complementary tracking.** Start from the native fixed batch of 1024 randomly selected '
        'rays. Order it by coarse 8×8 image-direction cells, depth and original index, then '
        'interleave two complementary groups. The `ray_cycle` path always performs this '
        'ordering; the archived `ray_strategy` field does not select it. A private '
        'seed-dependent initial phase determines '
        'which half is omitted first. Iterations 0, 3, 6 and 9 use all rays; the other six updates '
        'alternate halves. Each ray therefore appears four times at full checkpoints and three '
        'times at half updates: exactly seven of ten updates, giving 30% fewer joint field '
        'queries without changing the ten Adam updates. Spatial/depth ordering is a coverage '
        'heuristic, not a guaranteed pose-information coreset.', '',
        'For native `tracking.best=True`, only the full-ray checkpoint losses can select the '
        'saved pose. If θᵢ denotes the pose before optimizer step i, the output is the eligible '
        'θᵢ with the smallest measured checkpoint loss. For native `best=False` (TUM), the '
        'original last pre-step pose θ₉ is saved; the final update is still executed. Restricting '
        'selection avoids comparing objectives from different ray populations. Checkpoints '
        'still redraw native depth jitter, so they are noisy full-ray evaluations, not identical '
        'deterministic objective evaluations. Half-update depth normalization and geometry '
        'balancing use the native selected-batch ratios. Equal ray exposure therefore does not '
        'make tracking losses, gradients or the optimizer trajectory unbiased or identical to '
        'dense tracking. The saved diagnostics prove query totals and the structural schedule; '
        'they do not record enough data to independently replay each winning pose.', '',
        '**Graduated historical BA constraints.** Retain the native mixed ray draw: 2048 '
        'historical rays plus the native current-frame budget. Every current-frame ray, every '
        'sampled ray from the first keyframe, and every sampled ray from the two newest '
        'historical keyframes remains. For historical index g with M available keyframes, '
        'age is a=M−g. Ages 3–4 use period p=2; ages at least 5 use p=4. At BA call b, '
        'a non-anchor group is eligible when `g % p == (b + seed) % p`. All native rays '
        'belonging to eligible groups are retained throughout that call, while the native '
        'mixed ray population is redrawn at every BA iteration. Retained ray losses receive '
        'weight p (1 for anchors, 2 for transition groups, 4 for mature groups), with the '
        'original full-batch denominators, valid-depth count and geometry balancing. '
        'The cycle bounds eligibility gaps; it does not guarantee an actual observation '
        'from every eligible keyframe, because a finite native ray draw can miss a group.', '',
        'For fixed field/poses θ, reference ray batch, jitter, group membership and age '
        'classification, let A be anchor rays, Rg the rays from group g, and D the native '
        'denominator of an additive loss term (with its native balancing coefficient held '
        'fixed). For phase s,', '',
        '```text\nL_s(θ) = [Σ_(r∈A) ℓ_r(θ) + Σ_g p_g I_s(g) Σ_(r∈Rg) ℓ_r(θ)] / D\n'
        '(1/4) Σ_(s=0)^3 p_g I_s(g) = 1\n'
        '(1/4) Σ_(s=0)^3 L_s(θ) = L_dense(θ).\n```', '',
        'The same fixed-state identity holds for gradients where the losses are differentiable. '
        'Online BA changes parameters, ray draws and age classes between phases, so this '
        'identity does not establish unbiased online gradients, an unbiased Adam trajectory or '
        'equivalent reconstruction. Only historical poses associated with retained groups '
        'and the current pose when natively enabled enter optimization; the first pose stays '
        'fixed. With fresh pose-Adam state for each call and no cross-pose loss term, removed '
        'groups have zero pose gradients. Avoiding these empty blocks follows from constraint '
        'pruning, rather than being presented as an independent novel pose algorithm. The '
        'native early-BA pose rule and pose accumulation schedule remain in effect.', '',
        '**Feature regularization.** The native feature smoothness is a quadratic neighbor '
        f'difference energy, despite the shorthand TV. Use the existing estimator with ρ={rho}: '
        'sample directed neighboring edges independently with replacement, equally across '
        'three axes, and evaluate both endpoints freshly. For native `smooth_pts=s`, set '
        'n=s−1, E=(n−1)n² edges per axis and K=max(1, floor(ρn³/6)). With the native random '
        'offset, shared jitter and voxel spacing,', '',
        '```text\nL_hat = E / (K s³) × Σ_axis Σ_(k=1)^K ||f(x_k + v e_axis) − f(x_k)||².\n```', '',
        'This uses 6K endpoint feature queries per BA update. Conditional on a fixed field '
        'and native offset/jitter, it estimates the native energy and its parameter gradient '
        'without bias. Duplicate edges/endpoints are allowed and counted as queries. The '
        'original s³ denominator is preserved even though the lattice has n³ vertices. '
        'This is an existing Monte Carlo regularization component, not a new claim introduced '
        'by the present BA schedule. Fixed-field unbiasedness does not certify meshes or '
        'unchanged training.', '',
        '## Attribution, prior art and limitations', '',
        'These runs measure the combined method. Without matched ablations, they do not '
        'establish that spatial ordering, complementary coverage, checkpoint pose selection, '
        'graduated rehearsal, loss reweighting, omitted pose blocks or sampled regularization '
        'individually caused the observed accuracy or speed. No neural-only speed attribution '
        'is made.', '',
        'The transition band changes both revisit eligibility and loss weights. With the '
        'native synchronized keyframe/BA cadence and these phase offsets, seeds 0 and 1 '
        'already selected the relevant age-3/4 groups under the period-4 rule; for those '
        'groups the transition changes weight 4→2 without extra revisits. Seed 2 also gains '
        'an age-4 revisit. Thus seed-dependent improvements cannot uniformly be attributed '
        'to additional observations. A 2×2 control of age-3/4 eligibility period (4 versus 2) '
        'and retained weight (4 versus 2), with all other settings fixed, would separate '
        'these mechanisms. Independent phase-offset controls would also test modulo '
        'alignment with age. These are required attribution experiments, not results '
        'claimed by the present comparison.', '',
        'Additional matched controls would compare fixed versus complementary tracking '
        'subsets at the same query budget, all-iteration versus checkpoint-only pose '
        'selection, dense tracking with the same checkpoint selection rule, matched-budget '
        'uniform old-ray pruning versus grouped pruning, cyclic versus randomized cohort '
        'phases, and retained versus omitted empty pose blocks with identical observations. '
        'The feature sampling budget must be held fixed when isolating those effects.', '',
        '[EC-SLAM](https://arxiv.org/html/2404.13346v2) combines feature points with grid-uniform '
        'pixel sampling and geometrically constrained keyframe selection. The present method '
        'uses complementary subsets of the native random batch and age-dependent historical '
        'coverage; it does not reproduce EC-SLAM’s feature detector or keyframe scoring. '
        'Grid coverage and selective BA are established ideas.', '',
        '[HS-SLAM (2025), §III-C](https://arxiv.org/html/2503.21778v1) prioritizes historical '
        'observations using loss thresholds and top-loss keyframes. The present age schedule '
        'avoids a neural importance-scoring pass and instead supplies prescribed revisit '
        'coverage. [HL-SLAM (2026)](https://www.sciencedirect.com/science/article/pii/S107731422600175X) '
        'describes temporal SDF-variance-driven ray sampling for global neural BA in its '
        'publisher abstract. Its full implementation was unavailable in this audit, so '
        'comprehensive implementation-level overlap has not been ruled out.', '',
        '[NeRF Director (CVPR 2024)](https://arxiv.org/abs/2406.08839) supports the broader '
        'motivation of selecting observations for coverage without neural error scoring. '
        '[Soft Mining (CVPR 2024)](https://arxiv.org/html/2312.00075v1) supplies prior art '
        'for neural-field sampling with probability-based loss correction. '
        '[Co-SLAM (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Co-SLAM_Joint_Coordinate_and_Sparse_Parametric_Encodings_for_Neural_Real-Time_CVPR_2023_paper.pdf) '
        'provides the native neural field, renderer and feature-smoothness objective. '
        'These relationships motivate a scientific sampling hypothesis; neither a '
        'particular combination nor the present performance tables prove publication novelty.', '',
        'Constraint-subproblem selection also has established classical BA precedents. '
        '[Good Graph to Optimize](https://arxiv.org/abs/2008.10123) selects smaller local BA '
        'graphs using conditioning and a computational budget. '
        '[Stochastic Bundle Adjustment (ECCV 2020)](https://arxiv.org/abs/2008.00446) '
        'approximately decomposes the reduced camera system inside Levenberg–Marquardt '
        'through visibility-graph clustering and relaxed cross-cluster constraints. The '
        'present method instead omits age-scheduled ray groups and their empty pose blocks '
        'while retaining a shared neural field and Adam optimization. It does not implement '
        'those graph-selection or linear-system algorithms, and their conditioning or '
        'convergence guarantees do not transfer to this method.', '',
        '**This is a small development/validation suite, not an independent held-out benchmark.** '
        'Earlier workspace experiments informed the method; one common configuration was '
        'then kept fixed across these eight scenes, without scene-specific retuning. '
        'Three seeds do not establish broad robustness, '
        'statistical significance or generalization to other sequences/devices. Only ATE '
        'and measured compute/count metrics were evaluated here. Mesh accuracy, completion, '
        'surface detail and unobserved-space behavior remain untested; low ATE degradation '
        'cannot certify reconstruction quality.', '',
        '## Independent verification and artifacts', '',
        f'The report writer called the independent verifier afresh with all eight scenes, '
        f'`seeds=(0,1,2)` and `require_all=True`, and required `qualified is True` before '
        f'creating this file. It did not trust a stored acceptance verdict. The audit checked '
        f'{result["source_files_verified"]} file hashes across 48 run archives; each original '
        'runtime manifest contains the same 17 files. Dense/candidate non-sampling '
        'configurations, native inherited hash capacities, dense initialization and iteration '
        'counts were checked. ATE was recomputed from saved trajectories and complete '
        'stage timings from frame records. BA group draws, age-category counts, weights '
        'and pose-block schedules were replayed independently; field and feature query '
        'totals reconcile with diagnostics and checkpoint schedules.', '',
        'The audit does not replay the actual per-ray tracking identity cycle from saved '
        'records, each point mask, the winning tracking pose, or unchanged inactive pose '
        'values throughout training. The seven-appearance statement is an independent '
        'structural schedule proof combined with verified counts, not a per-ray trace replay. '
        'Archived runtime identity and successful CPU checks do not imply bitwise '
        'deterministic GPU optimization.', '',
        regression_note(regression_log, output), '',
        cleanup_note(clean_proof, result, output), '',
        f'Artifacts: {link("dense controls", "controls/")}, '
        f'{link("candidate runs", candidate + "/")}, '
        f'{markdown_link("independent verifier", verifier_path, output)}. '
        'Every run contains its inherited configuration, summary, saved trajectories, '
        'complete frame timings, query diagnostics, original source and source hashes. '
        'Candidate runs also contain cohort-pose and sampled-smoothness diagnostics.', '',
        '## Reproduction and mesh follow-up', '',
        'Use conda `coslam` and the archived runtime source, native scene configuration '
        'and saved sampling settings. Repeat the complete paired-seed protocol rather '
        'than comparing one favorable seed. The candidate’s active settings are:', '',
        '```json', json.dumps({key: result['method_config'][key] for key in EXPECTED_METHOD
                              if key not in ARCHIVE_ONLY_FIELDS}, indent=2),
        '```', '',
        'The original run archives retain their complete configuration, including '
        '`tracking_ray_fraction`, `ba_ray_fraction`, `ray_strategy`, `ray_estimator` '
        'and `tracking_pose_average=0`. Those archived values are inert for the selected '
        'cyclic tracking/cohort BA path and are omitted from the cleaned reproduction CLI '
        'and active-settings JSON above. They are still checked when verifying the '
        'original source/configuration provenance.', '',
        '**Mesh reproduction requires new SLAM runs.** The measured benchmark intentionally '
        'saved trajectories, timings, query counts, diagnostics and source, but disabled '
        'trained-map checkpoints. The exact maps that produced the original table cannot '
        'be loaded later for mesh extraction. `--export-mesh` reruns the paired frame/phase '
        'protocol and extracts the final mesh after recording timing and ATE; extraction '
        'is outside those timing measurements. CUDA hash-grid updates are not bitwise '
        'deterministic, so every mesh rerun must retain and report its own ATE alongside '
        'the mesh metrics rather than inherit the original table’s exact ATE values.', '',
        'From the cleaned reproduction checkout, activate `coslam` and use a fresh output '
        'directory. One candidate scene/seed, using only active cohort options, is:', '',
        '```bash\nconda activate coslam\n'
        'python scripts/Validation/benchmark_ba_accuracy.py \\\n'
        '  --dataset Replica --scene office0 --seed 0 --frames 0 \\\n'
        '  --tracking baseline --variant baseline --pruning ray_cohort \\\n'
        '  --cohort-count 4 --recent-keyframes 2 --cohort-transition-keyframes 2 \\\n'
        '  --cohort-weighting population --cohort-tracking ray_cycle \\\n'
        '  --tracking-cycle-count 2 --tracking-anchor-every 3 \\\n'
        '  --tracking-pose-selection dense_anchors --tv-fraction 0.125 \\\n'
        '  --output /tmp/pruning8_mesh_single/selected/Replica_office0_seed0 \\\n'
        '  --export-mesh\n```', '',
        'Its paired dense control must use the same scene and seed, `--tracking baseline '
        '--variant baseline --pruning dense`, native pose selection, a different fresh '
        'output directory, and `--export-mesh`. The full runner performs the eight-scene, '
        'three-seed paired protocol:', '',
        '```bash\npython scripts/Validation/run_pruning8_study.py \\\n'
        '  --output /tmp/pruning8_mesh_reproduction --export-mesh\n```', '',
        'The cleaned reproduction source is distinct from the frozen runtime used for '
        'the original measurements; its supplemental CPU equivalence record, when supplied, '
        'is documented above. Fresh mesh runs archive their own source and measurements.', '',
        'This report writer performs verification and reporting only; it does not run GPU '
        'experiments or extract meshes. The next scientific validation is paired mesh '
        'extraction and reconstruction metrics under the same retained method and seeds. '
        'Until those measurements exist, this result supports the stated trajectory/'
        'compute tradeoff on these development scenes only.', '',
    ]
    for filename in ('protocol.json', 'environment.json'):
        candidates = (root / filename, root.parent / filename)
        existing = next((path for path in candidates if path.is_file()), None)
        if existing is not None:
            lines.insert(-1, f'Study metadata: {markdown_link(filename, existing, output)}.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Run root containing controls and candidate directories')
    parser.add_argument('--candidate', default='cohort4_transition2_anchor')
    parser.add_argument('--output', type=Path, required=True, help='Markdown file below /tmp only')
    parser.add_argument('--verifier', type=Path, default=DEFAULT_VERIFIER)
    parser.add_argument('--artifact-prefix', help='Optional run-root link prefix for later promotion')
    parser.add_argument('--link-base', type=Path,
                        help='Intended final Markdown path for relative links; output still must be below /tmp')
    parser.add_argument('--clean-proof', type=Path, help='Optional separate cleanup-equivalence JSON')
    parser.add_argument('--regression-log', type=Path, help='Optional completed unittest log')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing scratch report after revalidation')
    args = parser.parse_args()
    root, output = args.root.resolve(strict=True), args.output.resolve()
    require(output.is_relative_to(Path('/tmp').resolve()) and output.suffix.lower() == '.md',
            'Output must be a Markdown file whose resolved path is below /tmp')
    require(not output.exists() or args.overwrite, 'Output exists; use --overwrite to revalidate and replace it')
    verifier = load_verifier(args.verifier)
    # Critical gate: no directory creation, report file or acceptance artifact before this call succeeds.
    result = checked_result(verifier.evaluate(root, args.candidate, seeds=SEEDS, require_all=True))
    configs = checked_configs(root, args.candidate)
    link_base = args.link_base.resolve() if args.link_base is not None else output
    document = build_report(result, root, link_base, configs, args.verifier.resolve(),
                            artifact_prefix=args.artifact_prefix, clean_proof=args.clean_proof,
                            regression_log=args.regression_log)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=output.parent,
                                         prefix='.' + output.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f'Wrote independently qualified report: {output}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f'Refused to write report: {error}') from error
