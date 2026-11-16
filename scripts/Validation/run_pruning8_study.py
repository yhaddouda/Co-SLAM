#!/usr/bin/env python3
"""Run the fixed eight-scene paired study serially, then independently audit it.

Use the coslam environment and an empty scratch output directory. Only a fully
qualified study receives acceptance.json. Mesh export, if requested, occurs
after each benchmark has recorded its trajectory and timing measurements.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCENES = [('Replica', 'office0', 2000), ('Synthetic', 'tg', 395),
          ('Tum', 'fr1_desk', 592), ('ScanNet', 'scene0106', 2324),
          ('Replica', 'room0', 2000), ('Synthetic', 'ck', 1211),
          ('Tum', 'fr3_office', 2515), ('ScanNet', 'scene0000', 5578)]
METHOD = ['--pruning', 'ray_cohort', '--cohort-count', '4',
          '--recent-keyframes', '2', '--cohort-transition-keyframes', '2',
          '--cohort-weighting', 'population', '--cohort-tracking', 'ray_cycle',
          '--tracking-cycle-count', '2', '--tracking-anchor-every', '3',
          '--tracking-pose-selection', 'dense_anchors', '--tv-fraction', '.125']
PARAMETERS = dict(policy='ray_cohort', cohort_count=4, recent_keyframes=2,
                  cohort_transition_keyframes=2, cohort_weighting='population',
                  cohort_tracking='ray_cycle', tracking_cycle_groups=2,
                  tracking_anchor_every=3, tracking_pose_selection='dense_anchors',
                  tv_fraction=.125, cohort_keep_empty_pose_blocks=False)


def reusable(path, source, family, dataset, scene, frames, seed, export_mesh):
    summary_path = path / 'summary.json'
    if not summary_path.exists():
        raise RuntimeError(f'Incomplete run cannot be reused or overwritten: {path}')
    summary = json.loads(summary_path.read_text())
    for key, expected in dict(dataset=dataset, scene=scene, frames=frames, seed=seed,
                              diagnostic=False, warmup_frames=20,
                              tracking='baseline', variant='baseline').items():
        if summary.get(key) != expected:
            raise RuntimeError(f'Existing run has a different {key}: {path}')
    config = summary['sample_pruning']
    expected = PARAMETERS if family == 'selected' else dict(policy='dense', tracking_pose_selection='native')
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f'Existing run has a different sampling setting {key}: {path}')
    for filename, digest in json.loads((path / 'source_hashes.json').read_text()).items():
        current = hashlib.sha256((source / filename).read_bytes()).hexdigest()
        archived = hashlib.sha256((path / 'source' / filename).read_bytes()).hexdigest()
        if current != digest or archived != digest:
            raise RuntimeError(f'Source changed since the completed run: {filename} in {path}')
    if export_mesh and not (path / 'mesh_export.json').exists():
        raise RuntimeError(f'The completed run has no mesh export; use a new output directory: {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export-mesh', action='store_true')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    benchmark = source / 'scripts/Validation/benchmark_ba_accuracy.py'
    verifier = source / 'scripts/Validation/validate_pruning8.py'
    if not benchmark.is_file() or not verifier.is_file():
        raise RuntimeError('The source directory must contain the benchmark and independent verifier')
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError('Use an empty scratch output directory, or explicitly resume this study')
    output.mkdir(parents=True, exist_ok=True)
    plan = dict(scenes=[list(s) for s in SCENES], seeds=[0, 1, 2], source=str(source),
                method=PARAMETERS, export_mesh=args.export_mesh,
                ordering='even seeds: dense then selected; odd seeds: selected then dense',
                ate='each scene ratio of three-seed means, maximum increase 15%',
                stage_time='whole CUDA-synchronized stage; frames 0–19 omitted',
                speed_targets_pct=dict(tracking=10, ba=30),
                speed_acceptance_floors_pct=dict(tracking=5, ba=25))
    plan_path = output / 'study_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan:
            raise RuntimeError('The existing study plan differs from the requested protocol')
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    for dataset, scene, frames in SCENES:
        for seed in (0, 1, 2):
            order = ('controls', 'selected') if seed % 2 == 0 else ('selected', 'controls')
            for family in order:
                directory = output / family / f'{dataset}_{scene}_seed{seed}'
                if directory.exists():
                    if not args.resume:
                        raise RuntimeError(f'Refusing to overwrite {directory}')
                    reusable(directory, source, family, dataset, scene, frames, seed, args.export_mesh)
                    print(f'REUSE {family} {dataset}/{scene} seed={seed}', flush=True)
                    continue
                directory.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(benchmark), '--dataset', dataset, '--scene', scene,
                           '--seed', str(seed), '--frames', '0', '--warmup', '20',
                           '--tracking', 'baseline', '--variant', 'baseline', '--output', str(directory)]
                command += METHOD if family == 'selected' else ['--pruning', 'dense', '--tracking-pose-selection', 'native']
                if args.export_mesh:
                    command.append('--export-mesh')
                log_path = directory.parent / (directory.name + '.log')
                print(f'RUN {family} {dataset}/{scene} seed={seed}; log={log_path}', flush=True)
                with log_path.open('x') as stream:
                    subprocess.run(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT, check=True)
                print(f'COMPLETE {family} {dataset}/{scene} seed={seed}', flush=True)
    subprocess.run([sys.executable, str(verifier), str(output), '--candidate', 'selected', '--write'],
                   cwd=source, check=True)


if __name__ == '__main__':
    main()
