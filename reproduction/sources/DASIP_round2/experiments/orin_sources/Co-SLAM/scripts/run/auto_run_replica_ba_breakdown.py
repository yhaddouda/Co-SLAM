#!/usr/bin/env python3
"""Run matched dense BA wall passes and explanatory CUDA block passes serially.

Uses the existing BA-only harness: baseline tracking, paired frame/phase RNG,
fresh processes, fixed ray/sample/update counts and final rigid-aligned ATE.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.Validation.summarize_ba_breakdown import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', nargs='+', default=['office1', 'room1', 'office0'])
    parser.add_argument('--variants', nargs='+', choices=['baseline', 'staging_only', 'pose_only', 'baseline_fast'],
                        default=['baseline', 'baseline_fast'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--frames', type=int, default=0, help='0 = full scene; otherwise prefix length')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--passes', nargs='+', choices=['wall', 'breakdown'], default=['wall', 'breakdown'])
    parser.add_argument('--start-frame', type=int, default=100)
    parser.add_argument('--window', type=int, default=50)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--resume', action='store_true', help='Skip complete matching runs; incomplete directories remain protected')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.frames < 0 or args.warmup < 0 or args.start_frame < args.warmup or args.window < 1 or args.repeats < 1:
        parser.error('Require nonnegative frames/warmup, start-frame >= warmup, positive window/repeats')
    if len(args.seeds) != len(set(args.seeds)) or min(args.seeds) < 0:
        parser.error('seeds must be distinct nonnegative integers')
    if len(args.scenes) != len(set(args.scenes)) or len(args.variants) != len(set(args.variants)) or len(args.passes) != len(set(args.passes)):
        parser.error('scenes, variants and passes must not contain duplicates')
    for scene in args.scenes:
        if Path(scene).name != scene or not (ROOT / f'configs/Replica/{scene}.yaml').is_file():
            parser.error(f'Invalid Replica scene: {scene}')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume and not args.dry_run:
        parser.error('Output is nonempty; use a fresh root or --resume for completed matching runs')
    jobs = []
    for repeat in range(args.repeats):
        for seed in args.seeds:
            for index, scene in enumerate(args.scenes):
                variants = args.variants if (repeat + index + seed) % 2 == 0 else list(reversed(args.variants))
                for mode in args.passes:
                    for variant in variants:
                        run = output / mode / scene / f'{variant}_seed{seed}_repeat{repeat}'
                        ba_variant = 'baseline_fast' if variant in {'staging_only', 'pose_only'} else variant
                        cmd = [sys.executable, '-u', str(ROOT / 'scripts/Validation/benchmark_replica_ba.py'),
                               '--scene', scene, '--variant', ba_variant, '--frames', str(args.frames),
                               '--seed', str(seed), '--warmup', str(args.warmup), '--output', str(run)]
                        if variant in {'staging_only', 'pose_only'}:
                            cmd += ['--label', variant,
                                    '--preload-rays' if variant == 'staging_only' else '--no-preload-rays',
                                    '--fast-pose-commit' if variant == 'pose_only' else '--no-fast-pose-commit']
                        if mode == 'breakdown':
                            cmd.append('--ba-breakdown')
                        jobs.append((run, cmd, dict(scene=scene, variant=variant, seed=seed, repeat=repeat,
                                                   mode=mode, frames=args.frames, warmup=args.warmup)))
    if args.dry_run:
        for run, cmd, meta in jobs:
            print(json.dumps(dict(run=str(run), command=cmd, **meta)))
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / 'suite.json').write_text(json.dumps(dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                                                       jobs=[dict(run=str(r), command=c, **m) for r,c,m in jobs]), indent=2))
    for run, cmd, meta in jobs:
        if (run / 'summary.json').exists() and args.resume:
            if not (run / 'run.json').exists() or json.loads((run / 'run.json').read_text()) != meta:
                raise RuntimeError(f'Resume configuration mismatch: {run}')
            if meta['mode'] == 'breakdown' and not (run / 'ba_breakdown.csv').is_file():
                raise RuntimeError(f'Missing completed profile: {run}')
            hashes = json.loads((run / 'source_hashes.json').read_text())
            if any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != value
                   for name, value in hashes.items()):
                raise RuntimeError(f'Runtime source changed since completed run: {run}. Use a new output root.')
            print('SKIP_COMPLETE', run, flush=True)
            continue
        run.parent.mkdir(parents=True, exist_ok=True)
        log = run.with_suffix('.log')
        if log.exists() or (run.exists() and any(run.iterdir())):
            raise RuntimeError(f'Preserving incomplete run/log: {run}. Use a new output root for retry.')
        print('RUN', meta, flush=True)
        with log.open('w') as stream:
            subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        (run / 'run.json').write_text(json.dumps(meta, indent=2))
    summarize(output, args.start_frame, args.window, args.plot)
    if set(args.variants) == {'baseline', 'staging_only', 'pose_only', 'baseline_fast'}:
        from scripts.Validation.summarize_ba_mechanism_ablation import summarize as summarize_mechanisms
        summarize_mechanisms(output)
    (output / 'SUITE_COMPLETE').write_text('All requested runs and timing summary completed.\n')
    print('SUITE_COMPLETE', output / 'BA_BREAKDOWN_REPORT.md', flush=True)


if __name__ == '__main__':
    main()
