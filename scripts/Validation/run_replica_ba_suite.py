"""Run fresh-process BA experiments sequentially on one GPU; resume completed runs."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', nargs='+', default=['office1', 'room1', 'office0'])
    parser.add_argument('--variants', nargs='+', default=['baseline', 'bucket', 'lattice_guarded'])
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--margin', type=float, default=0.)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for', type=Path, help='Do not compete with an already running benchmark')
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    # Optional file-backed dependencies let queued suites share one GPU without
    # competing. Paths are relative to the repository root, like CLI paths.
    dependency_file = args.output / 'wait_for.json'
    dependencies = json.loads(dependency_file.read_text()).get('paths', []) if dependency_file.exists() else []
    for dependency in dependencies:
        while not Path(dependency).exists():
            print(f'Waiting for dependency {dependency}', flush=True)
            time.sleep(30)
    if args.wait_for:
        while not args.wait_for.exists():
            print(f'Waiting for {args.wait_for}', flush=True)
            time.sleep(30)
    for seed in args.seeds:
        for scene_index, scene in enumerate(args.scenes):
            # Rotate order to spread thermal/order effects across conditions.
            order = args.variants[scene_index:] + args.variants[:scene_index]
            for variant in order:
                output = args.output / scene / f'{variant}_seed{seed}'
                summary = output / 'summary.json'
                if summary.exists():
                    record = json.loads(summary.read_text())
                    expected_frames = args.frames or 2000
                    if record['frames'] != expected_frames or record['diagnostic'] != args.diagnostic or record['margin'] != args.margin:
                        raise ValueError(f'Incompatible completed run: {summary}')
                    print(f'REUSE {scene} {variant} seed={seed}', flush=True)
                else:
                    if output.exists() and any(output.iterdir()):
                        raise ValueError(f'Incomplete existing run; choose a new output directory: {output}')
                    log = args.output / f'{scene}_{variant}_seed{seed}.log'
                    command = [sys.executable, '-u', 'scripts/Validation/benchmark_replica_ba.py',
                        '--scene', scene, '--variant', variant, '--frames', str(args.frames),
                        '--seed', str(seed), '--margin', str(args.margin), '--output', str(output),
                        '--warmup', str(1 if 0 < args.frames <= 20 else 20)]
                    if args.diagnostic:
                        command.append('--diagnostic')
                    print(f'START {scene} {variant} seed={seed} frames={args.frames or 2000}', flush=True)
                    with log.open('w') as handle:
                        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
                    record = json.loads(summary.read_text())
                    print(f"DONE {scene} {variant} ATE={1000 * record['ate_rmse_m']:.4f}mm "
                          f"BA={record['ba_invocations']['mean_ms']:.2f}ms "
                          f"BA-frame={record['ba_frames']['mean_ms']:.2f}ms", flush=True)
                completed = [json.loads(p.read_text()) for p in args.output.glob('*/*/summary.json')]
                (args.output / 'summaries.json').write_text(json.dumps(completed, indent=2))
    print(f'SUITE_COMPLETE {args.output}', flush=True)


if __name__ == '__main__':
    main()
