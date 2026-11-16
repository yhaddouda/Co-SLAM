"""Run fresh-process execution experiments sequentially on one GPU."""
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
    parser.add_argument('--system', choices=('coslam', 'ec_slam'), required=True)
    parser.add_argument('--scenes', nargs='+', default=['office1', 'room1', 'office0'])
    parser.add_argument('--variants', nargs='+', required=True)
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for', type=Path)
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.wait_for:
        while not args.wait_for.exists():
            print(f'WAIT {args.wait_for}', flush=True)
            time.sleep(30)
    script = ('benchmark_replica_tracking.py' if args.system == 'coslam'
              else 'benchmark_ec_replica_execution.py')
    for seed in args.seeds:
        for scene_index, scene in enumerate(args.scenes):
            index = scene_index % len(args.variants)
            order = args.variants[index:] + args.variants[:index]
            for variant in order:
                output = args.output / scene / f'{variant}_seed{seed}'
                summary_path = output / 'summary.json'
                expected_frames = args.frames or 2000
                if summary_path.exists():
                    result = json.loads(summary_path.read_text())
                    if (result['frames'] != expected_frames or result['diagnostic'] != args.diagnostic
                            or result['system'] != args.system or result['variant'] != variant):
                        raise ValueError(f'Incompatible completed run: {summary_path}')
                    print(f'REUSE {args.system} {scene} {variant} seed={seed}', flush=True)
                else:
                    if output.exists() and any(output.iterdir()):
                        raise ValueError(f'Incomplete run exists; use a new output root: {output}')
                    command = [sys.executable, '-u', str(ROOT / 'scripts/Validation' / script),
                               '--scene', scene, '--variant', variant, '--seed', str(seed),
                               '--frames', str(args.frames), '--output', str(output),
                               '--warmup', str(1 if 0 < args.frames <= 20 else 20)]
                    if args.diagnostic:
                        command.append('--diagnostic')
                    print(f'START {args.system} {scene} {variant} seed={seed} frames={expected_frames}', flush=True)
                    with (args.output / f'{scene}_{variant}_seed{seed}.log').open('w') as stream:
                        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                    result = json.loads(summary_path.read_text())
                    print(f"DONE {args.system} {scene} {variant} ATE={1000 * result['ate_rmse_m']:.4f}mm "
                          f"tracking={result['tracking_invocations']['mean_ms']:.2f}ms", flush=True)
                completed = [json.loads(path.read_text()) for path in args.output.glob('*/*/summary.json')]
                (args.output / 'summaries.json').write_text(json.dumps(completed, indent=2))
    (args.output / 'SUITE_COMPLETE').write_text(json.dumps(dict(
        system=args.system, scenes=args.scenes, variants=args.variants, seeds=args.seeds,
        frames=args.frames or 2000), indent=2))
    print(f'SUITE_COMPLETE {args.output}', flush=True)


if __name__ == '__main__':
    main()
