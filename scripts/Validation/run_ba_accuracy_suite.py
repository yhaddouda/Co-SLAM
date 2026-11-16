"""Sequential ATE ablations on one scene from each of four datasets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCENES = {'Tum': 'fr1_desk', 'ScanNet': 'scene0000', 'Synthetic': 'tg', 'Replica': 'office0'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frames', type=int, default=300, help='0 = full scenes')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0])
    parser.add_argument('--datasets', nargs='+', choices=SCENES, default=list(SCENES))
    parser.add_argument('--variants', nargs='+', default=['baseline', 'bucket_dense', 'bucket', 'lattice_guarded'])
    parser.add_argument('--tracking', choices=['baseline', 'depth_spatial', 'balanced_dense'], default='depth_spatial')
    parser.add_argument('--margin', type=float, default=0.)
    parser.add_argument('--ba-embed-eps', type=float)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in args.seeds:
        for dataset in args.datasets:
            for variant in args.variants:
                name = f'{dataset}_{variant}_seed{seed}'
                output = args.output / name
                summary = output / 'summary.json'
                if summary.exists():
                    row = json.loads(summary.read_text())
                    cfg = json.loads((output / 'config.json').read_text())
                    hashes = json.loads((output / 'source_hashes.json').read_text())
                    changed = [name for name, digest in hashes.items()
                               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
                    if changed:
                        raise ValueError(f'Source changed since {output}: {changed}; use a fresh output root')
                    if (row['tracking'] != args.tracking or row['seed'] != seed
                            or row['dataset'] != dataset or row['variant'] != variant
                            or row['margin'] != args.margin
                            or row.get('ba_embed_eps') != args.ba_embed_eps
                            or cfg['timing']['max_frames'] != (args.frames or None)):
                        raise ValueError(f'Incompatible completed run: {output}')
                    print(f'REUSE {name}', flush=True)
                else:
                    if output.exists() and any(output.iterdir()):
                        raise ValueError(f'Incomplete run exists: {output}; choose another output')
                    command = [sys.executable, '-u', 'scripts/Validation/benchmark_ba_accuracy.py',
                               '--dataset', dataset, '--scene', SCENES[dataset], '--variant', variant,
                               '--tracking', args.tracking, '--seed', str(seed), '--frames', str(args.frames),
                               '--margin', str(args.margin),
                               '--output', str(output)]
                    if args.ba_embed_eps is not None:
                        command.extend(['--ba-embed-eps', str(args.ba_embed_eps)])
                    print('START ' + ' '.join(command), flush=True)
                    with (args.output / f'{name}.log').open('w') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    row = json.loads(summary.read_text())
                summaries.append(row)
                (args.output / 'summaries.json').write_text(json.dumps(summaries, indent=2))
                print(f"DONE {name}: ATE={100 * row['ate_rmse_m']:.4f} cm; "
                      f"relative ATE={100 * row['ate_relative_rmse_m']:.4f} cm; "
                      f"frames={row['frames']}", flush=True)
    print('SUITE_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
