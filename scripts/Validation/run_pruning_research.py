"""Full-scene pruning ablations. No ScanNet; no execution speedup variants.

Presets are fixed across datasets. Every run is an isolated CUDA process and
saves source/config hashes, trajectories, actual network-query counts and ATE.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCENES = {'Tum': 'fr1_desk', 'Replica': 'office0', 'Synthetic': 'tg'}
PRESETS = {
    'dense': ['--pruning', 'dense'],
    'legacy': ['--tracking', 'depth_spatial', '--variant', 'bucket'],
    'tail': ['--pruning', 'tail', '--pruning-margin', '0'],
    'margin': ['--pruning', 'tail', '--pruning-margin', '.15'],
    'ht2': ['--pruning', 'ht', '--free-stride', '2'],
    'ht4': ['--pruning', 'ht', '--free-stride', '4'],
    'adaptive': ['--pruning', 'adaptive', '--free-stride', '2'],
    'refine': ['--pruning', 'refine', '--pruning-margin', '0'],
    'refine_dense': ['--pruning', 'refine_dense', '--pruning-margin', '0'],
    'features': ['--pruning', 'adaptive', '--free-stride', '2', '--pixels', 'features'],
    'features_dense': ['--pruning', 'dense', '--pixels', 'features'],
    'tail_ba': ['--pruning', 'tail', '--pruning-margin', '0', '--pruning-stages', 'Bundle_Adjustment'],
    'tail_tracking': ['--pruning', 'tail', '--pruning-margin', '0', '--pruning-stages', 'Tracking'],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--presets', nargs='+', choices=PRESETS, default=['dense','legacy','tail','margin','ht2','adaptive','features'])
    p.add_argument('--datasets', nargs='+', choices=SCENES, default=list(SCENES))
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--frames', type=int, default=0)
    args = p.parse_args()
    os.chdir(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    for preset in args.presets:
        for seed in args.seeds:
            for dataset in args.datasets:
                name = f'{dataset}_{preset}_seed{seed}'
                out = args.output / name
                result = out / 'summary.json'
                if result.exists():
                    hashes = json.loads((out/'source_hashes.json').read_text())
                    changed = [f for f,h in hashes.items() if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h]
                    if changed:
                        raise ValueError(f'Sources changed for {out}: {changed}; use fresh directory')
                    print('REUSE',name,flush=True)
                    continue
                command = [sys.executable, '-u', 'scripts/Validation/benchmark_ba_accuracy.py',
                    '--dataset',dataset,'--scene',SCENES[dataset], '--seed',str(seed),'--frames',str(args.frames),
                    '--tracking','baseline','--variant','baseline', '--output',str(out), *PRESETS[preset]]
                print('START',name,flush=True)
                with (args.output/f'{name}.log').open('w') as log:
                    proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if proc.returncode:
                    print('FAILED',name,proc.returncode,flush=True)
                    (args.output/f'{name}.failure.json').write_text(json.dumps(dict(command=command,returncode=proc.returncode)))
                    raise RuntimeError(f'{name} failed; inspect log')
                row = json.loads(result.read_text())
                print('DONE',name,'ATE_cm',100*row['ate_rmse_m'],'queries',row['stage_queries'],flush=True)
    print('COMPLETE',args.output,flush=True)


if __name__ == '__main__':
    main()
