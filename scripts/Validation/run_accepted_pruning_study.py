"""Retain a pruning study only after all accuracy and runtime gates pass.

All runs execute in an automatically removed temporary directory. The requested
output directory receives results only after independent full-scene acceptance,
including a matched full-sample control proving a pruning-dependent speed gain.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from validate_pruning_acceptance import SCENES, evaluate, report

ROOT = Path(__file__).resolve().parents[2]


def run(root, family, dataset, scene, seed, margin, every):
    path=root/family/f'{dataset}_seed{seed}'
    path.parent.mkdir(parents=True,exist_ok=True)
    mode='dense' if family=='controls' else 'support'
    args=[sys.executable,'scripts/Validation/benchmark_ba_accuracy.py',
        '--dataset',dataset,'--scene',scene,'--seed',str(seed),'--frames','0',
        '--tracking','baseline','--variant','baseline','--pruning',mode,
        '--pruning-margin',str(margin),'--anchor-every',str(every),'--output',str(path)]
    print('Validating',dataset,'seed',seed,family,flush=True)
    with (path.parent/(path.name+'.log')).open('w') as log:
        result=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        # TemporaryDirectory removes failed trial files on exception as well.
        tail=(path.parent/(path.name+'.log')).read_text()[-3000:]
        raise RuntimeError('Benchmark process failed:\n'+tail)
    return json.loads((path/'summary.json').read_text())


def archive_validation(root):
    names=[Path('scripts/Validation')/name for name in (
        'validate_pruning_acceptance.py','verify_pruning_results.py',
        'run_accepted_pruning_study.py','plot_accepted_pruning.py',
        'capture_execution_environment.py')]
    names+=sorted(Path('tests').glob('*.py'))
    hashes={}
    for name in names:
        source=ROOT/name
        target=root/'validation_source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        hashes[str(name)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (root/'validation_source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--pruning-margin',type=float,default=.15)
    parser.add_argument('--anchor-every',type=int,default=3)
    parser.add_argument('--plot-python',type=Path,help='Python with Matplotlib; defaults to the active interpreter')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.exists():
        raise ValueError('Use a fresh output directory')
    if len(args.seeds)!=len(set(args.seeds)) or len(args.seeds)<3:
        raise ValueError('Provide at least three distinct seeds for retained results')
    with tempfile.TemporaryDirectory(prefix='coslam_pruning_validation_') as temporary:
        root=Path(temporary)
        for seed in args.seeds:
            for dataset,(scene,_) in SCENES.items():
                # Alternate ordering to reduce systematic run-order bias.
                order=['controls','selected'] if seed%2==0 else ['selected','controls']
                for family in order:
                    run(root,family,dataset,scene,seed,args.pruning_margin,args.anchor_every)
        result=evaluate(root,args.seeds)
        if not result['passed']:
            raise SystemExit('Independent acceptance not met. Trial discarded.')
        for dataset,(scene,_) in SCENES.items():
            a=run(root,'matched_control',dataset,scene,args.seeds[0],1000000,args.anchor_every)
            b=json.loads((root/'selected'/f'{dataset}_seed{args.seeds[0]}'/'summary.json').read_text())
            if not all(b[k]['mean_ms']<a[k]['mean_ms'] for k in ('tracking_invocations','ba_invocations')):
                raise SystemExit('Pruning-dependent runtime benefit not verified. Trial discarded.')
        result=evaluate(root,args.seeds)
        if not result['passed']:
            raise SystemExit('Final acceptance not met. Trial discarded.')
        subprocess.run([sys.executable,'scripts/Validation/capture_execution_environment.py','--output',str(root/'environment.json')],cwd=ROOT,check=True)
        with (root/'regression.log').open('w') as log:
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        archive_validation(root)
        (root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
        text=report(result)
        method=ROOT/'docs/interval_support_pruning.md'
        if method.exists():
            text+='\n'+method.read_text()
        (root/'REPORT.md').write_text(text)
        plotting=subprocess.run([str(args.plot_python or sys.executable),'scripts/Validation/plot_accepted_pruning.py',str(root)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if plotting.returncode:
            (root/'REPORT.md').write_text(text.replace('![Accepted pruning, ATE and complete stage wall time](accepted_pruning.png)','Plot not generated; run scripts/Validation/plot_accepted_pruning.py with a Python environment containing Matplotlib.'))
            print('Optional plot unavailable; acceptance and all raw data are retained.',flush=True)
        output.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(root,output)
    print('Accepted study saved to',output)


if __name__=='__main__':
    main()
