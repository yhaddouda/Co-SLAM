"""Run full paired scenes in scratch; export only after the 17%/30% gates pass."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from validate_ba30_pruning import evaluate, SCENES

ROOT=Path(__file__).resolve().parents[2]


def run(root,family,dataset,scene,seed,margin=.15,every=3,fraction=.125):
    path=root/family/f'{dataset}_seed{seed}'
    path.parent.mkdir(parents=True,exist_ok=True)
    policy='dense' if family=='controls' else ('support' if family=='interval_only' else 'support_tv')
    if family=='matched_control':margin,fraction=1000000,1.
    command=[sys.executable,'scripts/Validation/benchmark_ba_accuracy.py',
        '--dataset',dataset,'--scene',scene,'--seed',str(seed),'--frames','0',
        '--tracking','baseline','--variant','baseline','--pruning',policy,
        '--pruning-margin',str(margin),'--anchor-every',str(every),
        '--tracking-anchor-every',str(every),'--tv-fraction',str(fraction),'--output',str(path)]
    print('RUN',dataset,seed,family,flush=True)
    with (path.parent/(path.name+'.log')).open('w') as stream:
        process=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError((path.parent/(path.name+'.log')).read_text()[-3000:])
    print('COMPLETE',dataset,seed,family,flush=True)


def archive_validation(root):
    names=[Path('scripts/Validation')/n for n in (
        'validate_ba30_pruning.py','validate_pruning_acceptance.py','verify_pruning_results.py',
        'run_ba30_pruning_study.py','plot_ba30_pruning.py','write_ba30_report.py',
        'capture_execution_environment.py')]
    names+=sorted((ROOT/'tests').glob('*.py'))
    hashes={}
    for name in names:
        source=ROOT/name
        relative=source.relative_to(ROOT)
        target=root/'validation_source'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (root/'validation_source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    # These dependencies supplement the per-run runtime manifests.
    supplements=['utils.py','config.py','docs/sampled_smoothness_pruning.md']
    supplements += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'configs/PruningBA30').glob('*.yaml'))]
    for name in supplements:
        source=ROOT/name;target=root/'supplemental_source'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    (root/'supplemental_source_hashes.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in supplements},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--prior-root',type=Path,help='Optional additional guard against the earlier published dense reference')
    p.add_argument('--tv-fraction',type=float,default=.125)
    p.add_argument('--pruning-margin',type=float,default=.15)
    p.add_argument('--anchor-every',type=int,default=3)
    p.add_argument('--plot-python',type=Path)
    args=p.parse_args();output=args.output.resolve()
    if output.exists():raise ValueError('Use a new output directory')
    if len(args.seeds)<3 or len(set(args.seeds))!=len(args.seeds):raise ValueError('At least three distinct seeds required')
    with tempfile.TemporaryDirectory(prefix='coslam_ba30_') as scratch:
        root=Path(scratch)
        for dataset,(scene,_) in SCENES.items():
            for seed in args.seeds:
                for family in (('controls','selected') if seed%2==0 else ('selected','controls')):
                    run(root,family,dataset,scene,seed,args.pruning_margin,args.anchor_every,args.tv_fraction)
        result=evaluate(root,args.prior_root,args.seeds)
        if not result['passed']:raise SystemExit('Criteria not met; trial discarded.')
        for family in ('matched_control','interval_only'):
            for dataset,(scene,_) in SCENES.items():
                run(root,family,dataset,scene,args.seeds[0],args.pruning_margin,args.anchor_every,args.tv_fraction)
        result=evaluate(root,args.prior_root,args.seeds,require_ablations=True)
        if not result['passed']:raise SystemExit('Final validation failed; trial discarded.')
        (root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
        subprocess.run([sys.executable,'scripts/Validation/capture_execution_environment.py','--output',str(root/'environment.json')],cwd=ROOT,check=True)
        with (root/'regression.log').open('w') as stream:
            subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        archive_validation(root)
        subprocess.run([sys.executable,'scripts/Validation/write_ba30_report.py',str(root)],cwd=ROOT,check=True)
        subprocess.run([str(args.plot_python or sys.executable),'scripts/Validation/plot_ba30_pruning.py',str(root)],cwd=ROOT,check=True)
        output.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(root,output)
    print('Accepted study saved to',output)

if __name__=='__main__':main()
