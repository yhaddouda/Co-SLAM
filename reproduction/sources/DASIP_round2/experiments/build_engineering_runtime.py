#!/usr/bin/env python3
"""Freeze the native Co runtime plus a separately toggled engineering ablation."""
import ast
import hashlib
import json
from pathlib import Path
import shutil


def main():
    here=Path(__file__).resolve().parent;source=here/'coslam_reference_runtime_v1';target=here/'coslam_engineering_runtime_v1'
    if target.exists():raise RuntimeError('Do not overwrite a frozen runtime')
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(here/'round2_engineering.py',target/'round2_engineering.py')
    driver=target/'run_coslam_round2.py';text=driver.read_text()
    text=text.replace("    args=parser.parse_args()", "    parser.add_argument('--engineering',action='store_true')\n    args=parser.parse_args()\n    if args.engineering and args.condition!='BASE':parser.error('This ablation compares native BASE/ENG only')")
    text=text.replace('    coslam.cfg=cfg', "    cfg['round2']['engineering']=args.engineering\n    coslam.cfg=cfg")
    text=text.replace('    install(slam.model,', "    if args.engineering:\n        from round2_engineering import install as install_engineering\n        install_engineering(slam)\n    install(slam.model,")
    text=text.replace("    (output/'summary.json').write_text", "    summary.update(variant='ENG' if args.engineering else 'BASE',engineering=args.engineering)\n    (output/'summary.json').write_text")
    ast.parse(text);driver.write_text(text)
    from round2_engineering import transformed
    functions=transformed(target/'coslam.py')
    (target/'ENGINEERING_METHODS.py.txt').write_text('\n\n'.join(ast.unparse(f) for f in functions)+'\n')
    native=json.loads((source/'RUNTIME_MANIFEST.json').read_text())
    for name,value in native['runtime_hashes'].items():
        if name=='run_coslam_round2.py':continue
        assert hashlib.sha256((target/name).read_bytes()).hexdigest()==value
    native['ablation']='Optional cached fixed tracking targets, grouped ray transfer, batched pose conversion/commit; no scientific algorithm change; native pose aliases preserved'
    native['runtime_hashes']={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='RUNTIME_MANIFEST.json'}
    (target/'RUNTIME_MANIFEST.json').write_text(json.dumps(native,indent=2)+'\n')
    jobs=[]
    for seed in (0,1,2):
        for eng in ([False,True] if seed%2==0 else [True,False]):
            jobs.append(dict(id=f'office0_{"ENG" if eng else "BASE"}_seed{seed}',dataset='Replica',scene='office0',datadir='/home/yh279050/work/datasets/data/Replica/office0',condition='BASE',variant='ENG' if eng else 'BASE',engineering=eng,seed=seed,quality=True,export_mesh=False))
    (here/'manifests/rtx_engineering_office0.json').write_text(json.dumps(dict(purpose='Separate contemporaneous native BASE/ENG comparison, no pruning/Morton, office0 native H16, three paired seeds; complete-call timing and ATE',jobs=jobs),indent=2)+'\n')
    print(target,len(jobs))


if __name__=='__main__':main()
