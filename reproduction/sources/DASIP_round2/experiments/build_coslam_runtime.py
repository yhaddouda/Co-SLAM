#!/usr/bin/env python3
"""Build a separate public-reference runtime with explicit Round 2 adapters."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REVISION = '7b83b26d5ae24f4baa6c1c684dd745898276816d'
HERE = Path(__file__).resolve().parent


def timer(body, name):
    ctx = ast.parse("self.coslam_timing.stage(" + repr(name) + ")", mode='eval').body
    return ast.With(items=[ast.withitem(context_expr=ctx)], body=body)


class Instrument(ast.NodeTransformer):
    """Only enclose original statements in event ranges; retain estimator math."""
    def __init__(self):
        self.stage = None
        self.loops = 0
    def visit_FunctionDef(self, node):
        previous = self.stage
        self.stage = {'tracking_render': 'tracking', 'global_BA': 'bundle_adjustment'}.get(node.name)
        self.generic_visit(node)
        self.stage = previous
        return node
    def visit_For(self, node):
        self.generic_visit(node)
        if self.stage and ast.unparse(node.iter) in (
                "range(self.config['tracking']['iter'])", "range(self.config['mapping']['iters'])"):
            stage = self.stage
            transformed = []
            for statement in node.body:
                if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call) and ast.unparse(statement.value.func)=='self.model.forward':
                    transformed.append(timer([statement], 'forward_tr' if stage=='tracking' else 'Forward_BA'))
                elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Attribute) and statement.value.func.attr=='backward':
                    transformed.append(timer([statement], 'backward_tr' if stage=='tracking' else 'Backward_BA'))
                else:
                    transformed.append(statement)
            iteration = ast.parse('self.model._r2_iteration = i').body[0]
            node.body = [iteration, timer(transformed, 'TR_ITER_PROFILE_TOTAL' if stage=='tracking' else 'BA_ITER_PROFILE_TOTAL')]
            self.loops += 1
        return node


class UndoInstrumentation(ast.NodeTransformer):
    def visit_With(self, node):
        if len(node.items)==1 and ast.unparse(node.items[0].context_expr).startswith('self.coslam_timing.stage('):
            statements=[]
            for item in node.body:
                visited=self.visit(item)
                if isinstance(visited,list):statements.extend(visited)
                elif visited is not None:statements.append(visited)
            return statements
        return self.generic_visit(node)
    def visit_Assign(self, node):
        if any(ast.unparse(t)=='self.model._r2_iteration' for t in node.targets):
            return None
        return self.generic_visit(node)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=HERE.parents[1])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh runtime directory')
    paths=subprocess.check_output(['git','-C',str(args.repo),'ls-tree','-r','--name-only',REVISION],text=True).splitlines()
    original={}
    args.output.mkdir(parents=True)
    for name in paths:
        if Path(name).suffix not in ('.py','.yaml','.txt') or name=='requirements.txt':continue
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError(name)
        data=subprocess.check_output(['git','-C',str(args.repo),'show',REVISION+':'+name])
        path=args.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        original[name]=hashlib.sha256(data).hexdigest()
    coslam=args.output/'coslam.py'
    native=ast.parse(coslam.read_text()); modified=ast.parse(coslam.read_text())
    transform=Instrument();modified=transform.visit(modified)
    assert transform.loops==2,transform.loops
    ast.fix_missing_locations(modified)
    recovered=UndoInstrumentation().visit(ast.parse(ast.unparse(modified)))
    assert ast.dump(recovered,include_attributes=False)==ast.dump(native,include_attributes=False)
    coslam.write_text(ast.unparse(modified)+'\n')
    # New encoder selector is explicit; public FP32 requests remain unchanged.
    enc=args.output/'model/encodings.py'
    source=enc.read_text().replace('desired_resolution=512):','desired_resolution=512, hash="CoherentPrime"):')
    assert source!=enc.read_text()
    source=source.replace('"otype": \'HashGrid\',','"otype": \'HashGrid\',\n                "hash": hash,')
    assert '"hash": hash' in source
    enc.write_text(source)
    scene=args.output/'model/scene_rep.py'; source=scene.read_text()
    for attribute,resolution in [('embed_fn','resolution_sdf'),('embed_fn_color','resolution_color')]:
        old='desired_resolution=self.'+resolution+')'
        assert old in source
        source=source.replace(old,'desired_resolution=self.'+resolution+', hash=config["grid"].get("hash", "CoherentPrime"))')
    scene.write_text(source)
    # Keep the exact reviewed Orin event logger, without replacing its working-tree copy.
    timer_source=HERE/'orin_sources/Co-SLAM/model/layer_timing.py'
    shutil.copy2(timer_source,args.output/'round2_timing.py')
    for name in ('round2_sampling.py','run_coslam_round2.py'):
        shutil.copy2(HERE/name,args.output/name)
    extension=args.output/'model/round2_morton';extension.mkdir()
    shutil.copy2(args.repo/'model/fast_morton/morton_cuda.cu',extension/'morton_cuda.cu')
    runtime={str(p.relative_to(args.output)):digest(p) for p in args.output.rglob('*') if p.is_file()}
    record=dict(public_revision=REVISION,public_file_hashes=original,runtime_hashes=runtime,
        estimator_AST_preserved_after_removing_event_instrumentation=True,
        declared_changes=['coslam.py: event ranges and iteration identifier only; no estimator statement removed',
          'model/encodings.py and scene_rep.py: explicit optional hash selector, public FP32 retained',
          'Round 2 driver/adapters: opt-in support pruning, optional Morton ordering, separate call timers and output workflow'],
        preserved_public_files=['model/utils.py','model/decoder.py','datasets/dataset.py','model/keyframe.py','optimization/utils.py'])
    for name in record['preserved_public_files']:assert runtime[name]==original[name]
    (args.output/'RUNTIME_MANIFEST.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Built separate runtime:',args.output)


if __name__=='__main__':main()
