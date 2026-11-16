#!/usr/bin/env python3
"""Wrap native EC active call/iteration scopes and prove AST unwrapping parity."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil


def context(method,*args):
    return ast.Call(func=ast.Attribute(value=ast.Name(id='R2_TIMER',ctx=ast.Load()),attr=method,ctx=ast.Load()),args=[ast.Constant(a) if isinstance(a,str) else a for a in args],keywords=[])


def wrap(body,method,*args):
    return ast.With(items=[ast.withitem(context_expr=context(method,*args),optional_vars=None)],body=body,type_comment=None)


class Waits(ast.NodeTransformer):
    def visit_While(self,node):
        if 'mapping_idx' in ast.unparse(node.test):return wrap([node],'wait')
        return self.generic_visit(node)


class Strip(ast.NodeTransformer):
    def visit_With(self,node):
        node=self.generic_visit(node)
        call=node.items[0].context_expr
        if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and isinstance(call.func.value,ast.Name) and call.func.value.id=='R2_TIMER':return node.body
        return node


def instrument(source,kind):
    original=ast.parse(source);tree=copy.deepcopy(original)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    for func in cls.body:
        if not isinstance(func,ast.FunctionDef):continue
        if func.name in ('optimize_cam_in_batch','optimize_map'):
            index='cam_iter' if kind=='Tracker' else 'joint_iter'
            loop=next(n for n in ast.walk(func) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id==index)
            for i,node in enumerate(loop.body):
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='bg_loss':loop.body[i]=wrap([node],'component','forward_cuda_ms')
                if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='backward':loop.body[i]=wrap([node],'component','backward_cuda_ms')
            loop.body=[wrap(loop.body,'iteration',ast.Name(id=index,ctx=ast.Load()))]
        if func.name=='run' and kind=='Tracker':
            loop=next(n for n in func.body if isinstance(n,ast.For))
            publish=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='self.idx[0]')
            assert ast.unparse(loop.body[0]).startswith("idx = sample['frame_id']")
            active=loop.body[1:publish]
            active=[Waits().visit(n) for n in active]
            loop.body[1:publish]=[wrap(active,'call','tracking',ast.Name(id='idx',ctx=ast.Load()))]
        if func.name=='run' and kind=='Mapper':
            outer=next(n for n in func.body if isinstance(n,ast.While))
            first=next(i for i,n in enumerate(outer.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='pre_mapping_index != idx')
            assert 'self.optimize_map(idx)'==ast.unparse(outer.body[first+1])
            outer.body[first:first+2]=[wrap(outer.body[first:first+2],'call','bundle_adjustment',ast.Name(id='idx',ctx=ast.Load()))]
    ast.fix_missing_locations(tree)
    stripped=Strip().visit(copy.deepcopy(tree))
    assert ast.dump(stripped,include_attributes=False)==ast.dump(original,include_attributes=False),kind
    return ast.unparse(tree)+'\n'


def main():
    here=Path(__file__).resolve().parent;source=here/'ec_quality_runtime_v1';target=here/'ec_timing_runtime_v1'
    if target.exists():raise RuntimeError('Do not overwrite a frozen runtime')
    shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__'))
    audits={}
    for name in ('Tracker','Mapper'):
        p=target/'src'/(name+'.py');original=p.read_text();new=instrument(original,name);p.write_text(new)
        audits[name]=dict(native_sha256=hashlib.sha256(original.encode()).hexdigest(),timed_sha256=hashlib.sha256(new.encode()).hexdigest(),non_timer_ast_identical=True)
    shutil.copy2(here/'round2_ec_timing.py',target/'round2_ec_timing.py')
    driver=target/'run_ec_quality_round2.py';text=driver.read_text()
    text=text.replace('    from src.system import EC_SLAM',"    from round2_ec_timing import Timer\n    import importlib\n    timer=Timer()\n    for module in ('src.Tracker','src.Mapper'):setattr(importlib.import_module(module),'R2_TIMER',timer)\n    from src.system import EC_SLAM")
    text=text.replace('    estimate = slam.estimate_c2w_list', '    measured=timer.write(args.output)\n    estimate = slam.estimate_c2w_list')
    text=text.replace("    if args.export_mesh:\n", "    summary.update(measured,warmup_frames=20,timing_scope='Synchronized active worker call including input preparation and pose finalization; native handshake waits excluded and reported separately; native threads/streams retained; overlapping intervals are not additive',purpose='native_complete_call_profile',iteration_csv_schema='frame_id,iteration_type,iteration_id,forward_cuda_ms,backward_cuda_ms,full_iteration_cuda_ms')\n    if args.export_mesh:\n")
    ast.parse(text);driver.write_text(text)
    manifest=json.loads((source/'RUNTIME_MANIFEST.json').read_text())
    manifest.update(purpose='native EC complete-call and iteration timing; instrumented-run ATE retained separately',native_files_byte_identical=False,timer_ast_audit=audits,measurement_limit='Synchronization and events can perturb native thread interleaving. Same wrapper in BASE/MC, separate handshake waits, no summed-worker FPS claim.')
    manifest['runtime_hashes']={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='RUNTIME_MANIFEST.json'}
    (target/'RUNTIME_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (here/'EC_TIMING_AST_AUDIT.json').write_text(json.dumps(audits,indent=2)+'\n')
    jobs=json.loads((here/'manifests/rtx_ec_quality_all_replica.json').read_text())['jobs']
    (here/'manifests/rtx_ec_complete_call_timing.json').write_text(json.dumps(dict(purpose='Complete active-worker call timing, native threads and settings, all eight Replica scenes H16/H19, seeds 0/1/2; source statements unchanged after removing timer contexts; ATE is instrumentation sensitivity, not a replacement for primary native-quality runs',jobs=jobs),indent=2)+'\n')
    print(target,len(jobs),'timing jobs; non-timer AST parity passed')


if __name__=='__main__':main()
