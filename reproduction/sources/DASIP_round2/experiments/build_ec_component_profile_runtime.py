#!/usr/bin/env python3
"""Freeze native EC code with removable, thread-local timing contexts only."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrap(body, method, *args):
    call = ast.Call(func=ast.Attribute(value=ast.Name(id='R2_PROFILE', ctx=ast.Load()),
                                     attr=method, ctx=ast.Load()),
                    args=[ast.Constant(a) if isinstance(a, str) else a for a in args], keywords=[])
    return ast.With(items=[ast.withitem(context_expr=call)], body=body)


class Strip(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        call = node.items[0].context_expr
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'R2_PROFILE':
            return node.body
        return node


def verify_ast(native, timed):
    assert ast.dump(ast.parse(native), include_attributes=False) == ast.dump(
        Strip().visit(ast.parse(timed)), include_attributes=False), 'Non-timing AST changed'


class Waits(ast.NodeTransformer):
    def visit_While(self, node):
        if 'mapping_idx' in ast.unparse(node.test):
            return wrap([node], 'wait')
        return self.generic_visit(node)


def iteration_parts(nodes):
    result = []
    for node in nodes:
        code = ast.unparse(node)
        if 'vMapModel.forward' in code:
            result.append(node)  # Model.forward defines its inclusive parent.
            continue
        if '.backward()' in code:
            category = 'backward'
        elif '.step()' in code or '.zero_grad()' in code:
            category = 'optimizer_update'
        elif any(key in code for key in ('current_loss', 'loss_camera_tensor', 'last_best', 'candidate_cam_tensor', 'debugShow', 'self.verbose', 'showTrackerLoss')):
            category = 'pose_selection'
        elif code == 'batch_loss = bg_loss':
            category = 'external_loss'
        else:
            category = 'ray_pose_preparation'
        context = f"R2_PROFILE.scope('iteration_component', {category!r})"
        if result and isinstance(result[-1], ast.With) and ast.unparse(result[-1].items[0].context_expr) == context:
            result[-1].body.append(node)
        else:
            result.append(wrap([node], 'scope', 'iteration_component', category))
    return result


def instrument(source, kind):
    tree = ast.parse(source)
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        for func in cls.body:
            if not isinstance(func, ast.FunctionDef):
                continue
            if kind == 'Model':
                if cls.name == 'Model' and func.name == 'forward':
                    func.body = [wrap(func.body, 'scope', 'forward_parent', 'model_forward')]
                elif cls.name == 'Model' and func.name == 'render_rays':
                    func.body = [wrap(func.body, 'scope', 'forward_category', 'sampling_rendering')]
                elif cls.name == 'HashGrid' and func.name == 'forward':
                    for i, node in enumerate(func.body):
                        if ast.unparse(node).startswith('embed = self.embed_fn('):
                            func.body[i] = wrap([node], 'query', ast.Name(id='points', ctx=ast.Load()))
                    func.body = [wrap(func.body, 'scope', 'forward_category', 'other_forward')]
                elif cls.name in ('ColorNet', 'SDFNet') and func.name == 'forward':
                    func.body = [wrap(func.body, 'scope', 'forward_category', 'decoder')]
                continue
            if func.name in ('optimize_cam_in_batch', 'optimize_map'):
                index = 'cam_iter' if kind == 'Tracker' else 'joint_iter'
                seen = False
                body = []
                for node in func.body:
                    if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == index:
                        node.body = [wrap(iteration_parts(node.body), 'iteration', ast.Name(id=index, ctx=ast.Load()))]
                        body.append(wrap([node], 'scope', 'call_phase', 'loop_gaps'))
                        seen = True
                    else:
                        body.append(wrap([node], 'scope', 'call_phase', 'call_finalization' if seen else 'call_setup'))
                assert seen
                func.body = body
            if func.name == 'run' and kind == 'Tracker':
                loop = next(n for n in func.body if isinstance(n, ast.For))
                publish = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'self.idx[0]')
                active = [Waits().visit(n) for n in loop.body[1:publish]]
                # Everything after native optimize_cam_in_batch is pose finalization.
                boundary = next(i for i, n in enumerate(active) if isinstance(n, ast.If) and ast.unparse(n.test) == 'idx == 0')
                branch = active[boundary]
                opt = next(i for i, n in enumerate(branch.orelse) if 'candidate_cam_tensor = self.optimize_cam_in_batch' in ast.unparse(n))
                branch.orelse[opt+1:] = [wrap(branch.orelse[opt+1:], 'scope', 'call_phase', 'call_finalization')]
                active[boundary+1:] = [wrap(active[boundary+1:], 'scope', 'call_phase', 'call_finalization')]
                loop.body[1:publish] = [wrap(active, 'call', 'tracking', ast.Name(id='idx', ctx=ast.Load()))]
            if func.name == 'run' and kind == 'Mapper':
                outer = next(n for n in func.body if isinstance(n, ast.While))
                first = next(i for i, n in enumerate(outer.body) if isinstance(n, ast.If) and ast.unparse(n.test) == 'pre_mapping_index != idx')
                assert ast.unparse(outer.body[first+1]) == 'self.optimize_map(idx)'
                outer.body[first:first+2] = [wrap(outer.body[first:first+2], 'call', 'bundle_adjustment', ast.Name(id='idx', ctx=ast.Load()))]
    changed = ast.unparse(ast.fix_missing_locations(tree))+'\n'
    verify_ast(source, changed)
    return changed


def build(base, name='ec_component_profile_runtime_v1'):
    source = base/'experiments/ec_quality_runtime_v1'
    assert Path(name).name == name
    target = base/'experiments'/name
    assert not target.exists(), 'Never overwrite a frozen runtime'
    reference = json.loads((source/'RUNTIME_MANIFEST.json').read_text())
    target.mkdir()
    for name, expected in reference['runtime_hashes'].items():
        assert sha(source/name) == expected, name
        path = target/name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, path)
    (target/'reference/src').mkdir(parents=True)
    for name in ('Tracker', 'Mapper', 'Model'):
        relative = f'src/{name}.py'
        shutil.copy2(source/relative, target/'reference'/relative)
        (target/relative).write_text(instrument((source/relative).read_text(), name))
    shutil.copy2(source/'RUNTIME_MANIFEST.json', target/'REFERENCE_RUNTIME_MANIFEST.json')
    for name in ('component_profile_timer.py', 'ec_component_profile_timer.py', 'run_ec_component_profile.py', 'validate_ec_component_profile.py'):
        shutil.copy2(base/'experiments'/name, target/name)
    protocol = json.loads((base/'experiments/manifests/component_profile_protocol_v1.json').read_text())
    protocol.update(system='EC-SLAM', protocol_version=1,
                    public_reference=reference['public_revision'],
                    native_reference_manifest_sha256=sha(source/'RUNTIME_MANIFEST.json'),
                    workloads=[w for w in protocol['workloads'] if w['dataset']=='Replica'],
                    status='EXECUTABLE_ISOLATED_RUNTIME_REQUIRES_LOCAL_PREFLIGHTS')
    for work in protocol['workloads']:
        work.update(native_scene_config='configs/Replica/office0.yaml',
                    native_scene_config_sha256=sha(source/'configs/Replica/office0.yaml'),
                    max_tracking_updates=8, native_ba_updates=10,
                    smoothness='Native Model.forward accepts smooth but contains no smoothness computation')
    protocol['platforms'] = [dict(id=p, seeds=seeds, full_profile_passes=2*len(seeds)) for p,seeds in (
        ('RTX_4070',[0,1,2]), ('ORIN_MAXN',[0,1]), ('ORIN_15W',[0,1]))]
    protocol['total_full_profile_passes'] = 14
    (target/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2)+'\n')
    manifest = dict(system='EC-SLAM', public_revision=reference['public_revision'],
                    reference_manifest_sha256=sha(source/'RUNTIME_MANIFEST.json'),
                    computational_ast_preserved=True, baseline_only=True, engineering=False, pruning=False,
                    native_concurrent_workers=True,
                    changes='Removable timing contexts only in Tracker/Mapper/Model; isolated native-thread driver; no estimator, reader or configuration algorithm changes',
                    runtime_hashes={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
    (target/'RUNTIME_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('EC_COMPONENT_RUNTIME_BUILT_AST_PARITY', target)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',default='ec_component_profile_runtime_v1')
    args=parser.parse_args()
    build(Path(__file__).resolve().parents[1],args.name)
