#!/usr/bin/env python3
"""Create a new timing-only snapshot; verify computation ASTs are unchanged."""
import ast
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def block(kind, label, body):
    expr = ast.parse(f"self._r2_component.scope({kind!r}, {label!r})", mode='eval').body
    return ast.With(items=[ast.withitem(context_expr=expr)], body=body)


def text(node):
    return ast.unparse(node)


def iteration_category(node):
    code = text(node)
    if 'self.model.forward' in code:
        return None  # Its model wrapper defines the exact forward interval.
    if 'loss.backward' in code:
        return 'backward'
    if 'self.get_loss_from_ret' in code:
        return 'external_loss'
    if 'best_sdf_loss' in code or 'thresh >' in code or 'torch.no_grad()' in code:
        return 'pose_selection'
    if '.step()' in code:
        return 'optimizer_update'
    return 'ray_pose_preparation'


def grouped_statements(nodes):
    output = []
    for node in nodes:
        category = iteration_category(node)
        if category is None:
            output.append(node)
        elif output and isinstance(output[-1], ast.With) and text(output[-1].items[0].context_expr) == f"self._r2_component.scope('iteration_component', {category!r})":
            output[-1].body.append(node)
        else:
            output.append(block('iteration_component', category, [node]))
    return output


class AddRanges(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name in ('global_BA', 'tracking_render'):
            body, seen_loop = [], False
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    body.append(stmt)
                    continue
                if isinstance(stmt, ast.For) and 'range' in text(stmt.iter):
                    seen_loop = True
                    for child in stmt.body:
                        if isinstance(child, ast.With) and 'ITER_PROFILE_TOTAL' in text(child.items[0].context_expr):
                            child.body = grouped_statements(child.body)
                    body.append(block('call_phase', 'loop_gaps', [stmt]))
                else:
                    phase = 'call_finalization' if seen_loop else 'call_setup'
                    body.append(block('call_phase', phase, [stmt]))
            assert seen_loop
            node.body = body
        elif node.name == 'smoothness':
            node.body = [block('iteration_component', 'dense_smoothness', node.body)]
        return node


class StripRanges(ast.NodeTransformer):
    def visit_With(self, node):
        node = self.generic_visit(node)
        if len(node.items) == 1 and '_r2_component.scope' in text(node.items[0].context_expr):
            return node.body
        return node


def verify_ast(reference, instrumented):
    left = ast.dump(ast.parse(reference), include_attributes=False)
    right = ast.dump(StripRanges().visit(ast.parse(instrumented)), include_attributes=False)
    assert left == right, 'Timing wrappers changed computational statements'


def build(base, name='component_profile_runtime_v1'):
    source = base / 'experiments/coslam_reference_runtime_v1'
    target = base / 'experiments' / name
    assert not target.exists(), 'Preserve an existing profiling snapshot'
    target.mkdir()
    native_manifest = json.loads((source / 'RUNTIME_MANIFEST.json').read_text())
    for name, expected in native_manifest['runtime_hashes'].items():
        assert sha(source / name) == expected, name
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, path)
    (target / 'reference/model').mkdir(parents=True)
    for name in ('coslam.py', 'model/scene_rep.py'):
        shutil.copy2(source / name, target / 'reference' / name)
    shutil.copy2(source / 'RUNTIME_MANIFEST.json', target / 'REFERENCE_RUNTIME_MANIFEST.json')
    original = (source / 'coslam.py').read_text()
    changed = ast.unparse(ast.fix_missing_locations(AddRanges().visit(ast.parse(original)))) + '\n'
    verify_ast(original, changed)
    (target / 'coslam.py').write_text(changed)
    original = (source / 'model/scene_rep.py').read_text()
    tree = ast.parse(original)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if method.name == 'forward':
                method.body = [block('forward_parent', 'model_forward', method.body)]
            elif method.name in ('render_rays', 'run_network'):
                category = 'sampling_rendering' if method.name == 'render_rays' else 'other_forward'
                method.body = [block('forward_category', category, method.body)]
    changed = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    verify_ast(original, changed)
    (target / 'model/scene_rep.py').write_text(changed)
    for name in ('component_profile_timer.py', 'run_component_profile.py', 'validate_component_profile.py'):
        shutil.copy2(base / 'experiments' / name, target / name)
    shutil.copy2(base / 'experiments/manifests/component_profile_protocol_v1.json', target / 'PROTOCOL.json')
    files = {str(p.relative_to(target)): sha(p) for p in target.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    manifest = dict(public_revision=native_manifest['public_revision'],
                    reference_manifest_sha256=sha(source / 'RUNTIME_MANIFEST.json'),
                    runtime_hashes=files, computational_ast_preserved=True,
                    baseline_only=True, engineering=False, pruning=False,
                    changes='Timing-only AST wrappers, dynamic encoder/decoder timing wrappers and separate selected-call driver; original sources and benchmark runtime retained')
    (target / 'RUNTIME_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('PROFILING_SNAPSHOT_BUILT_AST_MATCH', target)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='component_profile_runtime_v1')
    args = parser.parse_args()
    assert Path(args.name).name == args.name
    build(Path(__file__).resolve().parents[1], args.name)
