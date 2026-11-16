"""CPU-only preparation of the pinned native EC timing adapter."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / 'ec_public_reference'
TARGET = ROOT / 'ec_runtime'


def scope(expr, body):
    return ast.With(items=[ast.withitem(context_expr=ast.parse(expr, mode='eval').body)], body=body)


class Instrument(ast.NodeTransformer):
    def __init__(self, kind):
        self.kind = kind
        self.function = ''

    def visit_FunctionDef(self, node):
        old = self.function
        self.function = node.name
        self.generic_visit(node)
        self.function = old
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        if self.function == 'run' and self.kind == 'Tracker' and isinstance(node.target, ast.Name) and node.target.id == 'sample':
            node.body = [scope('self._r2_timer.frame("tracking_frame", sample["frame_id"])', node.body)]
        elif isinstance(node.target, ast.Name) and node.target.id in ('cam_iter', 'joint_iter'):
            stage = '"tracking"' if self.kind == 'Tracker' else '("first_frame_mapping" if idx == 0 else "bundle_adjustment")'
            node.body = [scope(f'self._r2_timer.iteration({stage}, {node.target.id})', node.body)]
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        if self.kind == 'Mapper' and self.function == 'run' and any(isinstance(x, ast.While) for x in node.body):
            # Keep the native scheduling wait outside mapper frame work.
            node.body = [node.body[0], scope('self._r2_timer.frame("mapping_frame", idx)', node.body[1:])]
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if self.kind == 'Mapper' and self.function == 'run' and 'c2wSave' in ast.unparse(node.test):
            node.test = ast.BoolOp(op=ast.And(), values=[ast.parse('not self.cfg.get("round2_timing_only", False)', mode='eval').body, node.test])
        return node

    def visit_Assign(self, node):
        if self.function in ('optimize_map', 'optimize_cam_in_batch') and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'forward':
                return scope('self._r2_timer.component("forward_cuda_ms")', [node])
        return node

    def visit_Expr(self, node):
        if self.function in ('optimize_map', 'optimize_cam_in_batch') and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'backward':
                return scope('self._r2_timer.component("backward_cuda_ms")', [node])
        return node


class Strip(ast.NodeTransformer):
    def visit_With(self, node):
        self.generic_visit(node)
        if '_r2_timer' in ast.unparse(node.items[0].context_expr):
            return node.body
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.BoolOp) and 'round2_timing_only' in ast.unparse(node.test.values[0]):
            node.test = node.test.values[1]
        return node


def main():
    native = json.loads((REFERENCE / 'NATIVE_SOURCE_MANIFEST.json').read_text())
    for name, digest in native['file_hashes'].items():
        assert hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() == digest, name
    if TARGET.exists():
        raise RuntimeError('Preserving existing isolated EC runtime')
    shutil.copytree(REFERENCE, TARGET)
    audit = {'public_revision': native['public_revision'], 'preserves_native_thread_scheduling': True,
             'declared_changes': ['Timing contexts only in Tracker/Mapper; native waits/transfers/order retained',
                                  'Disable only post-run trajectory/checkpoint/ATE block in timing-only mode',
                                  'Add explicit hash selector to native HashGrid constructor; both encoders remain FP32',
                                  'Driver supplies absolute local data path, actual once-only seed, diagnostic length cap and optional Morton permutation',
                                  'Whole-call synchronizations may include other-thread work on shared CUDA stream; not additive isolated costs'],
             'estimator_ast_verified': {}, 'native_source_manifest': native}
    diffs = []
    for name in ('Tracker', 'Mapper'):
        path = Path('src') / (name + '.py')
        old = (REFERENCE / path).read_text()
        tree = Instrument(name).visit(ast.parse(old));ast.fix_missing_locations(tree)
        new = ast.unparse(tree) + '\n'
        stripped = Strip().visit(ast.parse(new));ast.fix_missing_locations(stripped)
        assert ast.dump(stripped, include_attributes=False) == ast.dump(ast.parse(old), include_attributes=False), name
        audit['estimator_ast_verified'][name] = True
        (TARGET / path).write_text(new)
        diffs.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='native/'+str(path), tofile='adapted/'+str(path)))
    model = TARGET / 'src/Model.py'
    old = model.read_text()
    new = old.replace('import tinycudann as tcnn', 'import tinycudann as tcnn\nROUND2_HASH = "CoherentPrime"')
    assert new != old
    new = new.replace('"otype": \'HashGrid\',', '"otype": \'HashGrid\',\n                "hash": ROUND2_HASH,')
    assert '"hash": ROUND2_HASH' in new
    model.write_text(new)
    diffs.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='native/src/Model.py', tofile='adapted/src/Model.py'))
    (ROOT / 'EC_NATIVE_CHANGE_AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
    (ROOT / 'EC_NATIVE_CHANGES.diff').write_text(''.join(diffs))
    print('Prepared isolated native EC adapter; Tracker/Mapper estimator AST verified')


if __name__ == '__main__':
    main()
