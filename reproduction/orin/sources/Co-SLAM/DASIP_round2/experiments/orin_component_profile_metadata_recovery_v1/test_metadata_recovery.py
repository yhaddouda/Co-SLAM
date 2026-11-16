"""CPU-only regressions for Orin optional telemetry and preserved required checks."""
import ast
import copy
import errno
import glob
import types
from unittest import mock
import optional_sysfs_metadata as telemetry
from local_support import *

OLD = EXPERIMENTS / 'orin_ec_component_profile_runtime_v2_single_seed'


class RemoveEnvironment(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.name == 'environment':
            node.body = [ast.Pass()]
            return node
        return self.generic_visit(node)


def main():
    tests = {}
    for error in [BlockingIOError(errno.EAGAIN, 'Resource temporarily unavailable'),
                  TypeError("can't concat NoneType to bytes"), FileNotFoundError('missing optional sensor')]:
        with mock.patch.object(telemetry.os, 'open', return_value=37), mock.patch.object(telemetry.os, 'read', side_effect=error), mock.patch.object(telemetry.os, 'close') as close:
            value, detail = telemetry.read_optional_sysfs('/sys/optional/sensor')
            assert value is None and detail['field'] == '/sys/optional/sensor'
            assert detail['exception_type'] == type(error).__name__ and detail['message'] == str(error)
            close.assert_called_once_with(37)
        tests[type(error).__name__ + '_explicitly_recorded'] = True
    for raw in [None, b'\xff']:
        with mock.patch.object(telemetry.os, 'open', return_value=37), mock.patch.object(telemetry.os, 'read', return_value=raw), mock.patch.object(telemetry.os, 'close'):
            value, detail = telemetry.read_optional_sysfs('/sys/optional/sensor')
            assert value is None and detail['status'] == 'unavailable'
    tests['None_read_and_invalid_text_do_not_become_valid_values'] = True
    with mock.patch.object(telemetry.os, 'open', return_value=37), mock.patch.object(telemetry.os, 'read', return_value=b'47343\n'), mock.patch.object(telemetry.os, 'close'):
        assert telemetry.read_optional_sysfs('/sys/optional/good') == ('47343', None)
    tests['available_value_preserved'] = True
    for bundle, driver in [('co_bundle', 'run_component_profile.py'), ('bundle', 'run_ec_component_profile.py')]:
        old_rt, new_rt = OLD / bundle / 'runtime', ROOT / bundle / 'runtime'
        original, changed = ast.parse((old_rt / driver).read_text()), ast.parse((new_rt / driver).read_text())
        assert ast.dump(RemoveEnvironment().visit(copy.deepcopy(original)), include_attributes=False) == ast.dump(RemoveEnvironment().visit(copy.deepcopy(changed)), include_attributes=False)
        tests[bundle + '_all_driver_computation_except_optional_metadata_AST_identical'] = True
        for path in old_rt.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            relative = path.relative_to(old_rt)
            if relative.as_posix() not in [driver, 'RUNTIME_MANIFEST.json']:
                assert sha(path) == sha(new_rt / relative), str(relative)
        tests[bundle + '_all_models_configs_timers_validators_protocols_byte_identical'] = True
        power = next(node for node in ast.walk(changed) if isinstance(node, ast.FunctionDef) and node.name == 'power')
        scope = dict(args=types.SimpleNamespace(power_mode='MAXN'))
        exec(compile(ast.Module(body=[copy.deepcopy(power)], type_ignores=[]), str(new_rt / driver), 'exec'), scope)
        with mock.patch('subprocess.check_output', return_value='NV Power Mode: MODE_15W\n1'):
            try:
                scope['power']()
            except AssertionError:
                pass
            else:
                raise AssertionError('Wrong power mode did not propagate')
        with mock.patch('subprocess.check_output', side_effect=PermissionError('required power query unavailable')):
            try:
                scope['power']()
            except PermissionError:
                pass
            else:
                raise AssertionError('Required power failure did not propagate')
        tests[bundle + '_required_wrong_or_unreadable_power_mode_still_raises'] = True
    fields, errors = {}, {}
    for pattern in ['/sys/class/devfreq/*/cur_freq', '/sys/class/thermal/thermal_zone*/temp']:
        for name in glob.glob(pattern):
            value, error = telemetry.read_optional_sysfs(name)
            fields[name] = value
            if error is not None:
                errors[name] = error
    assert fields and all(fields[name] is None for name in errors)
    save(ROOT / 'METADATA_READER_LIVE_CHECK.json', dict(utc=now(), sysfs_snapshot=fields, sysfs_read_errors=errors))
    tests['live_sysfs_unavailable_fields_explicit_without_losing_available_fields'] = True
    save(ROOT / 'METADATA_REGRESSION_CHECKS.json', dict(utc=now(), status='PASS', tests=tests, CUDA_initialized=False))
    print(json.dumps(tests, indent=2))


if __name__ == '__main__':
    main()
