"""Patch isolated optional telemetry, preserve original hashes, and rebind the five-profile chain."""
import ast
import copy
import difflib
import shlex
import shutil
from local_support import *

OLD = EXPERIMENTS / 'orin_ec_component_profile_runtime_v2_single_seed'
PYTHON = '/home/yh279050/miniforge3/envs/coslam/bin/python'


def main():
    assert read(OLD / 'PROFILING_COMPLETE.json')['full_profiles_completed'] == 0
    assert read(DP / 'DP_COMPLETION_STATUS.json')['expanded_completed'] == 292
    differences = []
    identities = {}
    for bundle, driver in [('co_bundle', 'run_component_profile.py'), ('bundle', 'run_ec_component_profile.py')]:
        rt = ROOT / bundle / 'runtime'
        original = (OLD / bundle / 'runtime' / driver).read_text()
        lines = original.splitlines(keepends=True)
        node = next(n for n in ast.walk(ast.parse(original)) if isinstance(n, ast.FunctionDef) and n.name == 'environment')
        old_environment = ''.join(lines[node.lineno - 1:node.end_lineno])
        start = old_environment.index('        readings = {}\n')
        end = old_environment.index('        return dict(')
        block = ('        from optional_sysfs_metadata import read_optional_sysfs\n'
                 '        readings, read_errors = {}, {}\n'
                 "        for pattern in ('/sys/class/devfreq/*/cur_freq', '/sys/class/thermal/thermal_zone*/temp'):\n"
                 '            import glob\n'
                 '            for name in glob.glob(pattern):\n'
                 '                value, error = read_optional_sysfs(name)\n'
                 '                readings[name] = value\n'
                 '                if error is not None:\n'
                 '                    read_errors[name] = error\n')
        new_environment = old_environment[:start] + block + old_environment[end:]
        assert 'sysfs_snapshot=readings' in new_environment
        new_environment = new_environment.replace('sysfs_snapshot=readings', 'sysfs_snapshot=readings, sysfs_read_errors=read_errors')
        changed = original.replace(old_environment, new_environment)
        assert changed != original
        (rt / driver).write_text(changed)
        shutil.copyfile(ROOT / 'optional_sysfs_metadata.py', rt / 'optional_sysfs_metadata.py')
        manifest_path = rt / 'RUNTIME_MANIFEST.json'
        parent_hash = sha(OLD / bundle / 'runtime/RUNTIME_MANIFEST.json')
        manifest = read(manifest_path)
        manifest['runtime_hashes'][driver] = sha(rt / driver)
        manifest['runtime_hashes']['optional_sysfs_metadata.py'] = sha(rt / 'optional_sysfs_metadata.py')
        manifest['metadata_reader_recovery'] = dict(parent_runtime_manifest_sha256=parent_hash,
            change='Optional sysfs reads only: direct os.read; null plus field/exception/errno for unavailable telemetry',
            numerical_checks_event_boundaries_configs_and_concurrency_unchanged=True)
        save(manifest_path, manifest)
        identities[bundle] = dict(parent_runtime_manifest_sha256=parent_hash,
                                  recovered_runtime_manifest_sha256=sha(manifest_path))
        for name in [driver, 'optional_sysfs_metadata.py', 'RUNTIME_MANIFEST.json']:
            parent = OLD / bundle / 'runtime' / name
            before = parent.read_text().splitlines(keepends=True) if parent.exists() else []
            differences.extend(difflib.unified_diff(before, (rt / name).read_text().splitlines(keepends=True),
                fromfile='original/' + bundle + '/runtime/' + name, tofile='recovered/' + bundle + '/runtime/' + name))
        # Preserve original package identities and declare a new local payload.
        shutil.copyfile(OLD / bundle / 'SHA256SUMS.json', ROOT / bundle / 'ORIGINAL_SHA256SUMS.json')
        shutil.copyfile(OLD / bundle / 'BUILD_MANIFEST.json', ROOT / bundle / 'ORIGINAL_BUILD_MANIFEST.json')
        save(ROOT / bundle / 'BUILD_MANIFEST.json', dict(revision='Orin optional-metadata recovery v1',
            parent_build_manifest_sha256=sha(ROOT / bundle / 'ORIGINAL_BUILD_MANIFEST.json'),
            parent_payload_sha256=sha(ROOT / bundle / 'ORIGINAL_SHA256SUMS.json'), **identities[bundle]))
        hashes = {p.relative_to(ROOT / bundle).as_posix(): sha(p) for p in (ROOT / bundle).rglob('*')
                  if p.is_file() and p.name != 'SHA256SUMS.json' and '__pycache__' not in p.parts}
        # Include nested files named SHA256SUMS only when present; original catalog was copied above.
        save(ROOT / bundle / 'SHA256SUMS.json', hashes)
    (ROOT / 'METADATA_PATCH.diff').write_text(''.join(differences))
    save(ROOT / 'RECOVERY_RUNTIME_IDENTITIES.json', identities)
    for name in ['manifests', 'gates', 'results', '15W_deferred']:
        (ROOT / name).mkdir(exist_ok=True)
    old_prefix, new_prefix = str(OLD), str(ROOT)
    for source_dir in ['manifests', 'gates', '15W_deferred']:
        for source in (OLD / source_dir).glob('*.json'):
            value = json.loads(source.read_text().replace(old_prefix, new_prefix))
            save(ROOT / source_dir / source.name, value)
    config = read(OLD / 'INSTALLATION_CONFIG.json')
    config.update(created_utc=now(), session='dasip_r2_profile_metadata_recovery_maxn',
                  retired_waiting_follower=read(OLD / 'FOLLOWER_IDENTITY.json'),
                  previous_failed_profile_root=str(OLD), recovery_runtime_identities=identities)
    save(ROOT / 'INSTALLATION_CONFIG.json', config)
    for system, bundle, queue in [('CO', 'co_bundle', 'co_maxn'), ('EC', 'bundle', 'ec_maxn')]:
        save(ROOT / f'{system}_COMPONENT_PROFILE_ACTIVE.json', dict(queue=str(ROOT / 'results' / queue),
            runtime=str(ROOT / bundle / 'runtime'), manifest=str(ROOT / 'manifests' / f'{system}_MAXN_SINGLE_SEED.json')))
    gate = read(ROOT / 'gates/EC_AFTER_CO_SUCCESS.json')
    replacements = {}
    for bundle in ['co_bundle', 'bundle']:
        replacements[sha(OLD / bundle / 'runtime/RUNTIME_MANIFEST.json')] = sha(ROOT / bundle / 'runtime/RUNTIME_MANIFEST.json')
    replacements[sha(OLD / 'manifests/CO_MAXN_SINGLE_SEED.json')] = sha(ROOT / 'manifests/CO_MAXN_SINGLE_SEED.json')
    for condition in gate['conditions']:
        for check in condition['checks']:
            if isinstance(check['equals'], str) and check['equals'] in replacements:
                check['equals'] = replacements[check['equals']]
    gate['wait_processes'].append(read(OLD / 'CO_CONTROLLER_IDENTITY.json'))
    save(ROOT / 'gates/EC_AFTER_CO_SUCCESS.json', gate)
    # Keep all original successful benchmark gates; additionally require failed profile parents gone.
    for gate_name in ['UPSTREAM_SUCCESS_AND_EXPORTS.json', 'CO_CONTROLLER_GATE.json']:
        value = read(ROOT / 'gates' / gate_name)
        value['wait_processes'].extend(read(OLD / name) for name in ['FOLLOWER_IDENTITY.json', 'CO_CONTROLLER_IDENTITY.json', 'EC_CONTROLLER_IDENTITY.json'])
        save(ROOT / 'gates' / gate_name, value)
    shutil.copyfile(OLD / 'FROZEN_SELECTIONS.json', ROOT / 'FROZEN_SELECTIONS.json')
    script = '#!/bin/bash\nset -euo pipefail\nsource /home/yh279050/miniforge3/etc/profile.d/conda.sh\nconda activate coslam\nexport PYTHONDONTWRITEBYTECODE=1\ncd /home/yh279050/work/Co-SLAM\nexec python -B -u ' + shlex.quote(str(ROOT / 'continuation.py')) + ' >> ' + shlex.quote(str(ROOT / 'follower.log')) + ' 2>&1\n'
    (ROOT / 'launch_follow_maxn.sh').write_text(script)
    os.chmod(ROOT / 'launch_follow_maxn.sh', 0o755)
    files = {str(p): sha(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.log' and p.name != 'INSTALLATION_FREEZE.json'}
    files.update(read(DP / 'INSTALLATION_FREEZE.json')['files'])
    files.update(read(DP / 'EXECUTION_FREEZE.json')['files'])
    for c in config['native_profile_controls'] + config['native_ec_controls']:
        for key in ['config_path', 'summary_path', 'completion_path']:
            files[c[key]] = sha(c[key])
    receipt = read(ROOT / 'PRESERVED_FAILURE_ARCHIVE.json')
    files[receipt['path']] = receipt['sha256']
    save(ROOT / 'INSTALLATION_FREEZE.json', dict(utc=now(), files=files))
    print('NEW_RUNTIME_AND_GATES_PREPARED', ROOT)


if __name__ == '__main__':
    main()
