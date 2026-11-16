"""CPU-only local bindings; never import Torch, load data, or alter power/clocks."""
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
TOP = ROOT.parents[1]
EXPERIMENTS = ROOT.parent
DP = EXPERIMENTS / 'orin_dp_completion_maxn_v1'
SCAN = EXPERIMENTS / 'orin_scannet_repair_v1'
OLD_EXPORT = EXPERIMENTS / 'orin_existing_dp_export_v1'
GPU_LOCK = '/tmp/dasip_round2_gpu0.lock'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_identity(pid):
    try:
        p = Path('/proc', str(pid))
        tail = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), start_ticks=int(tail[19]), state=tail[0],
                    cmdline=(p / 'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


def still_alive(expected):
    current = process_identity(expected['pid'])
    return bool(current and current['state'] != 'Z' and
                current['start_ticks'] == expected['start_ticks'] and
                current['boot_id'] == expected['boot_id'])


def require_mode(expected='MAXN'):
    raw = subprocess.check_output(['/usr/sbin/nvpmodel', '-q'], text=True, timeout=10).strip()
    name = next((line.split(':', 1)[1].strip() for line in raw.splitlines() if line.startswith('NV Power Mode:')), None)
    assert name == expected, expected + ' required; no settings changed: ' + raw
    if expected == 'MAXN':
        assert raw == 'NV Power Mode: MAXN\n0', raw
    return raw


def require_maxn():
    return require_mode('MAXN')


def idle_sample(expected_mode='MAXN'):
    # This is the exact established Orin checker, with profile children added.
    checker = load_module('validated_orin_idle', EXPERIMENTS / 'orin_after_v4_maxn_v1/post_v4_follower.py')
    sample = checker.idle()
    extra = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except FileNotFoundError:
            continue
        except PermissionError as error:
            # A visible non-kernel process that cannot be inspected is unknown.
            raise RuntimeError('Cannot establish process ownership: ' + str(p)) from error
        if any(Path(a).name in ['run_component_profile.py', 'validate_component_profile.py'] for a in args if a):
            extra.append(dict(pid=int(p.name), args=args))
    assert not extra, 'Profiling child is still present: ' + repr(extra)
    sample['power'] = require_mode(expected_mode)
    return sample


def fresh_idle_three():
    samples = []
    for _ in range(3):
        samples.append(idle_sample())
        time.sleep(2)
    return samples


def frozen_sources_ok():
    for path, expected in read(ROOT / 'INSTALLATION_FREEZE.json')['files'].items():
        assert sha(path) == expected, 'Frozen source/config changed: ' + path


def pointer_get(record, pointer):
    value = record
    for key in pointer.strip('/').split('/'):
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def gate_ready(spec, loader=read, exists=lambda p: Path(p).exists(), alive=still_alive):
    for path in spec.get('failure_files', []):
        if exists(path):
            raise RuntimeError('Upstream failure preserved: ' + path)
    for condition in spec['conditions']:
        if not exists(condition['file']):
            return False, 'Waiting for ' + condition['file']
        record = loader(condition['file'])
        for check in condition['checks']:
            try:
                value = pointer_get(record, check['pointer'])
            except (KeyError, IndexError):
                return False, 'Missing field ' + condition['file'] + check['pointer']
            if value != check['equals']:
                return False, 'Waiting for ' + condition['file'] + check['pointer']
    for process in spec.get('wait_processes', []):
        if alive(process):
            return False, 'Waiting for normal parent exit: ' + str(process['pid'])
    return True, 'All exact successful upstream/export records and parent exits are present'


def wake(config, message, receipt_name):
    receipt = ROOT / receipt_name
    if receipt.exists():
        return
    n = config['notification']
    try:
        result = subprocess.run([n['codex_executable'], 'queue', '--thread', n['thread_id'],
                                 '--message', message], capture_output=True, text=True, timeout=55)
        save(receipt if result.returncode == 0 else receipt.with_suffix('.error.json'),
             dict(utc=now(), returncode=result.returncode, stdout=result.stdout,
                  stderr=result.stderr, email_sent=False, mechanism='Existing same-conversation completion wake'))
    except Exception as error:
        save(receipt.with_suffix('.error.json'), dict(utc=now(), error=repr(error), email_sent=False))
