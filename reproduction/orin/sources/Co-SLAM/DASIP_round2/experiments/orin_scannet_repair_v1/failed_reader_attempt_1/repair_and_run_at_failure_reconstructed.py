"""Isolated, exclusive ScanNet preparation followed by the authorized serial queue.

No scientific source edits, downloads, power changes, quality work or EC ENG.
"""
import ast
from collections import Counter
from datetime import datetime, timezone
import difflib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'orin_after_v4_maxn_v1'
V4 = ROOT.parent / 'orin_continuous_v4'
CO = ROOT.parent / 'orin_coslam_runtime_v1/runtime'
TOP = ROOT.parents[1]
sys.path.insert(0, str(OLD))
import post_v4_execute as support
from post_v4_follower import idle, identity
support.ROOT = ROOT  # Only the new readiness records use this destination.


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def update(s):
    s['updated_utc'] = now()
    s['counts'] = dict(Counter(j['status'] for j in s['jobs']))
    done = sum(j['status'] in ['done', 'reused'] for j in s['jobs'])
    s['ALL_REQUIRED_MAXN_COMPLETE'] = done == 24 and s.get('prior_completed_slots') == 172
    save(ROOT / 'SCANNET_STATUS.json', s)
    previous = read(ROOT / 'preserved/CURRENT_SCOPE_STATUS.json')
    changes = {j['id']: j for j in s['jobs']}
    for j in previous['jobs']:
        if j['id'] in changes:
            j.update(changes[j['id']])
    previous.update(updated_utc=now(), scope='Co engineering only; EC BASE/MC unchanged; EC ENG excluded',
                    phase=s['phase'], active_child_pid=s.get('active_child_pid'),
                    active_log=s.get('active_log'), authoritative_scannet_status=str(ROOT / 'SCANNET_STATUS.json'))
    previous['counts'] = dict(Counter(j['status'] for j in previous['jobs']))
    previous['ALL_REQUIRED_MAXN_COMPLETE'] = s['ALL_REQUIRED_MAXN_COMPLETE']
    save(ROOT / 'CURRENT_SCOPE_STATUS.json', previous)
    # Current overlays are mutable observer documents, never frozen v4 source/history.
    save(V4 / 'CURRENT_SCOPE_STATUS.json', previous)
    lines = ['# MAXN ScanNet reader repair and continuation', '', s['updated_utc'], '',
             f"Phase: {s['phase']}; ScanNet counts: {s['counts']}; prior completed slots: 172/196.",
             f"ALL_REQUIRED_MAXN_COMPLETE={str(s['ALL_REQUIRED_MAXN_COMPLETE']).lower()}. EC ENG excluded.", '',
             f"Parent PID: {os.getpid()}; active child: {s.get('active_child_pid')}; log: {s.get('active_log')}", '',
             '| Job | State | Evidence / reason |', '|---|---|---|']
    lines += [f"| {j['id']} | {j['status']} | {j.get('output', j.get('reason', ''))} |" for j in s['jobs']]
    (ROOT / 'ORIN_RUN_AGENDA.md').write_text('\n'.join(lines) + '\n')


def preserve():
    dest = ROOT / 'preserved'
    dest.mkdir(exist_ok=False)
    names = ['SCANNET_PREPARATION_STATUS.json', 'prepare_scannet_after_gate.py',
             'FOLLOWER_FREEZE.json', 'POST_V4_COMPLETE.json', 'NORMAL_EXHAUSTION_TRANSITION.json',
             'SCANNET_SCENES.json', 'SCANNET_ACCESS_AUTHORIZATION.json',
             'CO_ENGINEERING_LOCAL_VALIDATION.json', 'ENGINEERING_FAILURE_REPORT.json',
             'ENGINEERING_FAILURE_REPORT.md', 'data_utilities/SensorData.py.original']
    hashes = {}
    for name in names:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OLD / name, target)
        hashes[name] = sha(target)
    shutil.copyfile(V4 / 'CURRENT_SCOPE_STATUS.json', dest / 'CURRENT_SCOPE_STATUS.json')
    hashes['CURRENT_SCOPE_STATUS.json'] = sha(dest / 'CURRENT_SCOPE_STATUS.json')
    save(dest / 'PRESERVATION_MANIFEST.json', dict(utc=now(), files=hashes, originals_untouched=True))


def export_completed():
    """Deliver two verified original result bundles plus explicit scope/import mapping."""
    archives = ['ORIN_CURRENT_MODE_RESULTS.zip', 'ORIN_POST_V4_MAXN_RESULTS.zip']
    receipts = []
    for name in archives:
        p = TOP / name
        digest = sha(p)
        assert (TOP / (name + '.sha256')).read_text().split()[0] == digest
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None
            manifest = read_zip = json.loads(z.read('RETURN_MANIFEST.json'))
            files = manifest['files']
            for path, expected in files.items():
                assert hashlib.sha256(z.read(path)).hexdigest() == expected, (name, path)
        receipts.append(dict(name=name, sha256=digest, verified_entries=len(files), bytes=p.stat().st_size))
    scope = read(ROOT / 'preserved/CURRENT_SCOPE_STATUS.json')
    completed = [j for j in scope['jobs'] if j['status'] in ['done', 'reused']]
    assert len(completed) == 172 and sum(j['status'] == 'reused' for j in completed) == 16
    assert read(OLD / 'CO_ENGINEERING_LOCAL_VALIDATION.json')['status'] == 'PASS'
    doc = dict(utc=now(), completed_slots=172, required_slots=196, pending_scannet_slots=24,
               EC_ENG='excluded from scope and completion', ALL_REQUIRED_MAXN_COMPLETE=False,
               archives=receipts, completed_identity_mapping=completed,
               duplicate_policy='The 16 reused full slots are already included in the v4 archive. Import by identity, never count reuse twice.',
               engineering='Use matched RTX-reference BASE controls and ENG in post-v4 archive. Preserve original failed online checks; later fixed-state pass uses unchanged 1e-6 tolerances.')
    save(ROOT / 'RTX_IMPORT_MANIFEST.json', doc)
    note = ('# Completed Orin MAXN results for RTX import\n\n'
            'Snapshot made before ScanNet timings restart. 172/196 required full slots completed; '
            '24 ScanNet BASE/MC slots remain. EC ENG is excluded; EC BASE/MC remain unchanged.\n\n'
            'Extract the two nested ZIPs separately. ORIN_CURRENT_MODE_RESULTS.zip contains v4 '
            'raw iteration and complete-call timings, effective configs, runtime/source identities, '
            'all 16 reused slots and the EC source-only runtime. ORIN_POST_V4_MAXN_RESULTS.zip '
            'contains four completed matched Co reference BASE/ENG jobs, local fixed-state '
            'validation, the original engineering failures and diagnostic evidence. '
            'Their internal RETURN_MANIFEST.json files retain original hashes and historical scope. '
            'RTX_IMPORT_MANIFEST.json supplies the current Co-only ENG scope and 172-slot identity mapping. '
            'Do not count historical EC ENG as blocked requirements or count reused data twice.\n\n'
            'No datasets, compiled extensions, meshes or reconstruction metrics are included. '
            'Reader guard failure and exact old trace are preserved under reader_failure/. '
            'Power is MAXN; author-selected clocks retained, full jetson_clocks lock state unknown.\n')
    (ROOT / 'RTX_IMPORT_README.md').write_text(note)
    target = TOP / 'ORIN_COMPLETED_MAXN_FOR_RTX_20261020.zip'
    assert not target.exists(), 'Preserve an existing export; choose a new version'
    entries = {}
    with zipfile.ZipFile(target.with_suffix('.zip.tmp'), 'w', compression=zipfile.ZIP_STORED) as z:
        for name in archives:
            z.write(TOP / name, name)
            entries[name] = sha(TOP / name)
        for p in [ROOT / 'RTX_IMPORT_MANIFEST.json', ROOT / 'RTX_IMPORT_README.md']:
            z.write(p, p.name)
            entries[p.name] = sha(p)
        for p in (ROOT / 'preserved').rglob('*'):
            if p.is_file():
                name = 'reader_failure/' + p.relative_to(ROOT / 'preserved').as_posix()
                z.write(p, name)
                entries[name] = sha(p)
        z.writestr('TRANSFER_MANIFEST.json', json.dumps(dict(files=entries), indent=2) + '\n')
    temp = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
        for n, h in entries.items():
            assert hashlib.sha256(z.read(n)).hexdigest() == h
    temp.replace(target)
    digest = sha(target)
    target.with_suffix('.zip.sha256').write_text(digest + '  ' + target.name + '\n')
    save(ROOT / 'RTX_EXPORT_VERIFIED.json', dict(utc=now(), path=str(target), sha256=digest,
         bytes=target.stat().st_size, entries_verified=len(entries), original_archives=receipts,
         completed_slots=172, before_scannet_timings=True))
    print('RTX_EXPORT_VERIFIED', target, digest, flush=True)


def make_reader():
    target = ROOT / 'data_utilities'
    target.mkdir(exist_ok=False)
    revision = subprocess.check_output(['git', 'ls-remote', 'https://github.com/ScanNet/ScanNet.git',
                                       'refs/heads/master'], text=True, timeout=90).split()[0]
    assert len(revision) == 40 and all(c in '0123456789abcdef' for c in revision)
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package
    converter = RefactoringTool(get_fixers_from_package('lib2to3.fixes'))
    provenance = dict(repository='https://github.com/ScanNet/ScanNet', revision=revision, files={})
    for name in ['SensorData.py', 'reader.py']:
        url = f'https://raw.githubusercontent.com/ScanNet/ScanNet/{revision}/SensReader/python/{name}'
        with urllib.request.urlopen(url, timeout=90) as f:
            raw = f.read(500001)
        assert len(raw) < 500000
        if name == 'SensorData.py':
            assert raw == (OLD / 'data_utilities/SensorData.py.original').read_bytes(), 'Upstream source changed; review required'
        (target / (name + '.original')).write_bytes(raw)
        converted = str(converter.refactor_string(raw.decode(), name))
        converted = converted.replace("''.join(struct.unpack", "b''.join(struct.unpack").replace('"".join(struct.unpack', 'b"".join(struct.unpack')
        tree = ast.parse(converted)
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ['eval', 'exec', 'compile', '__import__'] for n in ast.walk(tree))
        assert not any(isinstance(n, ast.Import) and any(x.name in ['subprocess', 'socket', 'requests'] for x in n.names) for n in ast.walk(tree))
        if name == 'SensorData.py':
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SensorData')
            methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
            assert {'__init__', 'load', 'export_color_images', 'export_depth_images', 'export_poses', 'export_intrinsics'} <= methods.keys()
            # Validate the actual native declaration/check, rather than a literal source spelling.
            assert any(isinstance(n, ast.Assign) and ast.dump(n.targets[0]) == ast.dump(ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='version', ctx=ast.Store())) and isinstance(n.value, ast.Constant) and n.value.value == 4 for n in ast.walk(methods['__init__']))
            assert any(isinstance(n, ast.Assert) and ast.unparse(n.test) == 'self.version == version' for n in ast.walk(methods['load']))
        (target / name).write_text(converted)
        (target / (name + '.diff')).write_text(''.join(difflib.unified_diff(raw.decode().splitlines(True), converted.splitlines(True), fromfile=name+'.original', tofile=name)))
        provenance['files'][name] = dict(url=url, original_sha256=hashlib.sha256(raw).hexdigest(), adapted_sha256=sha(target/name))
    if importlib.util.find_spec('png') is None:
        (target / 'png.py').write_text('import cv2\nimport numpy as np\nclass Writer:\n def __init__(self,width,height,bitdepth=16,greyscale=True,**kw):\n  assert bitdepth==16 and greyscale\n  self.width,self.height=width,height\n def write(self,stream,rows):\n  a=np.asarray(list(rows),dtype=np.uint16).reshape(self.height,self.width)\n  ok,b=cv2.imencode(".png",a);assert ok\n  stream.write(b.tobytes())\n')
        provenance['png_adapter_sha256'] = sha(target / 'png.py')
    provenance['edits'] = ['lib2to3 syntax', 'struct c byte joins', 'optional local lossless uint16 PNG adapter']
    provenance['native_version_requirement'] = 'self.version=4 and assert self.version==version retained; each binary header independently required to be 4'
    save(target / 'READER_PROVENANCE.json', provenance)
    return target


def preparation_worker(scene_name):
    """Official full export, then validate against raw native values before releasing memory."""
    import numpy as np
    import cv2
    import imageio.v2 as imageio
    sys.path.insert(0, str(ROOT / 'data_utilities'))
    from SensorData import SensorData
    scene = next(x for x in read(OLD / 'SCANNET_SCENES.json')['scenes'] if x['scene'] == scene_name)
    raw = OLD / 'downloaded_data/scans' / scene['raw_scan_id'] / (scene['raw_scan_id'] + '.sens')
    with raw.open('rb') as f:
        assert struct.unpack('<I', f.read(4))[0] == 4
    target = ROOT / 'prepared_data/scans' / scene['raw_scan_id']
    assert not target.exists(), 'Preserve partial extractions; no overwrite'
    print('LOAD_NATIVE_SENS', raw, flush=True)
    data = SensorData(str(raw))
    n = scene['frames']
    assert len(data.frames) == n
    assert data.color_compression_type == 'jpeg' and data.depth_compression_type == 'zlib_ushort'
    data.export_color_images(str(target / 'color'), frame_skip=1)
    data.export_depth_images(str(target / 'depth'), frame_skip=1)
    data.export_poses(str(target / 'pose'), frame_skip=1)
    data.export_intrinsics(str(target / 'intrinsic'))
    invalid = []
    files = {}
    probes = []
    for directory, suffix in [('color', '.jpg'), ('depth', '.png'), ('pose', '.txt')]:
        paths = list((target / directory).glob('*' + suffix))
        assert len(paths) == n and {int(p.stem) for p in paths} == set(range(n))
    for i, frame in enumerate(data.frames):
        p = target / 'depth' / f'{i}.png'
        depth = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        expected_depth = np.frombuffer(frame.decompress_depth(data.depth_compression_type), dtype=np.uint16).reshape(data.depth_height, data.depth_width)
        assert depth.dtype == np.uint16 and np.array_equal(depth, expected_depth), ('depth', i)
        pose = np.loadtxt(target / 'pose' / f'{i}.txt')
        # Native export uses six-decimal %f text, including original NaN/inf locations.
        expected_pose = np.array([[float('%f' % v) for v in row] for row in frame.camera_to_world])
        assert pose.shape == (4, 4) and np.array_equal(pose, expected_pose, equal_nan=True), ('pose', i)
        if not np.isfinite(pose).all():
            invalid.append(i)
        for directory, suffix in [('color', '.jpg'), ('depth', '.png'), ('pose', '.txt')]:
            f = target / directory / f'{i}{suffix}'
            files[f.relative_to(target).as_posix()] = sha(f)
        if i in {0, n//2, n-1}:
            rgb = imageio.imread(target / 'color' / f'{i}.jpg')
            expected_jpeg = imageio.imwrite(imageio.RETURN_BYTES, frame.decompress_color(data.color_compression_type), format='jpg')
            assert expected_jpeg == (target / 'color' / f'{i}.jpg').read_bytes(), ('native JPEG export', i)
            assert rgb.shape == (data.color_height, data.color_width, 3)
            probes.append(dict(frame=i, rgb_shape=list(rgb.shape), depth_shape=list(depth.shape), depth_dtype=str(depth.dtype), native_jpeg_bytes_equal=True, pose_finite=bool(np.isfinite(pose).all())))
        if i % 250 == 0:
            print('VALIDATED_NATIVE_VALUES', scene_name, i, '/', n, flush=True)
    intrinsics = {}
    for name in ['intrinsic_color', 'extrinsic_color', 'intrinsic_depth', 'extrinsic_depth']:
        p = target / 'intrinsic' / (name + '.txt')
        a = np.loadtxt(p)
        expected = np.array([[float('%f' % v) for v in row] for row in getattr(data, name)])
        assert np.array_equal(a, expected, equal_nan=True)
        intrinsics[name] = a.tolist()
        files[p.relative_to(target).as_posix()] = sha(p)
    save(ROOT / 'data_manifests' / (scene_name + '.json'), dict(files=files))
    record = dict(status='ready', scene=scene_name, path=str(target), frames=n, frame_stride=1,
                  raw_archive=str(raw), raw_sha256=sha(raw), sens_version=4, depth_shift=data.depth_shift,
                  frame_ids_complete=True, all_depth_values_equal_native=True, all_pose_values_equal_native_text=True,
                  color_export='Unmodified official imageio JPEG export; first/middle/last re-encoded bytes exactly match',
                  invalid_pose_indices_preserved=invalid, pose_axes_preflipped=False,
                  intrinsics=intrinsics, probes=probes, file_manifest_sha256=sha(ROOT / 'data_manifests' / (scene_name + '.json')))
    save(ROOT / 'validation' / (scene_name + '.json'), record)
    print('SCENE_VALIDATED', scene_name, n, 'invalid_GT_poses_preserved', len(invalid), flush=True)


def child(argv, log, fd, s):
    support.maxn()
    idle()
    with Path(log).open('x') as stream:
        p = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, pass_fds=(fd,))
        s.update(active_child_pid=p.pid, active_command=argv, active_log=str(log))
        save(ROOT / 'ACTIVE_CHILD_LAUNCH.json', dict(utc=now(), identity=identity(p.pid), argv=argv, log=str(log)))
        update(s)
        while p.poll() is None:
            time.sleep(10)
        rc = p.wait()
    s['active_child_pid'] = None
    support.maxn()
    for _ in range(15):
        try:
            idle()
            break
        except AssertionError:
            time.sleep(2)
    else:
        raise RuntimeError('GPU/process ownership did not clear after child')
    update(s)
    return rc


def verify_freeze():
    for path, h in read(ROOT / 'EXECUTION_FREEZE.json')['files'].items():
        assert sha(path) == h, 'Frozen source changed: ' + path


def main():
    (ROOT / 'logs').mkdir(exist_ok=True)
    with open('/tmp/dasip_round2_scannet_repair_v1.lock', 'a') as controller, open('/tmp/dasip_round2_gpu0.lock', 'a') as workload:
        fcntl.flock(controller, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(workload, fcntl.LOCK_EX | fcntl.LOCK_NB)
        support.maxn()
        samples = []
        for _ in range(3):
            samples.append(idle())
            time.sleep(2)
        assert read(V4 / 'CONTINUOUS_COMPLETE.json')['major_admitted_series_terminal']
        assert read(OLD / 'POST_V4_COMPLETE.json')['major_admitted_series_terminal']
        for root, name in [(V4, 'CONTINUOUS_FREEZE.json'), (OLD, 'FOLLOWER_FREEZE.json')]:
            for path, h in read(root / name)['files'].items():
                p = Path(path)
                assert sha(p if p.is_absolute() else root / p) == h
        preserve()
        save(ROOT / 'EXCLUSIVE_TRANSITION.json', dict(utc=now(), parent=identity(os.getpid()), idle_samples=samples,
             workload_lock='/tmp/dasip_round2_gpu0.lock', power=support.maxn(), normal_exhaustion_reverified=True))
        baseline = support.hardware()
        save(ROOT / 'ENVIRONMENT.json', baseline)
        prior = read(ROOT / 'preserved/CURRENT_SCOPE_STATUS.json')
        assert sum(j['status'] in ['done', 'reused'] for j in prior['jobs']) == 172
        jobs = read(OLD / 'manifests/orin_scannet_native.json')['jobs']
        old_jobs = {j['id']: j for j in prior['jobs']}
        assert len(jobs) == 24 and all(old_jobs[j['id']]['status'] == 'blocked' for j in jobs)
        s = dict(phase='EXPORTING_COMPLETED_MAXN', parent_pid=os.getpid(), prior_completed_slots=172,
                 jobs=[dict(j, status='preparing') for j in jobs], active_child_pid=None, EC_ENG='excluded')
        update(s)
        try:
            export_completed()
            s['phase'] = 'PREPARING_ALL_SIX_SCENES'
            update(s)
            make_reader()
            for scene in read(OLD / 'SCANNET_SCENES.json')['scenes']:
                assert sha(CO / 'configs/ScanNet' / (scene['scene'] + '.yaml')) == scene['scene_config_sha256']
                argv = [sys.executable, '-u', str(Path(__file__).resolve()), '--prepare-scene', scene['scene']]
                log = ROOT / 'logs' / (scene['scene'] + '_prepare.log')
                if child(argv, log, workload.fileno(), s) != 0:
                    for j in s['jobs']:
                        if j['scene'] == scene['scene']:
                            j.update(status='blocked', reason='Preparation failed; full trace retained in ' + str(log))
                else:
                    record = read(ROOT / 'validation' / (scene['scene'] + '.json'))
                    for j in s['jobs']:
                        if j['scene'] == scene['scene']:
                            j.update(status='validated', datadir=record['path'])
                update(s)
            s['phase'] = 'CAPACITY_PREFLIGHT'
            update(s)
            # The original continuation admitted H12/H14 bounded diagnostics before full timing.
            for h in [12, 14]:
                j = next((j for j in s['jobs'] if j['hash_size'] == h and j['status'] == 'validated'), None)
                if not j:
                    continue
                for cond in ['BASE', 'MC']:
                    out = ROOT / 'diagnostics' / f"{j['scene']}_{cond}_H{h}"
                    argv = [sys.executable, str(CO / 'run_coslam_round2.py'), '--dataset', 'ScanNet', '--scene', j['scene'],
                            '--datadir', j['datadir'], '--condition', cond, '--seed', '0', '--hash-size', str(h),
                            '--frames', '50', '--warmup', '20', '--output', str(out), '--expected-power-mode', 'MAXN']
                    assert child(argv, ROOT / 'logs' / (out.name + '_preflight.log'), workload.fileno(), s) == 0, 'Capacity diagnostic failure; preserve and stop before full timings'
                    support.verify_output(dict(j, condition=cond, seed=0), out, diagnostic=True)
            save(ROOT / 'PREPARATION_COMPLETE.json', dict(utc=now(), scenes=[read(p) for p in sorted((ROOT / 'validation').glob('*.json'))],
                 all_preparation_finished=True, no_background_preparation=True, new_capacity_preflight='PASS',
                 frozen_slam_config='unchanged', engineering='Co only, already complete; EC ENG excluded'))
            queue_adapter = ROOT / 'run_queue_inherited_lock.py'
            shutil.copyfile(OLD / 'run_queue_inherited_lock.py', queue_adapter)
            manifest_files = []
            for j in s['jobs']:
                if j['status'] != 'validated':
                    continue
                j['status'] = 'queued'
                manifest = ROOT / 'manifests' / (j['id'] + '.json')
                save(manifest, dict(warmup_frames=20, jobs=[{k: v for k, v in j.items() if k not in ['status', 'reason']}]))
                manifest_files.append(manifest)
            files = [Path(__file__).resolve(), queue_adapter, OLD / 'post_v4_execute.py', OLD / 'post_v4_follower.py',
                     ROOT / 'PREPARATION_COMPLETE.json', ROOT / 'RTX_EXPORT_VERIFIED.json'] + manifest_files
            files += [p for p in (ROOT / 'data_utilities').iterdir() if p.is_file()]
            files += [CO / name for name in read(CO / 'RUNTIME_MANIFEST.json')['runtime_hashes']]
            files += [CO / 'RUNTIME_MANIFEST.json']
            save(ROOT / 'EXECUTION_FREEZE.json', dict(utc=now(), files={str(p): sha(p) for p in files}))
            s['phase'] = 'SETTLING_BEFORE_TIMINGS'
            update(s)
            support.readiness('after_all_preparation_and_preflight', baseline)
            s['phase'] = 'SCANNET_FULL_TIMINGS'
            update(s)
            for j in s['jobs']:
                if j['status'] != 'queued':
                    continue
                verify_freeze()
                manifest = ROOT / 'manifests' / (j['id'] + '.json')
                out = ROOT / 'results' / j['id']
                assert not out.exists(), 'Preserve partial/full results; no repeat'
                argv = [sys.executable, str(queue_adapter), '--inherited-workload-lock-fd', str(workload.fileno()),
                        '--runtime', str(CO), '--manifest', str(manifest), '--output', str(out), '--power-mode', 'MAXN']
                j.update(status='running', output=str(out / j['id']))
                update(s)
                if child(argv, ROOT / 'logs' / (j['id'] + '.log'), workload.fileno(), s) != 0:
                    j.update(status='blocked', reason='Serial queue failed; original output and log preserved')
                    for other in s['jobs']:
                        if other['scene'] == j['scene'] and other['status'] == 'queued':
                            other.update(status='blocked', reason='Affected scene stopped after ' + j['id'])
                    update(s)
                    continue
                assert read(out / (j['id'] + '.done.json'))['returncode'] == 0 and (out / 'queue_complete.json').is_file()
                result = support.verify_output(j, out / j['id'])
                j.update(status='done', summary=result, config_sha256=sha(out / j['id'] / 'config.json'), source_manifest_sha256=sha(out / j['id'] / 'source_manifest.json'))
                update(s)
                print('FULL_JOB_VERIFIED', j['id'], flush=True)
            s['phase'] = 'COMPLETE_ADMITTED_SCANNET_SERIES'
            update(s)
            save(ROOT / 'SCANNET_COMPLETE.json', dict(utc=now(), major_admitted_series_terminal=True,
                 counts=s['counts'], ALL_REQUIRED_MAXN_COMPLETE=s['ALL_REQUIRED_MAXN_COMPLETE'], power=support.maxn(), fresh_idle=idle()))
            print('SCANNET_MAJOR_SERIES_TERMINAL', s['counts'], flush=True)
        except Exception as e:
            s.update(phase='STOPPED', error=repr(e))
            update(s)
            save(ROOT / 'STOPPED.json', dict(utc=now(), error=repr(e), traceback=traceback.format_exc(), major_admitted_series_terminal=False))
            raise


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--prepare-scene':
        preparation_worker(sys.argv[2])
    else:
        main()
