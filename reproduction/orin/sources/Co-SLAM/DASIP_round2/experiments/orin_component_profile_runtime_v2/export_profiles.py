"""Idle-only audit and incremental profiling export, invoked by the durable parent."""
import json
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile
from local_support import *


def verify_archive(path, expected, check_entries=True):
    path = Path(path)
    assert path.is_file() and sha(path) == expected, 'Archive digest mismatch: ' + str(path)
    sidecar = path.with_suffix('.zip.sha256')
    if sidecar.exists():
        assert sidecar.read_text().split()[0] == expected
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, 'Archive CRC failure: ' + str(path)
        checked = 0
        if check_entries and 'RETURN_MANIFEST.json' in archive.namelist():
            manifest = json.loads(archive.read('RETURN_MANIFEST.json'))
            for name, digest in manifest.get('files', {}).items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
                checked += 1
    return dict(path=str(path), sha256=expected, entries_checked=checked, status='VERIFIED')


def audit_upstream(config):
    """Only called after all successful terminal/export gates and exclusive idle."""
    frozen_sources_ok()
    s = read(DP / 'DP_COMPLETION_STATUS.json')
    expected = config['expected_dp_jobs']
    assert len(s['jobs']) == len(expected) == 96
    assert {j['id'] for j in s['jobs']} == {j['id'] for j in expected}
    new_ids = {j['id'] for j in config['expected_missing_jobs']}
    verified = []
    for job, target in zip(s['jobs'], expected):
        for key in ['id', 'dataset', 'scene', 'condition', 'seed', 'hash_size', 'expected_native_frames']:
            assert job[key] == target[key], (job['id'], key)
        assert job['status'] == ('done' if job['id'] in new_ids else 'reused')
        output = Path(job['output'])
        summary = read(output / 'summary.json')
        done = Path(job['completion_record'])
        assert read(done)['returncode'] == 0
        assert summary['full_sequence'] and not summary['diagnostic'] and summary['finite_parameters_and_poses']
        assert summary['frames'] == job['expected_native_frames']
        for key in ['dataset', 'scene', 'condition', 'seed']:
            assert summary[key] == job[key]
        assert summary['initial_power'] == summary['final_power'] == 'NV Power Mode: MAXN\n0'
        assert not summary['quality'] and summary['warmup_frames'] == 20
        assert sha(output / 'source_manifest.json') == config['native_runtime_manifest_sha256']
        assert sha(output / 'config.json') == job['config_sha256']
        verified.append(dict(id=job['id'], output=str(output), done=str(done),
                             summary_sha256=sha(output / 'summary.json')))
    receipt = read(DP / 'RETURN_BUNDLE_VERIFIED.json')
    exports = [verify_archive(receipt['path'], receipt['sha256'])]
    old = read(OLD_EXPORT / 'EXPORT_RECEIPT.json')
    assert old['status'] == 'EXPORTED_AND_VERIFIED' and old['full_runs'] == 8 and old['diagnostics'] == 2
    exports.append(verify_archive(old['archive'], old['sha256']))
    scan_zip = TOP / 'ORIN_SCANNET_MAXN_RESULTS.zip'
    exports.append(verify_archive(scan_zip, scan_zip.with_suffix('.zip.sha256').read_text().split()[0]))
    earlier = read(SCAN / 'RTX_EXPORT_VERIFIED.json')
    exports.append(verify_archive(earlier['path'], earlier['sha256']))
    for control in config['native_profile_controls']:
        assert sha(control['config_path']) == control['config_sha256']
        assert sha(control['summary_path']) == control['summary_sha256']
        assert read(control['completion_path'])['returncode'] == 0
        assert read(control['config_path'])['data']['datadir'] == control['datadir']
        assert Path(control['datadir']).is_dir()
    proof = dict(utc=now(), status='VERIFIED', exact_dp_identities=96, new_completed=88, reused=8,
                 native_engineering_completed=196, benchmark_slots_complete=292,
                 profiling_jobs_outside_benchmark_envelope=True, jobs=verified, exports=exports,
                 native_runtime_manifest_sha256=config['native_runtime_manifest_sha256'])
    save(ROOT / 'UPSTREAM_VERIFIED.json', proof)
    return proof


def export_profiles(config, success):
    """Caller holds the shared workload lock and has verified a dead profile parent."""
    frozen_sources_ok()
    samples = fresh_idle_three()
    result = ROOT / 'results/maxn'
    state = read(result / 'status.json')
    manifest = read(ROOT / 'manifests/MAXN_RESOLVED.json')
    audit_error = None
    if success:
        assert state['status'] == 'COMPLETE' and state['completed'] == 6 and state['diagnostic_complete'] == 3 and state['failed'] == 0
        end = read(result / 'series_complete.json')
        assert end['status'] == 'COMPLETE' and end['gpu_idle'] is True
        protocol = read(ROOT / 'bundle/runtime/PROTOCOL.json')
        work = {w['id']: w for w in protocol['workloads']}
        for job in manifest['jobs']:
            folder = result / job['id']
            summary = read(folder / 'summary.json')
            assert summary['frames'] == summary['native_frames'] == work[job['workload']]['expected_native_frames']
            assert summary['initial_power'] == summary['final_power'] == 'NV Power Mode: MAXN\n0'
            assert read(folder / 'selection.json') == work[job['workload']]
            assert summary['protocol_sha256'] == sha(ROOT / 'bundle/runtime/PROTOCOL.json')
            assert summary['quality'] is False and summary['export_mesh'] is False
        for job in manifest['jobs'][::2]:
            folder = result / (job['id'] + '_diagnostic')
            summary = read(folder / 'summary.json')
            assert summary['frames'] == 50 and summary['diagnostic']
            assert read(folder / 'fixed_state_validation/PASSED.json')['status'] == 'PASSED'
        command = [sys.executable, '-B', str(ROOT / 'bundle/report_component_profiles.py'),
                   '--base', str(ROOT), '--active', str(ROOT / 'COMPONENT_PROFILE_ACTIVE.json')]
        with (ROOT / 'report.log').open('x') as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            success = False
            audit_error = 'Supplied report/partition audit failed; exact error is preserved in report.log'
        else:
            report = read(ROOT / 'results/component_profile_analysis_v1/SUMMARY.json')
            assert report['completed'] == report['total'] == 6
    completion = dict(utc=now(), status='COMPLETE' if success else 'FAILED',
                      full_completed=state['completed'], full_required=6,
                      diagnostics_completed=state['diagnostic_complete'], diagnostics_required=3,
                      failed=state['failed'] + int(bool(audit_error)), audit_error=audit_error,
                      benchmark_slots_completed=292, EC_ENG='excluded',
                      ALL_REQUIRED_MAXN_COMPLETE=success, mode='MAXN',
                      scope='292 unchanged benchmark slots plus separate six BASE profiles and three diagnostics',
                      manual_15W_switch_only=True, idle_samples=samples)
    save(ROOT / 'PROFILING_COMPLETE.json', completion)
    name = 'ORIN_COMPONENT_PROFILE_MAXN_RESULTS.zip' if success else 'ORIN_COMPONENT_PROFILE_MAXN_FAILURE_EVIDENCE.zip'
    target = TOP / name
    temporary = target.with_suffix('.zip.tmp')
    assert not target.exists() and not temporary.exists(), 'Never overwrite an existing export'
    mapping, hashes = {}, {}
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts:
                continue
            if path.suffix in ['.zip', '.tmp', '.so', '.pt', '.pth', '.ply', '.sens']:
                continue
            if path.name.startswith('EMAIL_'):
                continue
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            archive.writestr(relative, data)
            hashes[relative] = hashlib.sha256(data).hexdigest()
            mapping[relative] = str(path)
        archive.writestr('SHA256SUMS.json', json.dumps(hashes, indent=2) + '\n')
        archive.writestr('PATH_MAPPING.json', json.dumps(mapping, indent=2) + '\n')
        archive.writestr('RETURN_MANIFEST.json', json.dumps(dict(utc=now(), files=hashes,
                          completion=completion, prior_exports=read(ROOT / 'UPSTREAM_VERIFIED.json')['exports']), indent=2) + '\n')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for path, digest in hashes.items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    temporary.replace(target)
    digest = sha(target)
    target.with_suffix('.zip.sha256').write_text(digest + '  ' + target.name + '\n')
    receipt = dict(utc=now(), status='EXPORTED_AND_VERIFIED', campaign_success=success,
                   path=str(target), sha256=digest, bytes=target.stat().st_size,
                   entries_verified=len(hashes), completion=completion)
    save(ROOT / 'EXPORT_COMPLETE.json', receipt)
    return receipt
