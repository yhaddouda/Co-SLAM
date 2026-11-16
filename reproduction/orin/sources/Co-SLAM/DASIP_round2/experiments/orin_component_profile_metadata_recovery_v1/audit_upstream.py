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


