#!/usr/bin/env python3
"""CPU-only package/identity validation; never launches or imports a GPU driver."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    for name, expected in json.loads((root / 'SHA256SUMS.json').read_text()).items():
        path = (root / name).resolve()
        assert path.is_relative_to(root), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    plan = json.loads((root / 'REUSE_AND_CONTROLS.json').read_text())
    full = json.loads((root / 'manifests/orin_dp_mc_dp_all_targets.json').read_text())['jobs']
    missing = json.loads((root / 'manifests/orin_dp_mc_dp_missing_maxn.json').read_text())['jobs']
    ids = {job['id'] for job in full}
    missing_ids = {job['id'] for job in missing}
    reuse_ids = {job['id'] for job in plan['existing_full_runs']}
    assert len(full) == len(ids) == 96
    assert len(missing) == len(missing_ids) == 88
    assert len(reuse_ids) == 8 and not (missing_ids & reuse_ids)
    assert missing_ids | reuse_ids == ids
    scene_h = {(j['dataset'], j['scene']): j['hash_size'] for j in full}
    assert len(scene_h) == 24
    for job in full:
        assert job['seed'] in (0, 1) and job['condition'] in ('DP', 'MC_DP')
        assert job['frames'] == 0 and job['expected_native_frames'] > 20
        assert job['quality'] is False and job['export_mesh'] is False
        assert not job.get('engineering', False)
        assert job['id'] == f"{job['dataset']}_{job['scene']}_{job['condition']}_H{job['hash_size']}_seed{job['seed']}"
        for condition in ('BASE', 'MC'):
            assert job['control_ids'][condition] == f"{job['dataset']}_{job['scene']}_{condition}_H{job['hash_size']}_seed{job['seed']}"
    for (dataset, scene), capacity in scene_h.items():
        jobs = [j for j in full if (j['dataset'], j['scene']) == (dataset, scene)]
        assert len(jobs) == 4 and len({(j['condition'], j['seed']) for j in jobs}) == 4
        assert {j['hash_size'] for j in jobs} == {capacity}
        if dataset != 'ScanNet':
            assert capacity == 16
    with zipfile.ZipFile(root / 'orin_coslam_runtime_v1.zip') as archive:
        assert archive.testzip() is None
        hashes = json.loads(archive.read('SHA256SUMS.json'))
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        source = archive.read('runtime/RUNTIME_MANIFEST.json')
        assert hashlib.sha256(source).hexdigest() == plan['runtime_manifest_sha256']
    print('PACKAGE_AND_TARGETS_OK: 96 target, 8 inventory-backed existing, 88 missing')
    print('Runtime v1 identity verified. No local data, raw-result audit or remote launch asserted.')


if __name__ == '__main__':
    main()
