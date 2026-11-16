"""Metadata admission plus idle-only combined Co/EC raw export and supplied reports."""
import zipfile
import sys
from local_support import *


def series_paths(system):
    return (ROOT / 'results' / ('co_maxn' if system == 'CO' else 'ec_maxn'),
            ROOT / ('co_bundle' if system == 'CO' else 'bundle') / 'runtime',
            ROOT / 'manifests' / f'{system}_MAXN_SINGLE_SEED.json')


def validate_series(system):
    queue, runtime, manifest_path = series_paths(system)
    manifest = read(manifest_path)
    count = 3 if system == 'CO' else 2
    state = read(queue / 'status.json')
    end = read(queue / 'series_complete.json')
    assert state['status'] == end['status'] == 'COMPLETE'
    assert state['completed'] == state['diagnostic_complete'] == count and state['failed'] == 0
    assert end['gpu_idle'] is True
    identity = read(queue / 'identity.json')
    assert identity['manifest_sha256'] == sha(manifest_path)
    assert identity['runtime_manifest_sha256'] == sha(runtime / 'RUNTIME_MANIFEST.json')
    workloads = {w['id']: w for w in read(runtime / 'PROTOCOL.json')['workloads']}
    records = []
    for job in manifest['jobs']:
        folder = queue / job['id']
        summary = read(folder / 'summary.json')
        work = workloads[job['workload']]
        assert job['seed'] == summary['seed'] == 0
        assert summary['workload'] == job['workload'] and summary['condition'] == 'BASE'
        assert summary['full_sequence'] and not summary['diagnostic'] and summary['finite_parameters_and_poses']
        assert summary['frames'] == summary['native_frames'] == work['expected_native_frames']
        assert summary['selected_calls'] == {'tracking': 300, 'bundle_adjustment': 60}
        assert summary['initial_power'] == summary['final_power'] == 'NV Power Mode: MAXN\n0'
        assert summary['source_manifest_sha256'] == sha(runtime / 'RUNTIME_MANIFEST.json') == sha(folder / 'source_manifest.json')
        assert summary['protocol_sha256'] == sha(runtime / 'PROTOCOL.json')
        assert not summary['quality'] and not summary['export_mesh']
        assert read(folder / 'selection.json') == work
        assert read(queue / (job['id'] + '.done.json'))['summary_sha256'] == sha(folder / 'summary.json')
        cfg = read(folder / 'config.json')
        if system == 'CO':
            assert cfg['grid']['hash_size'] == work['hash_size'] and cfg['grid']['hash'] == 'CoherentPrime'
            assert cfg['round2']['sampler'] == 'dense' and cfg['round2']['engineering'] is False
            assert cfg['grid']['morton_sort'] is False
            assert cfg['data']['datadir'] == job['datadir']
        else:
            assert cfg['HashGrid']['hash_size'] == work['hash_size']
            assert cfg['data']['input_folder'] == job['datadir']
            assert cfg['tracking']['iters'] == 8 and cfg['mapping']['iters'] == 10 and cfg['mapping']['every_frame'] == 5
        diagnostic = queue / (job['id'] + '_diagnostic')
        ds = read(diagnostic / 'summary.json')
        assert ds['diagnostic'] and ds['frames'] == 50 and ds['finite_parameters_and_poses']
        assert read(diagnostic / 'fixed_state_validation/PASSED.json')['status'] == 'PASSED'
        assert read(queue / (job['id'] + '_diagnostic.done.json'))['summary_sha256'] == sha(diagnostic / 'summary.json')
        records.append(dict(id=job['id'], full_frames=summary['frames'], seed=0,
                            summary_sha256=sha(folder / 'summary.json'), source_manifest_sha256=summary['source_manifest_sha256']))
    return records


def package(config, success):
    frozen_sources_ok()
    idle = fresh_idle_three()
    failures = []
    if success:
        for system in ['CO', 'EC']:
            validate_series(system)
            helper = 'report_component_profiles.py' if system == 'CO' else 'report_ec_component_profiles.py'
            cmd = [sys.executable, '-B', str(ROOT / 'bundle' / helper), '--base', str(ROOT),
                   '--active', str(ROOT / f'{system}_COMPONENT_PROFILE_ACTIVE.json')]
            with (ROOT / f'{system}_report.log').open('x') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                failures.append(f'{system} supplied raw-partition/report audit failed; exact error retained in {system}_report.log')
            else:
                directory = 'component_profile_analysis_v1' if system == 'CO' else 'ec_component_profile_analysis_v1'
                report = read(ROOT / 'results' / directory / 'SUMMARY.json')
                assert report['completed'] == report['total'] == (3 if system == 'CO' else 2)
                assert all(v['sd_ms'] is None for g in report['groups'] for v in g['components'].values()), 'Single-seed SD must remain absent'
        success = not failures
    counts = {}
    for system in ['CO', 'EC']:
        queue, _, _ = series_paths(system)
        s = read(queue / 'status.json') if (queue / 'status.json').exists() else {}
        counts[system] = dict(full_completed=s.get('completed', 0), full_required=3 if system == 'CO' else 2,
                              diagnostics_completed=s.get('diagnostic_complete', 0), diagnostics_required=3 if system == 'CO' else 2,
                              status=s.get('status', 'NOT_STARTED'), failed=s.get('failed', 0))
    completion = dict(utc=now(), status='COMPLETE' if success else 'FAILED', systems=counts,
        full_profiles_completed=sum(c['full_completed'] for c in counts.values()), full_profiles_required=5,
        diagnostics_completed=sum(c['diagnostics_completed'] for c in counts.values()), diagnostics_required=5,
        benchmark_slots_completed=292, combined_full_slots_completed=292 + sum(c['full_completed'] for c in counts.values()),
        combined_full_slots_required=297, ALL_REQUIRED_MAXN_COMPLETE=success,
        seed_policy='One run, seed 0, per Orin workload; no across-seed SD', EC_ENG='excluded',
        mode='MAXN', failures=failures, fresh_idle=idle, manual_15W_switch_only=True)
    save(ROOT / 'PROFILING_COMPLETE.json', completion)
    target = TOP / ('ORIN_COMPONENT_PROFILE_MAXN_RESULTS.zip' if success else 'ORIN_COMPONENT_PROFILE_MAXN_FAILURE_EVIDENCE.zip')
    temporary = target.with_suffix('.zip.tmp')
    assert not target.exists() and not temporary.exists(), 'Never overwrite prior profiling results'
    hashes, mapping = {}, {}
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ROOT.rglob('*')):
            if not path.is_file() or path.is_symlink() or any(p in ['__pycache__', 'ckpts', 'mesh', 'targetPlot'] for p in path.parts):
                continue
            if path.suffix in ['.zip', '.tmp', '.so', '.pt', '.pth', '.ply', '.sens', '.png'] or path.name.startswith('EMAIL_'):
                continue
            name = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            archive.writestr(name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
            mapping[name] = str(path)
        archive.writestr('SHA256SUMS.json', json.dumps(hashes, indent=2) + '\n')
        archive.writestr('PATH_MAPPING.json', json.dumps(mapping, indent=2) + '\n')
        archive.writestr('RETURN_MANIFEST.json', json.dumps(dict(utc=now(), files=hashes, completion=completion,
                          prior_exports=read(ROOT / 'UPSTREAM_VERIFIED.json')['exports']), indent=2) + '\n')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    temporary.replace(target)
    digest = sha(target)
    target.with_suffix('.zip.sha256').write_text(digest + '  ' + target.name + '\n')
    receipt = dict(utc=now(), status='EXPORTED_AND_VERIFIED', campaign_success=success,
                   path=str(target), sha256=digest, bytes=target.stat().st_size, entries_verified=len(hashes), completion=completion)
    save(ROOT / 'EXPORT_COMPLETE.json', receipt)
    return receipt
