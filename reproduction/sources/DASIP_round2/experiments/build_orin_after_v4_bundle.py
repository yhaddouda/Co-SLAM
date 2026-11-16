#!/usr/bin/env python3
"""Package the post-v4 handoff; never launch or modify an Orin job."""
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    base = Path(__file__).resolve().parents[1]
    archive = base / 'bundles/to_orin/orin_after_v4_maxn_v2.zip'
    if archive.exists():
        raise FileExistsError(archive)
    entries = {}

    def add(name, content):
        assert name not in entries and '..' not in Path(name).parts
        entries[name] = content.encode() if isinstance(content, str) else content

    documents = base / 'experiments/orin_after_v4'
    for name in ('README.md', 'DATA_PREPARATION.md'):
        add(name, (documents / name).read_bytes())
    jobs_doc = json.loads((base / 'experiments/manifests/orin_continuous_coslam.json').read_text())
    jobs = [job for job in jobs_doc['jobs'] if job['dataset'] == 'ScanNet']
    inventory = json.loads((base / 'experiments/ALL_CONFIGURED_NONREPLICA_INVENTORY.json').read_text())
    scenes = [scene for scene in inventory['scenes'] if scene['dataset'] == 'ScanNet']
    assert len(jobs) == 24 and len(scenes) == 6
    expected = {(scene['scene'], seed, condition) for scene in scenes
                for seed in (0, 1) for condition in ('BASE', 'MC')}
    assert {(job['scene'], job['seed'], job['condition']) for job in jobs} == expected
    assert all(not job['quality'] and not job['export_mesh'] and job['frames'] == 0 for job in jobs)
    for scene in scenes:
        assert all(job['hash_size'] == scene['native_hash_size'] for job in jobs if job['scene'] == scene['scene'])
        scene['raw_scan_id'] = Path(scene['datadir']).name
    add('SCANNET_SCENES.json', json.dumps(dict(
        purpose='Expected full RTX input coverage for the same six configured ScanNet scenes',
        scenes=scenes, local_paths='RTX paths are provenance, not assumed Orin paths'), indent=2) + '\n')
    jobs_doc.update(purpose='Exact 24-job ScanNet subset of v4; run only after normal ready-backlog exhaustion and data preparation', jobs=jobs)
    add('manifests/orin_scannet_native.json', json.dumps(jobs_doc, indent=2) + '\n')
    add('manifests/orin_pending_engineering_coslam_only.json', (base / 'experiments/manifests/orin_pending_engineering_coslam_only.json').read_bytes())
    add('ENGINEERING_SCOPE_OVERRIDE.txt', (base / 'docs/protocols/ENGINEERING_SCOPE_OVERRIDE.txt').read_bytes())
    repo = base.parent
    add('download/download-scannet.py', (repo / 'scripts/download/download-scannet.py').read_bytes())

    runtime = base / 'experiments/coslam_engineering_runtime_v1'
    manifest = json.loads((runtime / 'RUNTIME_MANIFEST.json').read_text())
    for name, expected_hash in manifest['runtime_hashes'].items():
        content = (runtime / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_hash, name
        assert Path(name).suffix in ('.py', '.yaml', '.txt', '.cu'), name
        add('coslam_engineering_reference/' + name, content)
    add('coslam_engineering_reference/RUNTIME_MANIFEST.json', (runtime / 'RUNTIME_MANIFEST.json').read_bytes())
    for name in ('fixed_state.json', 'status.json'):
        add('engineering_evidence/RTX_' + name, (base / 'results/engineering_preflight_v1' / name).read_bytes())
    add('engineering_evidence/RTX_REPORT.md', (base / 'results/engineering_v1_analysis/REPORT.md').read_bytes())
    add('run_manifest_queue.py', (base / 'experiments/run_manifest_queue.py').read_bytes())
    add('BUILD_MANIFEST.json', json.dumps(dict(
        built_for='19 October 2026 author request; continuation after existing MAXN ready backlog',
        modifies_existing_v4_runtime=False, remote_install_or_launch_performed=False,
        automatic_follower_must_be_installed_and_verified_on_orin=True,
        native_scannet_jobs=24, conditional_engineering_slots=2,
        engineering_reference='RTX Co only; not an Orin or EC equivalence pass',
        extractor_included=False, datasets_included=False,
        wait_on='verified normal exhaustion of ready backlog, then exclusive ownership; not 22:00 or ALL_REQUIRED_MAXN_COMPLETE'
    ), indent=2) + '\n')
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sorted(entries.items())}
    add('SHA256SUMS.json', json.dumps(hashes, indent=2) + '\n')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for name, content in sorted(entries.items()):
            output.writestr(name, content)
    with zipfile.ZipFile(archive) as checked:
        assert checked.testzip() is None
        assert all(hashlib.sha256(checked.read(name)).hexdigest() == value for name, value in hashes.items())
    print(json.dumps(dict(archive=str(archive), files=len(entries), bytes=archive.stat().st_size,
                          sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                          native_scannet_jobs=24, remote_launched=False), indent=2))


if __name__ == '__main__':
    main()
