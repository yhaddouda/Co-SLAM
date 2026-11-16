#!/usr/bin/env python3
"""Build the authorized missing native-capacity DP/MC_DP handoff, without launch."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


SOURCE_SHA = '9786e843d6856dcc443fb08ebb9646a5d759384e19ab3456552b51e2281e616d'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    base = Path(__file__).resolve().parents[1]
    package = base / 'experiments/orin_dp_completion'
    archive_path = base / 'bundles/to_orin/orin_dp_completion_maxn_v1.zip'
    if archive_path.exists():
        raise FileExistsError('Preserve already delivered package: ' + str(archive_path))
    native = json.loads((base / 'experiments/manifests/orin_continuous_coslam.json').read_text())['jobs']
    inventory = json.loads((base / 'experiments/ALL_CONFIGURED_NONREPLICA_INVENTORY.json').read_text())
    metadata = {(s['dataset'], s['scene']): s for s in inventory['scenes']}
    scenes = {}
    for job in native:
        if job['dataset'] == 'Replica' and job['hash_size'] != 16:
            continue
        key = job['dataset'], job['scene']
        scenes.setdefault(key, dict(dataset=key[0], scene=key[1], datadir=job['datadir'],
                                    hash_size=job['hash_size'], expected_native_frames=2000 if key[0] == 'Replica' else metadata[key]['frames']))
    assert len(scenes) == 24
    groups = {dataset: sorted([s for s in scenes.values() if s['dataset'] == dataset],
                              key=lambda s: (s['expected_native_frames'], s['scene']))
              for dataset in ('Synthetic', 'Replica', 'Tum', 'ScanNet')}
    ordered = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                ordered.append(group.pop(0))
    targets = []
    for scene in ordered:
        for seed in (0, 1):
            for condition in (('DP', 'MC_DP') if seed == 0 else ('MC_DP', 'DP')):
                prefix = f"{scene['dataset']}_{scene['scene']}"
                targets.append(dict(id=f"{prefix}_{condition}_H{scene['hash_size']}_seed{seed}",
                                    **scene, condition=condition, seed=seed, frames=0,
                                    quality=False, export_mesh=False,
                                    control_ids={c: f"{prefix}_{c}_H{scene['hash_size']}_seed{seed}" for c in ('BASE', 'MC')}))
    existing = []
    remote = '/home/yh279050/work/Co-SLAM/DASIP_round2/experiments/'
    for job in targets:
        key = job['dataset'], job['scene']
        if key not in {('Replica', 'office0'), ('Tum', 'fr1_desk')}:
            continue
        if key[0] == 'Replica' and job['condition'] == 'DP':
            relative = 'orin_coslam_runtime_v1/results/full_maxn/'
        elif key[0] == 'Replica':
            relative = 'orin_paper_night_v2/results/O1_MC/'
        else:
            relative = 'orin_paper_night_v2/results/O5_TUM/'
        output = remote + relative + f"{job['dataset']}_{job['scene']}_{job['condition']}_seed{job['seed']}"
        existing.append(dict(**job, original_output=output, evidence='ORIN_EXISTING_DP_INVENTORY.md; local raw audit/export still required'))
    reused_ids = {job['id'] for job in existing}
    pending = [job for job in targets if job['id'] not in reused_ids]
    assert len(targets) == 96 and len(existing) == 8 and len(pending) == 88
    common = dict(power_mode='MAXN', system='Co-SLAM', warmup_frames=20, seeds=[0, 1],
                  runtime_manifest_sha256=SOURCE_SHA, native_capacities_only=True,
                  data_paths='Hints; resolve from actual matching BASE/MC configs at the idle boundary',
                  purpose='20 October author-authorized full DP/MC_DP continuation after ScanNet and idle export; no method changes')
    dump(package / 'manifests/orin_dp_mc_dp_all_targets.json', dict(**common, jobs=targets))
    dump(package / 'manifests/orin_dp_mc_dp_missing_maxn.json', dict(**common, jobs=pending))
    received = json.loads((base / 'results/orin_maxn_received_20261020/transfer/RTX_IMPORT_MANIFEST.json').read_text())
    control_ids = {value for job in targets for value in job['control_ids'].values()}
    controls = [entry for entry in received['completed_identity_mapping'] if entry['id'] in control_ids]
    assert len(controls) == 72
    dump(package / 'REUSE_AND_CONTROLS.json', dict(
        runtime_manifest_sha256=SOURCE_SHA, target_runs=96, inventoried_existing=8, missing_runs=88,
        existing_full_runs=existing, received_native_controls=controls,
        pending_control_ids=sorted(control_ids - {entry['id'] for entry in controls}),
        reuse_status='Reported full success and identities; raw audit at the existing idle export boundary remains required',
        controller_gate='Actual normal ScanNet-series completion, finished existing export with receipt, no child and exclusive ownership',
        expanded_maxn_envelope=dict(native_and_engineering=196, pruning_and_composition=96, total=292)))
    rows = list(csv.DictReader((base / 'results/orin_maxn_analysis/per_run.csv').open()))
    workload = {}
    for dataset in ('Replica', 'Synthetic', 'Tum'):
        dense = [r for r in rows if r['system'] == 'Co-SLAM' and r['category'] == 'MC'
                 and r['dataset'] == dataset and r['hash_size'] == '16'
                 and (dataset, r['scene']) not in {('Replica', 'office0'), ('Tum', 'fr1_desk')}]
        seconds = [json.loads((base / r['directory'] / 'summary.json').read_text())['slam_work_wall_seconds'] for r in dense]
        workload[dataset] = dict(identities=len(dense), dense_workload_seconds=sum(seconds))
    dump(package / 'BUILD_MANIFEST.json', dict(
        request_date='2026-10-20', remote_installed=False, remote_launched=False,
        target_runs=96, inventory_backed_existing=8, missing_runs=88,
        expected_new_runs_by_dataset={'Replica': 28, 'Synthetic': 28, 'Tum': 8, 'ScanNet': 24},
        diagnostic_runs_reused_as_full=0, h19_pruning_jobs=0, engineering_jobs=0, ec_jobs=0,
        included_runtime='Unchanged original runtime v1 ZIP',
        dense_non_scannet_workload_proxy=workload,
        eta_limitation='Not a DP runtime forecast; omits 24 ScanNet jobs, loading/launch overhead may differ; local timings needed'))
    entries = {name: (package / name).read_bytes() for name in (
        'README.md', 'validate_bundle.py', 'REUSE_AND_CONTROLS.json', 'BUILD_MANIFEST.json',
        'manifests/orin_dp_mc_dp_all_targets.json', 'manifests/orin_dp_mc_dp_missing_maxn.json')}
    for target, source in {
        'ORIN_EXISTING_DP_INVENTORY.md': base / 'docs/orin/ORIN_EXISTING_DP_INVENTORY.md',
        'NOTIFICATION_POLICY.md': base / 'docs/protocols/NOTIFICATION_POLICY.md',
        'orin_coslam_runtime_v1.zip': base / 'bundles/to_orin/orin_coslam_runtime_v1.zip',
        'run_manifest_queue.py': base / 'experiments/run_manifest_queue.py',
    }.items():
        entries[target] = source.read_bytes()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(entries.items())}
    for name, data in entries.items():
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    dump(package / 'SHA256SUMS.json', hashes)
    subprocess.run([sys.executable, str(package / 'validate_bundle.py')], check=True)
    entries['SHA256SUMS.json'] = (package / 'SHA256SUMS.json').read_bytes()
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == expected for name, expected in hashes.items())
    receipt = dict(archive=str(archive_path), bytes=archive_path.stat().st_size,
                   sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                   targets=96, inventory_backed_existing=8, missing=88,
                   remote_installed=False, remote_launched=False)
    dump(base / 'results/ORIN_DP_COMPLETION_BUNDLE.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
