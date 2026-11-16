#!/usr/bin/env python3
"""Fill the previously declared MC+DP quality/timing gap without repeating controls."""
import csv
import json
from pathlib import Path
from checked_reconstruction import save_json, sha


def main():
    base = Path(__file__).resolve().parents[1]
    runtime = base / 'experiments/coslam_reference_runtime_v1'
    frozen = json.loads((runtime / 'RUNTIME_MANIFEST.json').read_text())
    for name, expected in frozen['runtime_hashes'].items():
        assert sha(runtime / name) == expected, name
    manifest = base / 'experiments/manifests/rtx_composition_all_scenes.json'
    assert not manifest.exists(), 'Preserve the delivered manifest'
    controls = {}
    for filename in ('results/dp_all_scenes_analysis/per_run.csv', 'results/morton_live_verified_analysis/per_run.csv'):
        for row in csv.DictReader((base / filename).open()):
            directory = Path(row['queue']) / row['id']
            cfg = json.loads((directory / 'config.json').read_text())
            if row['dataset'] == 'Replica' and cfg['grid']['hash_size'] != 16:
                continue
            key = row['dataset'], row['scene'], int(row['seed']), row['condition']
            if key in controls:
                assert controls[key] == directory
            controls[key] = directory
    jobs, scores = [], []
    elapsed = 0.
    keys = sorted({key[:2] for key in controls})
    assert len(keys) == 24
    output = base / 'results/composition_v1'
    for dataset, scene in keys:
        for seed in (0, 1, 2):
            refs = {c: controls[(dataset, scene, seed, c)] for c in ('BASE', 'MC', 'DP')}
            cfg = json.loads((refs['DP'] / 'config.json').read_text())
            summary = json.loads((refs['DP'] / 'summary.json').read_text())
            for directory in refs.values():
                assert sha(directory / 'source_manifest.json') == sha(runtime / 'RUNTIME_MANIFEST.json')
                assert (directory.parent / (directory.name + '.done.json')).is_file()
            job = dict(id=f'{dataset}_{scene}_MC_DP_seed{seed}', dataset=dataset, scene=scene,
                       condition='MC_DP', seed=seed, hash_size=cfg['grid']['hash_size'],
                       datadir=cfg['data']['datadir'], quality=True,
                       export_mesh=dataset in ('Replica', 'Synthetic'), frames=0,
                       expected_native_frames=summary['frames'],
                       controls={c: str(path) for c, path in refs.items()})
            assert Path(job['datadir']).is_dir()
            jobs.append(job)
            done = json.loads((refs['DP'].parent / (refs['DP'].name + '.done.json')).read_text())
            elapsed += done['elapsed_seconds']
            if job['export_mesh']:
                directory = output / job['id']
                asset = base / f'experiments/reconstruction_assets_v1/{dataset}_{scene}.json'
                assert asset.is_file()
                scores.append(dict(id=f'Co_{dataset}_{scene}_MC_DP_H{job["hash_size"]}_seed{seed}',
                                   system='Co-SLAM', dataset=dataset, scene=scene, condition='MC_DP',
                                   hash_size=job['hash_size'], seed=seed,
                                   mesh=str(directory / f'artifacts/mesh_track{summary["frames"]-1}.ply'),
                                   summary=str(directory / 'summary.json'),
                                   source_manifest=str(directory / 'source_manifest.json'),
                                   source_manifest_sha256=sha(runtime / 'RUNTIME_MANIFEST.json'),
                                   upstream_complete=str(output / (job['id'] + '.done.json')),
                                   asset=str(asset), asset_sha256=sha(asset)))
    assert len(jobs) == 72 and len(scores) == 45
    save_json(manifest, dict(purpose='Close the agreed RTX MC+DP composition gap: all 24 native-capacity scenes, three paired seeds, quality/full-call and iteration timings; reuse existing BASE/MC/DP controls',
                            runtime_manifest_sha256=sha(runtime / 'RUNTIME_MANIFEST.json'), jobs=jobs))
    save_json(base / 'experiments/manifests/rtx_reconstruction_composition.json',
              dict(purpose='Score all 45 new MC+DP Replica/Synthetic meshes with the unchanged checked v2 evaluator', jobs=scores))
    save_json(base / 'results/RTX_COMPOSITION_PREPARATION.json',
              dict(training_jobs=72, mesh_scores=45, repeated_controls=0,
                   existing_dp_process_workload_hours=elapsed / 3600,
                   estimate_scope='Matched DP process durations including mesh export where requested; MC+DP may differ',
                   runtime_manifest_sha256=sha(runtime / 'RUNTIME_MANIFEST.json')))
    print('PREPARED 72 MC_DP full runs + 45 mesh scores; matched DP process workload hours:', round(elapsed / 3600, 2))


if __name__ == '__main__':
    main()
