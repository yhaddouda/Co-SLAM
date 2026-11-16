"""Lightweight metadata inventory only; no CSV scans, compression or CUDA imports."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT.parent
CO = EXPERIMENTS / 'orin_coslam_runtime_v1'
PAPER = EXPERIMENTS / 'orin_paper_night_v2'
RUNTIME = CO / 'runtime'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ROOT.mkdir(exist_ok=True)
    manifest = read(RUNTIME / 'RUNTIME_MANIFEST.json')
    manifest_hash = sha(RUNTIME / 'RUNTIME_MANIFEST.json')
    historical = read(PAPER / 'PAPER_NIGHT_STATUS.json')
    records = {j['output']: j for b in historical['blocks'] for j in b['jobs'] if j.get('output')}
    # Bounded result trees only. Never walk prepared ScanNet images or raw datasets.
    roots = [CO / 'results', PAPER / 'results']
    summaries = []
    for base in roots:
        summaries.extend(base.rglob('summary.json'))
    jobs = []
    for path in sorted(summaries):
        summary = read(path)
        if summary.get('condition') not in ['DP', 'MC_DP']:
            continue
        out = path.parent.resolve()
        cfg = read(out / 'config.json')
        source = read(out / 'source_manifest.json')
        done = out.parent / (out.name + '.done.json')
        launch = out.parent / (out.name + '.launch.json')
        identity = out.parent / 'queue_identity.json'
        complete = out.parent / 'queue_complete.json'
        argv = read(launch)
        is_full = summary['full_sequence'] and not summary['diagnostic']
        record = records.get(str(out))
        expected = record['expected_frames'] if record else 50
        assert summary['frames'] == expected
        assert summary['initial_power'] == summary['final_power'] == 'NV Power Mode: MAXN\n0'
        assert summary['finite_parameters_and_poses'] and summary['warmup_frames'] == 20
        assert summary['quality'] is False and cfg['round2']['engineering'] is False
        assert cfg['round2']['sampler'] == 'support' and cfg['round2']['margin_scene_units'] == .15 and cfg['round2']['anchor_every'] == 3
        assert cfg['grid']['hash_size'] == 16 and cfg['mapping']['n_pixels'] == .05
        assert cfg['grid']['hash'] == ('Morton' if summary['condition'] == 'MC_DP' else 'CoherentPrime')
        assert cfg['grid']['morton_sort'] == (summary['condition'] == 'MC_DP')
        assert source == manifest and sha(out / 'source_manifest.json') == manifest_hash
        assert read(done)['returncode'] == 0
        assert read(identity)['runtime_manifest_sha256'] == manifest_hash
        assert read(complete)['identity'] == read(identity)
        assert argv[argv.index('--frames') + 1] == ('0' if is_full else '50')
        assert '--no-iteration-events' not in argv and '--quality' not in argv and '--export-mesh' not in argv
        if record:
            assert record['status'] == 'done' and record['source_identity'] == manifest_hash
            assert record['actual_config_sha256'] == sha(out / 'config.json')
        files = []
        for p in sorted(out.iterdir()):
            if p.is_file():
                files.append(dict(path=str(p), bytes=p.stat().st_size))
        for name in ['calls.csv', 'iterations.csv', 'frames.csv', 'queries.csv', 'frame_iteration_sums.csv']:
            assert (out / name).is_file() and (out / name).stat().st_size > 0
        jobs.append(dict(id=out.name, system='Co-SLAM', dataset=summary['dataset'], scene=summary['scene'],
                         condition=summary['condition'], seed=summary['seed'], frames=summary['frames'],
                         expected_frames=expected, full_run=is_full, diagnostic=summary['diagnostic'],
                         status='RECORDED_FULL_SUCCESS' if is_full else 'RECORDED_DIAGNOSTIC_SUCCESS',
                         finite_parameters_and_poses=True, power_mode='MAXN', warmup_frames=20,
                         hash_size=16, hash=cfg['grid']['hash'], morton_sort=cfg['grid']['morton_sort'],
                         round2=cfg['round2'], datadir=cfg['data']['datadir'], output=str(out),
                         config_sha256=sha(out / 'config.json'), source_manifest_sha256=manifest_hash,
                         completion_record=str(done), completion_sha256=sha(done), returncode=0,
                         launch_record=str(launch), launch_sha256=sha(launch), command=argv,
                         queue_identity=str(identity), queue_completion=str(complete), files=files,
                         historical_status_path=str(PAPER / 'PAPER_NIGHT_STATUS.json') if record else None,
                         recorded_query_counts={stage: {k: summary[stage].get(k) for k in ['dense_queries', 'actual_queries', 'queries_pruned_pct']}
                                                for stage in ['tracking', 'bundle_adjustment']}))
    assert len([j for j in jobs if j['full_run']]) == 8
    assert len([j for j in jobs if j['diagnostic']]) == 2
    key_sources = ['round2_sampling.py', 'run_coslam_round2.py', 'coslam.py', 'round2_timing.py',
                   'model/encodings.py', 'model/scene_rep.py', 'model/round2_morton/morton_cuda.cu']
    for name in key_sources:
        assert sha(RUNTIME / name) == manifest['runtime_hashes'][name]
    controls = []
    for b in historical['blocks']:
        for j in b['jobs']:
            if j.get('system') == 'Co-SLAM' and j.get('dataset') in ['Replica', 'Tum'] and j.get('scene') in ['office0', 'fr1_desk'] and j.get('hash_size') == 16 and j.get('condition') in ['BASE', 'MC'] and j.get('status') == 'done':
                controls.append({k: j.get(k) for k in ['id', 'dataset', 'scene', 'condition', 'seed', 'hash_size', 'output', 'source_identity', 'actual_config_sha256']})
    doc = dict(utc=datetime.now(timezone.utc).isoformat(), scope='Existing MAXN Co-SLAM DP/MC_DP results only; no new experiments',
               full_runs=8, diagnostics=2, incomplete_dp_outputs_found=0, search_roots=[str(p) for p in roots],
               verification_now='Small summary/config/source/launch/done/queue metadata and file existence/size checked. Full CSV coverage, anchor behavior, bulk checksums and archive deferred until exclusive idle.',
               public_revision=manifest['public_revision'], runtime=str(RUNTIME), runtime_manifest_sha256=manifest_hash,
               method_source_hashes={n: manifest['runtime_hashes'][n] for n in key_sources},
               method=dict(DP='Buffered interval-support pruning with CoherentPrime hashing',
                           MC_DP='Same pruning combined with Morton hashing and Morton point ordering',
                           margin_scene_units=.15, anchor_every=3,
                           full_support='First/last optimizer iterations and zero-based i % 3 == 0; first-frame mapping dense',
                           keep='Interval lower bound <= depth + scaled truncation + 0.15; first two samples; all missing/nonfinite-depth ray samples',
                           tracking='Support indices cached per fixed tracking batch; fresh BA support for mixed batch',
                           retained='Native lattice/RNG draws, ray batches, losses, pose constraints, native full-feature smoothness; engineering disabled',
                           interpretation='Existing approximate-rendering candidate, not a new implementation or a quality guarantee; negative evidence retained'),
               timing='Both iteration CUDA events and complete tracking/BA wall/CUDA calls; 20-frame warmup with all-call means retained',
               clock_policy='Historical author policy retained existing clocks; full jetson_clocks lock state was not established',
               jobs=jobs, matched_control_references=controls,
               matched_controls_note='Already included in the prior 172-slot RTX export; not counted as additional DP measurements or rerun here',
               expected_export=str(ROOT.parents[1] / 'ORIN_EXISTING_MAXN_DP_RESULTS.zip'),
               bulk_export_status='WAITING_FOR_EXCLUSIVE_IDLE_BOUNDARY')
    (ROOT / 'ORIN_EXISTING_DP_INVENTORY.json').write_text(json.dumps(doc, indent=2) + '\n')
    lines = ['# Existing Orin MAXN DP / MC_DP inventory', '',
             doc['utc'], '', '**Eight recorded successful full runs; two successful 50-frame diagnostics.** All Co-SLAM, H16, MAXN, seeds 0/1 for full runs, 20-frame warmup, engineering/quality/meshes disabled.', '',
             '| Dataset / scene | Method | Seed | Frames | Status | Output |', '|---|---|---:|---:|---|---|']
    for j in jobs:
        lines.append(f"| {j['dataset']} / {j['scene']} | {j['condition']} | {j['seed']} | {j['frames']} | {'Full success' if j['full_run'] else 'Diagnostic only'} | `{j['output']}` |")
    lines += ['', 'Verified now: summary full/diagnostic flags and native frame counts, finite-state flag, successful individual/queue completion, exact config/source identities, launch arguments, MAXN at start/end, and presence of both timing scopes. CSV contents will be independently checked and bulk-hashed at the idle export boundary; no new quality acceptance is inferred.', '',
              f"Public Co-SLAM revision: `{doc['public_revision']}`.", '',
              f"All ten source manifests have SHA256 `{manifest_hash}` and match the unchanged runtime-v1 manifest. Per-run config, completion and launch hashes are in the JSON inventory.", '',
              '| Source | SHA256 |', '|---|---|']
    lines += [f'| `{name}` | `{manifest["runtime_hashes"][name]}` |' for name in key_sources]
    lines += ['', 'DP is the existing buffered interval-support candidate, with margin 0.15 scene units and full support on first/last updates and zero-based iterations divisible by 3. It retains the first two samples and all samples on missing/nonfinite-depth rays. First-frame mapping remains dense. MC_DP adds Morton hashing/point ordering; DP uses CoherentPrime. Native RNG/lattice, losses, pose population, keyframe fraction 0.05 and full feature smoothness remain as recorded. This is the unchanged historical approximate-rendering candidate; no new DP variant or quality claim is introduced.', '',
              'Code: `../orin_coslam_runtime_v1/runtime/round2_sampling.py:45` (support), `:63` (installation), `:91` (full-support schedule); `runtime/run_coslam_round2.py:75` (condition/hash), `:79` (recorded parameters), `:98` (activation). Both timing scopes remain separate; native complete-call CSVs are written after sequence completion.', '',
              'Eight matched BASE/MC controls are referenced in the JSON and were included in `../../ORIN_COMPLETED_MAXN_FOR_RTX_20261020.zip`. Original DP results also remain in the older runtime-v1/paper-night result trees and return archives; the deferred export creates a separate explicit DP evidence bundle without replacing any prior archive.', '',
              'Export target: `../../ORIN_EXISTING_MAXN_DP_RESULTS.zip`. At inventory time export is deferred; see EXPORT_STATUS.json and, after success, EXPORT_RECEIPT.json for live state. A detached follower must acquire the existing `/tmp/dasip_round2_gpu0.lock` after the captured current ScanNet parent releases it and exits, then verify no experiment child/program/device user and three fresh idle samples. Job-to-job gaps are not a boundary while this controller retains the lock. No dependency on ALL_REQUIRED_MAXN_COMPLETE, no queue pause or new experiment, no power/clock change, and no email for this routine export.']
    (ROOT / 'ORIN_EXISTING_DP_INVENTORY.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(full_runs=8, diagnostics=2, runtime_manifest_sha256=manifest_hash,
                          source_hashes_checked=len(key_sources), matched_controls=len(controls),
                          inventory=str(ROOT / 'ORIN_EXISTING_DP_INVENTORY.md')), indent=2))


if __name__ == '__main__':
    main()
