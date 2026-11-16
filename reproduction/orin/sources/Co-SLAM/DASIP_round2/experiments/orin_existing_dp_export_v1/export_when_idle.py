"""Export existing DP evidence only after exclusive host idle. Never runs SLAM."""
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'orin_after_v4_maxn_v1'))
from post_v4_follower import identity, idle


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p = Path(p)
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(value, indent=2) + '\n')
    t.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def status(phase, **kwargs):
    save(ROOT / 'EXPORT_STATUS.json', dict(utc=now(), phase=phase, follower_pid=os.getpid(),
         GPU_work_launched=False, experiments_repeated=0, **kwargs))


def validate_job(j):
    out = Path(j['output'])
    cfg = read(out / 'config.json')
    summary = read(out / 'summary.json')
    assert sha(out / 'config.json') == j['config_sha256']
    assert sha(out / 'source_manifest.json') == j['source_manifest_sha256']
    assert sha(j['completion_record']) == j['completion_sha256'] and read(j['completion_record'])['returncode'] == 0
    assert sha(j['launch_record']) == j['launch_sha256']
    assert summary['full_sequence'] == j['full_run'] and summary['diagnostic'] == j['diagnostic']
    assert summary['frames'] == j['expected_frames'] and summary['finite_parameters_and_poses']
    assert summary['initial_power'] == summary['final_power'] == 'NV Power Mode: MAXN\n0'
    assert not summary['quality'] and not cfg['round2']['engineering']
    n = j['expected_frames']
    with (out / 'calls.csv').open() as f:
        calls = list(csv.DictReader(f))
    with (out / 'iterations.csv').open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['frame_id', 'iteration_type', 'iteration_id', 'forward_cuda_ms', 'backward_cuda_ms', 'full_iteration_cuda_ms']
        events = list(reader)
    with (out / 'queries.csv').open() as f:
        queries = list(csv.DictReader(f))
    stage_records = {}
    for stage, frames, steps in [
        ('tracking', set(range(1, n)), cfg['tracking']['iter']),
        ('bundle_adjustment', set(range(cfg['mapping']['map_every'], n, cfg['mapping']['map_every'])), cfg['mapping']['iters'])]:
        expected = {(frame, i) for frame in frames for i in range(steps)}
        c = [x for x in calls if x['stage'] == stage]
        e = [x for x in events if x['iteration_type'] == stage]
        q = [x for x in queries if x['stage'] == stage]
        assert len(c) == len(frames) and {int(x['frame_id']) for x in c} == frames
        assert len(e) == len(expected) and {(int(x['frame_id']), int(x['iteration_id'])) for x in e} == expected
        assert len(q) == len(expected) and {(int(x['frame_id']), int(x['iteration'])) for x in q} == expected
        pruned = dense_total = actual_total = 0
        for row in q:
            i = int(row['iteration'])
            dense, actual = int(row['dense_queries']), int(row['actual_queries'])
            assert 0 < actual <= dense
            if i == 0 or i == steps - 1 or i % cfg['round2']['anchor_every'] == 0:
                assert actual == dense, (j['id'], stage, row)
            pruned += actual < dense
            dense_total += dense
            actual_total += actual
        assert pruned > 0, 'Recorded DP must actually prune in both stages'
        assert dense_total == summary[stage]['dense_queries'] and actual_total == summary[stage]['actual_queries']
        stage_records[stage] = dict(calls=len(c), iterations=len(e), query_records=len(q),
                                   all_anchors_full=True, pruned_iteration_records=pruned,
                                   dense_queries=dense_total, actual_queries=actual_total)
    assert all(math.isfinite(float(x[k])) and float(x[k]) >= 0 for x in calls for k in ['wall_ms', 'cuda_ms'])
    assert all(math.isfinite(float(x[k])) and float(x[k]) >= 0 for x in events for k in ['forward_cuda_ms', 'backward_cuda_ms', 'full_iteration_cuda_ms'])
    with (out / 'frames.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n and {int(x['frame_id']) for x in rows} == set(range(n))
    with (out / 'frame_iteration_sums.csv').open() as f:
        sums = list(csv.reader(f))
    assert len(sums) == n - 1 and {int(x[0]) for x in sums} == set(range(1, n))
    assert all(len(x) == 4 and all(math.isfinite(float(v)) for v in x) for x in sums)
    return dict(id=j['id'], full_run=j['full_run'], frames=n, status='PASS', stages=stage_records,
                full_frame_coverage=True, recorded_finite_state=True, both_timing_scopes=True)


def export(inventory, cfg):
    runtime = Path(inventory['runtime'])
    manifest = read(runtime / 'RUNTIME_MANIFEST.json')
    assert sha(runtime / 'RUNTIME_MANIFEST.json') == inventory['runtime_manifest_sha256']
    for name, expected in manifest['runtime_hashes'].items():
        assert sha(runtime / name) == expected, name
    checked = []
    for j in inventory['jobs']:
        status('VALIDATING_EXISTING_CSVS_AT_IDLE', job=j['id'])
        checked.append(validate_job(j))
    save(ROOT / 'DP_CSV_VALIDATION.json', dict(utc=now(), status='PASS', full_runs=8, diagnostics=2, jobs=checked))
    note = ('# Existing MAXN DP / MC_DP evidence for RTX\n\n'
            'Eight completed Co-SLAM full runs: Replica office0 (2000 frames), TUM fr1_desk (592 native associated frames), '
            'DP and MC_DP, each seeds 0/1. Two additional office0 seed-0 50-frame diagnostics are separate. '
            'H16, recorded initial/final MAXN, warmup 20, engineering/quality/meshes off. '
            'This archive contains existing measurements only, not new experiments or a new pruning implementation.\n\n'
            'DP is the runtime-v1 buffered interval-support rendering approximation with margin .15, full-support first/last '
            'and zero-based i%3==0 updates. MC_DP composes the same candidate with Morton hash/ordering. '
            'Public revision, exact source/config identities, original paths and controls are in ORIN_EXISTING_DP_INVENTORY.json. '
            'The inventory is a snapshot from the request time; DP_CSV_VALIDATION.json and IDLE_EXPORT_BOUNDARY.json '
            'record the later verified export boundary. Original source/results remain untouched.\n\n'
            'Fresh CSV checks cover full frame and optimizer iteration IDs, finite nonnegative timings, dense anchor updates '
            'and actual pruning in both stages. Both complete-call and iteration CSVs retain their original bytes/formats. '
            'No reconstruction/quality metric is computed. Historical limitations and negative evidence remain valid; '
            'success here means execution/coverage, not quality acceptance.\n\n'
            'Matched BASE/MC controls are referenced rather than duplicated: they are in '
            'ORIN_COMPLETED_MAXN_FOR_RTX_20261020.zip. Import by condition/scene/seed/mode/source/config identity; '
            'do not count the diagnostic runs as full results. No datasets, meshes, compiled extensions or credentials are included.\n')
    (ROOT / 'ARCHIVE_README.md').write_text(note)
    target = Path(cfg['archive'])
    temporary = target.with_suffix('.zip.tmp')
    assert not target.exists() and not temporary.exists(), 'Preserve existing complete/partial exports; no overwrite'
    selection = {}

    def add(path, name):
        path = Path(path)
        assert path.is_file()
        if name in selection:
            assert selection[name] == path
        else:
            selection[name] = path

    for name in ['ORIN_EXISTING_DP_INVENTORY.json', 'ORIN_EXISTING_DP_INVENTORY.md', 'DP_CSV_VALIDATION.json',
                 'IDLE_EXPORT_BOUNDARY.json', 'ARCHIVE_README.md', 'EXPORT_FOLLOWER_CONFIG.json',
                 'EXPORT_FREEZE.json', 'inventory_existing.py', 'export_when_idle.py']:
        add(ROOT / name, name)
    for j in inventory['jobs']:
        prefix = ('full_runs/' if j['full_run'] else 'diagnostics/') + j['id'] + '/'
        for item in j['files']:
            p = Path(item['path'])
            assert p.parent == Path(j['output']) and p.stat().st_size == item['bytes']
            add(p, prefix + p.name)
        for name in [j['id'] + '.done.json', j['id'] + '.launch.json', j['id'] + '.log']:
            p = Path(j['output']).parent / name
            add(p, prefix + 'queue_records/' + name)
        queue_dir = Path(j['output']).parent
        queue_name = queue_dir.relative_to(ROOT.parent).as_posix()
        for name in ['queue_identity.json', 'queue_complete.json', 'progress.json', 'job_manifest.json']:
            if (queue_dir / name).is_file():
                add(queue_dir / name, 'queue_records/' + queue_name + '/' + name)
    for name in list(manifest['runtime_hashes']) + ['RUNTIME_MANIFEST.json']:
        add(runtime / name, 'runtime_source/' + name)
    co = runtime.parent
    paper = ROOT.parent / 'orin_paper_night_v2'
    for p in [co / 'README.md', co / 'SHA256SUMS.json', co / 'run_manifest_queue.py', co / 'verify_handoff.py',
              co / 'run_records/MAXN_ENVIRONMENT.json', co / 'run_records/MAXN_FINAL_VALIDATION.json',
              co / 'run_records/MAXN_COMPLETION_POWER.json', paper / 'PAPER_NIGHT_STATUS.json',
              paper / 'PAPER_NIGHT_COMPLETE.json', paper / 'TIMING_REPORT.md', paper / 'ORIN_PAPER_NIGHT_PROMPT.md']:
        if p.is_file():
            add(p, 'historical_provenance/' + p.relative_to(ROOT.parent).as_posix())
    status('PACKAGING_AT_VERIFIED_IDLE', archive=str(target), selected_files=len(selection))
    file_hashes = {}
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in selection.items():
            data = path.read_bytes()
            file_hashes[name] = hashlib.sha256(data).hexdigest()
            z.writestr(name, data)
        z.writestr('RETURN_MANIFEST.json', json.dumps(dict(utc=now(), full_runs=8, diagnostics=2,
                   historical_mode='MAXN', files=file_hashes, runtime_manifest_sha256=inventory['runtime_manifest_sha256'],
                   csv_validation='PASS', no_experiments_repeated=True), indent=2) + '\n')
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        for name, expected in file_hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    temporary.replace(target)
    digest = sha(target)
    target.with_suffix('.zip.sha256').write_text(digest + '  ' + target.name + '\n')
    receipt = dict(utc=now(), status='EXPORTED_AND_VERIFIED', archive=str(target), sha256=digest,
                   bytes=target.stat().st_size, verified_entries=len(file_hashes), full_runs=8, diagnostics=2,
                   csv_validation='PASS', no_experiments_repeated=True, power_or_clocks_changed=False,
                   scope='Existing Co-SLAM DP/MC_DP evidence; ScanNet queue was not paused or changed')
    save(ROOT / 'EXPORT_RECEIPT.json', receipt)
    status('COMPLETE', receipt=receipt)
    print('EXISTING_DP_EXPORT_COMPLETE', str(target), digest, flush=True)


def main():
    cfg = read(ROOT / 'EXPORT_FOLLOWER_CONFIG.json')
    with open('/tmp/dasip_round2_existing_dp_export_v1.lock', 'a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(ROOT / 'EXPORT_FOLLOWER_IDENTITY.json', identity(os.getpid()))
        if (ROOT / 'EXPORT_RECEIPT.json').is_file():
            return
        expected_parent = cfg['current_queue_parent_identity']
        while True:
            state = read(cfg['queue_status'])
            parent = identity(expected_parent['pid'])
            if parent == expected_parent:
                status('WAITING_FOR_CURRENT_QUEUE_TO_RELEASE_OWNERSHIP', queue_parent_alive=True,
                       queue_phase=state['phase'], queue_counts=state['counts'], active_queue_child=state.get('active_child_pid'))
                time.sleep(cfg['poll_seconds'])
                continue
            with open('/tmp/dasip_round2_gpu0.lock', 'a') as workload:
                try:
                    fcntl.flock(workload, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    status('WAITING_FOR_NEXT_EXCLUSIVE_IDLE_BOUNDARY', captured_parent_exited=True, workload_lock_busy=True)
                    time.sleep(cfg['poll_seconds'])
                    continue
                samples = []
                try:
                    for _ in range(3):
                        samples.append(idle())
                        time.sleep(2)
                except AssertionError as e:
                    status('WAITING_FOR_FRESH_HOST_IDLE', error=str(e), captured_parent_exited=True)
                    time.sleep(cfg['poll_seconds'])
                    continue
                assert identity(expected_parent['pid']) != expected_parent
                for path, expected in read(ROOT / 'EXPORT_FREEZE.json')['files'].items():
                    assert sha(path) == expected, 'Installed export source/selection changed: ' + path
                state = read(cfg['queue_status'])
                power = subprocess.run(['/usr/sbin/nvpmodel', '-q'], capture_output=True, text=True, timeout=10)
                save(ROOT / 'IDLE_EXPORT_BOUNDARY.json', dict(utc=now(), captured_parent_identity=expected_parent,
                     captured_parent_exited=True, queue_phase=state['phase'], queue_counts=state['counts'],
                     queue_completion_claimed=state['phase'] == 'COMPLETE_ADMITTED_SCANNET_SERIES',
                     workload_lock='/tmp/dasip_round2_gpu0.lock', exclusive_ownership_held=True, fresh_idle_samples=samples,
                     current_power_query=power.stdout, current_power_returncode=power.returncode,
                     historical_result_mode='MAXN', boundary='Verified exclusive workload idle after captured parent exit; no dependence on ALL_REQUIRED_MAXN_COMPLETE; no queue pause'))
                status('VERIFIED_IDLE_EXPORT_START')
                export(read(ROOT / 'ORIN_EXISTING_DP_INVENTORY.json'), cfg)
                idle()
                return


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save(ROOT / 'EXPORT_STOPPED.json', dict(utc=now(), error=repr(e), traceback=traceback.format_exc(),
             original_results_preserved=True, queue_not_modified=True, no_routine_email=True))
        status('STOPPED_EXPORT_ERROR', error=repr(e))
        raise
