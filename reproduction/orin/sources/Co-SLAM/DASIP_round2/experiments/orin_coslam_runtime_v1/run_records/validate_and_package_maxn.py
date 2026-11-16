"""Validate recorded outputs and package MAXN results without GPU work."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['frame_id', 'iteration_type', 'iteration_id', 'forward_cuda_ms',
          'backward_cuda_ms', 'full_iteration_cuda_ms']


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def validate_queue(directory, manifest_path, full):
    manifest = json.loads(manifest_path.read_text())
    completion = json.loads((directory / 'queue_complete.json').read_text())
    assert completion['completed'] == len(manifest['jobs'])
    results = []
    for job in manifest['jobs']:
        folder = directory / job['id']
        summary = json.loads((folder / 'summary.json').read_text())
        cfg = json.loads((folder / 'config.json').read_text())
        assert (directory / (job['id'] + '.done.json')).is_file(), job['id']
        assert summary['condition'] == job['condition'] and summary['seed'] == job['seed']
        assert summary['finite_parameters_and_poses'] and not summary['quality']
        assert summary['frames'] == (2000 if full else 50)
        assert summary['full_sequence'] == full and summary['diagnostic'] != full
        assert summary['warmup_frames'] == 20
        assert cfg['grid']['hash_size'] == 16 and cfg['mapping']['n_pixels'] == .05
        assert all(summary[k] == 'NV Power Mode: MAXN\n0' for k in ('initial_power', 'final_power'))
        assert (folder / 'source_manifest.json').read_bytes() == (ROOT / 'runtime/RUNTIME_MANIFEST.json').read_bytes()
        frames = summary['frames']
        expected = {'tracking': set(range(1, frames)),
                    'bundle_adjustment': set(range(cfg['mapping']['map_every'], frames, cfg['mapping']['map_every']))}
        iterations = {'tracking': cfg['tracking']['iter'], 'bundle_adjustment': cfg['mapping']['iters']}
        calls = rows(folder / 'calls.csv')
        query = rows(folder / 'queries.csv')
        events = rows(folder / 'iterations.csv')
        with (folder / 'iterations.csv').open() as stream:
            assert next(csv.reader(stream)) == FIELDS
        for stage, frame_ids in expected.items():
            selected_calls = [r for r in calls if r['stage'] == stage]
            assert len(selected_calls) == len(frame_ids)
            assert {int(r['frame_id']) for r in selected_calls} == frame_ids
            for r in selected_calls:
                assert all(math.isfinite(float(r[k])) and float(r[k]) >= 0 for k in ('wall_ms', 'cuda_ms'))
            selected_events = [r for r in events if r['iteration_type'] == stage]
            selected_queries = [r for r in query if r['stage'] == stage]
            assert len(selected_events) == len(selected_queries) == len(frame_ids) * iterations[stage]
            expected_ids = {(f, i) for f in frame_ids for i in range(iterations[stage])}
            assert {(int(r['frame_id']), int(r['iteration_id'])) for r in selected_events} == expected_ids
            assert {(int(r['frame_id']), int(r['iteration'])) for r in selected_queries} == expected_ids
            pruned = 0
            for r in selected_queries:
                i = int(r['iteration']); actual = int(r['actual_queries']); dense = int(r['dense_queries'])
                assert 0 < actual <= dense
                if job['condition'] in ('BASE', 'MC') or i in (0, iterations[stage] - 1) or i % 3 == 0:
                    assert actual == dense, (job['id'], stage, r)
                pruned += dense - actual
            assert (pruned > 0) == (job['condition'] in ('DP', 'MC_DP'))
        assert len(rows(folder / 'frames.csv')) == frames
        with (folder / 'frame_iteration_sums.csv').open() as stream:
            sums = list(csv.reader(stream))
        assert len(sums) == frames - 1
        for row in sums:
            assert len(row) == 4 and abs(float(row[1]) - float(row[2]) - float(row[3])) < 1e-5
        results.append({'job': job['id'], 'frames': frames, 'calls': len(calls),
                        'iterations': len(events), 'anchors_full': True, 'finite': True})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics-only', action='store_true')
    parser.add_argument('--queue-exit-code', type=int, default=0)
    args = parser.parse_args()
    record = {'utc_time': datetime.now(timezone.utc).isoformat(),
              'scope': 'office0 MAXN, four diagnostics and first two-seed BASE/DP timing queue',
              'quality_or_speed_acceptance_claimed': False,
              'clock_policy': 'Retain current clock settings; explicitly confirmed by author',
              'queue_exit_code': args.queue_exit_code}
    try:
        checked = subprocess.run([sys.executable, str(ROOT / 'verify_handoff.py'),
                                  '--diagnostics', str(ROOT / 'results/diagnostic_maxn')],
                                 text=True, capture_output=True)
        record['supplied_verifier'] = {'returncode': checked.returncode, 'stdout': checked.stdout, 'stderr': checked.stderr}
        assert checked.returncode == 0, checked.stdout + checked.stderr
        record['diagnostics'] = validate_queue(ROOT / 'results/diagnostic_maxn', ROOT / 'manifests/orin_coslam_diagnostic.json', False)
        if not args.diagnostics_only:
            assert args.queue_exit_code == 0, 'Full queue failed; see exact logs and partial output'
            record['full_queue'] = validate_queue(ROOT / 'results/full_maxn', ROOT / 'manifests/orin_coslam_first_full.json', True)
        record['validation'] = 'PASS'
    except Exception:
        record['validation'] = 'FAIL'
        record['error'] = traceback.format_exc()
    destination = ROOT / 'run_records' / ('DIAGNOSTIC_VERIFICATION.json' if args.diagnostics_only else 'MAXN_FINAL_VALIDATION.json')
    with destination.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record, indent=2), flush=True)
    if args.diagnostics_only:
        return 0 if record['validation'] == 'PASS' else 1
    watcher = ROOT.parent / 'orin_maxn_addendum/watch_round2_queue.py'
    # Let the independent monitor finish its atomic writes before archiving them.
    event = ROOT / 'results/full_maxn/MONITOR_EVENT.json'
    deadline = time.monotonic() + 25
    while not event.exists() and time.monotonic() < deadline:
        time.sleep(.5)
    if event.exists():
        (ROOT / 'run_records/final_monitor.log').write_text('Independent monitor event recorded.\n' + event.read_text())
    else:
        monitor = subprocess.run([sys.executable, str(watcher), '--queue', str(ROOT / 'results/full_maxn'),
                                  '--status', str(ROOT / 'results/MAXN_FULL_LIVE_STATUS.md'),
                                  '--email-to', 'youssefhaddouda233@gmail.com'], text=True, capture_output=True)
        (ROOT / 'run_records/final_monitor.log').write_text(monitor.stdout + monitor.stderr)
    power = subprocess.run(['/usr/sbin/nvpmodel', '-q'], text=True, capture_output=True)
    (ROOT / 'run_records/MAXN_COMPLETION_POWER.json').write_text(json.dumps(
        {'returncode': power.returncode, 'stdout': power.stdout, 'stderr': power.stderr}, indent=2) + '\n')
    archive_path = ROOT.parents[1] / 'ORIN_MAXN_RUNTIME_V1_RESULTS.zip'
    included = [ROOT / name for name in ('README.md', 'SHA256SUMS.json', 'run_manifest_queue.py', 'verify_handoff.py', 'runtime/RUNTIME_MANIFEST.json')]
    for directory in ('manifests', 'run_records', 'results'):
        included.extend(p for p in (ROOT / directory).rglob('*') if p.is_file())
    hashes = {}
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(included)):
            name = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
        archive.writestr('RETURN_MANIFEST.json', json.dumps({'validation': record['validation'], 'files': hashes}, indent=2) + '\n')
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_path.with_suffix('.zip.sha256').write_text(digest + '  ' + archive_path.name + '\n')
    print('RETURN_ARCHIVE', archive_path, digest, flush=True)
    return 0 if record['validation'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
