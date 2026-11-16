"""Incremental raw export only after successful entire-profile parent exit and exclusive idle."""
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parent
TOP = ROOT.parents[1]
EXPERIMENTS = ROOT.parent
PROFILE = EXPERIMENTS / 'orin_component_profile_metadata_recovery_v1'
DP = EXPERIMENTS / 'orin_dp_completion_maxn_v1'
SCAN = EXPERIMENTS / 'orin_scannet_repair_v1'
GPU_LOCK = '/tmp/dasip_round2_gpu0.lock'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def profile_support():
    return module('profile_host_checks', PROFILE / 'local_support.py')


def permitted(name):
    path = Path(name)
    return (path.suffix.lower() in {'.csv', '.json', '.yaml', '.yml', '.log', '.txt', '.md', '.py', '.cu', '.cpp', '.h', '.sh', '.diff'}
            and not any(part in {'__pycache__', 'ckpts', 'checkpoints', 'mesh', 'targetPlot', 'prepared_data'} for part in path.parts))


def benchmark_jobs():
    native = [j for j in read(SCAN / 'CURRENT_SCOPE_STATUS.json')['jobs']
              if j.get('required_for_completion', True) and j['status'] in ['done', 'reused']]
    pruning = read(DP / 'DP_COMPLETION_STATUS.json')['jobs']
    assert len(native) == 196 and len(pruning) == 96
    jobs = {j['id']: j for j in native + pruning}
    assert len(jobs) == 292
    return jobs


def partition(request, previous, jobs):
    missing = {j['id']: j for j in request['jobs']}
    prior = {j['id']: j for j in previous}
    assert len(missing) == 120 and len(prior) == 172
    assert not (missing.keys() & prior.keys()), 'Requested missing identities overlap previous archive'
    assert missing.keys() | prior.keys() == jobs.keys(), '292-slot identity coverage mismatch'
    for key, target in missing.items():
        actual = jobs[key]
        for field in ['dataset', 'scene', 'condition', 'seed', 'hash_size']:
            assert actual[field] == target[field], (key, field)
        assert actual['status'] in ['done', 'reused']
        assert target['seed'] in [0, 1] and target['system'] == 'Co-SLAM'
    old_eight = [key for key in missing if jobs[key]['status'] == 'reused' and jobs[key]['condition'] in ['DP', 'MC_DP']]
    assert len(old_eight) == 8
    return missing, prior, old_eight


def verify_freeze():
    for path, expected in read(ROOT / 'INSTALLATION_FREEZE.json')['files'].items():
        assert sha(path) == expected, 'Exporter/input changed: ' + path


def wake(config, message, filename):
    path = ROOT / filename
    if path.exists():
        return
    n = config['notification']
    result = subprocess.run([n['codex_executable'], 'queue', '--thread', n['thread_id'], '--message', message],
                            capture_output=True, text=True, timeout=55)
    save(path if result.returncode == 0 else path.with_suffix('.error.json'),
         dict(utc=now(), returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, email_sent=False))


def build_export(config, support):
    """No call to this function occurs until all profile/export gates and parent exits pass."""
    request = read(config['request'])
    jobs = benchmark_jobs()
    previous_path = Path(config['previous_archive'])
    assert sha(previous_path) == request['previous_archive_sha256']
    # Read nested previous archive payloads only at the final exclusive idle boundary.
    nested = {}
    previous_refs = {}
    with zipfile.ZipFile(previous_path) as wrapper:
        assert wrapper.testzip() is None
        old_manifest_raw = wrapper.read('RTX_IMPORT_MANIFEST.json')
        old_manifest_sha256 = hashlib.sha256(old_manifest_raw).hexdigest()
        old_manifest = json.loads(old_manifest_raw)
        previous = old_manifest['completed_identity_mapping']
        missing, prior, old_eight = partition(request, previous, jobs)
        for item in old_manifest['archives']:
            data = wrapper.read(item['name'])
            assert hashlib.sha256(data).hexdigest() == item['sha256']
            archive = zipfile.ZipFile(io.BytesIO(data))
            assert archive.testzip() is None
            nested[item['name']] = archive
        for key, target in prior.items():
            basename = Path(target['output']).name
            matches = [(name, member) for name, archive in nested.items() for member in archive.namelist()
                       if member.endswith('/' + basename + '/summary.json')]
            assert len(matches) == 1, ('Previous output is missing/ambiguous', key, matches)
            archive_name, summary_name = matches[0]
            archive = nested[archive_name]
            prefix = summary_name[:-len('summary.json')]
            refs = []
            for member in archive.namelist():
                if member.startswith(prefix) and permitted(member):
                    data = archive.read(member)
                    refs.append(dict(archive=previous_path.name, nested_archive=archive_name,
                                     member=member, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
            names = {Path(row['member']).name for row in refs}
            assert {'summary.json', 'config.json', 'source_manifest.json'} <= names, key
            previous_refs[key] = refs
        for archive in nested.values():
            archive.close()
    sys.path.insert(0, str(DP))
    scientific_audit = module('incremental_raw_benchmark_audit', DP / 'common.py')
    target = Path(config['archive'])
    temporary = target.with_suffix('.zip.tmp')
    assert not target.exists() and not temporary.exists(), 'Never overwrite an existing transfer'
    hashes, mapping, benchmark_index, profile_index = {}, {}, [], []
    def add_bytes(archive, name, data, source):
        assert name not in hashes, 'Duplicate archive member: ' + name
        archive.writestr(name, data)
        hashes[name] = hashlib.sha256(data).hexdigest()
        mapping[name] = source
        return dict(archive=target.name, member=name, sha256=hashes[name], bytes=len(data))
    def add_path(archive, name, path):
        return add_bytes(archive, name, Path(path).read_bytes(), str(path))
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for key in sorted(jobs):
            job = jobs[key]
            row = {k: job.get(k) for k in ['id', 'system', 'dataset', 'scene', 'condition', 'seed', 'hash_size', 'status', 'output']}
            row['system'] = row['system'] or 'Co-SLAM'
            if key in prior:
                row.update(delivery='previous_172_archive', raw_files=previous_refs[key],
                           completion_evidence=dict(archive=previous_path.name, member='RTX_IMPORT_MANIFEST.json',
                                                    sha256=old_manifest_sha256, identity=key))
            else:
                wanted = missing[key]
                output = Path(job['output'])
                audit = scientific_audit.audit(dict(wanted, datadir=read(output / 'config.json')['data']['datadir']),
                                                output, config['scientific_runtime_manifest_sha256'])
                assert audit['status'] == 'PASS'
                done = Path(job.get('completion_record', output.parent / (output.name + '.done.json')))
                assert read(done)['returncode'] == 0, key
                refs = []
                for source in sorted(output.rglob('*')):
                    if source.is_file() and not source.is_symlink() and permitted(source.relative_to(output).as_posix()):
                        refs.append(add_path(archive, f'benchmarks/{key}/' + source.relative_to(output).as_posix(), source))
                required = {'calls.csv', 'iterations.csv', 'queries.csv', 'frames.csv', 'frame_iteration_sums.csv',
                            'summary.json', 'config.json', 'source_manifest.json'}
                assert required <= {Path(x['member']).name for x in refs}, key
                completion = add_path(archive, f'benchmarks/{key}/completion.json', done)
                refs.append(completion)
                for source in sorted(output.parent.iterdir()):
                    if not source.is_file() or source == done:
                        continue
                    related = source.name.startswith(output.name + '.') or source.name == 'queue_complete.json'
                    if related and permitted(source.name):
                        refs.append(add_path(archive, f'benchmarks/{key}/controller/' + source.name, source))
                row.update(delivery='incremental_120', raw_files=refs, completion_evidence=completion,
                           native_frames=wanted['expected_native_frames'], runtime_family=wanted['runtime_family'],
                           raw_call_iteration_query_audit='PASS', older_DP_run=key in old_eight)
            for filename, field in [('summary.json', 'summary_sha256'), ('config.json', 'config_sha256'), ('source_manifest.json', 'source_manifest_sha256')]:
                matches = [ref for ref in row['raw_files'] if Path(ref['member']).name == filename and '/artifacts/' not in ref['member']]
                assert len(matches) == 1, (key, filename, matches)
                row[field] = matches[0]['sha256']
            benchmark_index.append(row)
        # Include all five profiles and their raw stage directories, sources and validation once.
        profile_receipt = read(PROFILE / 'EXPORT_COMPLETE.json')
        profile_zip = Path(profile_receipt['path'])
        assert profile_receipt['campaign_success'] and sha(profile_zip) == profile_receipt['sha256']
        with zipfile.ZipFile(profile_zip) as profile_archive:
            assert profile_archive.testzip() is None
            pm = json.loads(profile_archive.read('RETURN_MANIFEST.json'))
            assert pm['completion']['full_profiles_completed'] == 5
            for member, expected in pm['files'].items():
                data = profile_archive.read(member)
                assert hashlib.sha256(data).hexdigest() == expected, member
                assert permitted(member), ('Unexpected non-raw profile payload', member)
                add_bytes(archive, 'component_profiles/' + member, data, str(profile_zip) + '::' + member)
            for system, queue in [('CO', 'co_maxn'), ('EC', 'ec_maxn')]:
                spec = read(PROFILE / 'manifests' / f'{system}_MAXN_SINGLE_SEED.json')
                for job in spec['jobs']:
                    prefix = f'component_profiles/results/{queue}/{job["id"]}/'
                    files = [dict(archive=target.name, member=name, sha256=digest) for name, digest in hashes.items() if name.startswith(prefix)]
                    assert files and job['seed'] == 0
                    profile_index.append(dict(id=job['id'], system=system, workload=job['workload'], seed=0, raw_files=files))
        scientific_root = EXPERIMENTS / 'orin_coslam_runtime_v1/runtime'
        manifest = read(scientific_root / 'RUNTIME_MANIFEST.json')
        assert sha(scientific_root / 'RUNTIME_MANIFEST.json') == config['scientific_runtime_manifest_sha256']
        add_path(archive, 'scientific_runtime/RUNTIME_MANIFEST.json', scientific_root / 'RUNTIME_MANIFEST.json')
        for member, expected in manifest['runtime_hashes'].items():
            path = scientific_root / member
            assert sha(path) == expected
            add_path(archive, 'scientific_runtime/' + member, path)
        # Mode/clock/thermal records and completed-series provenance; no dataset or binary payload.
        context_roots = [DP, SCAN, EXPERIMENTS / 'orin_existing_dp_export_v1', PROFILE,
                         EXPERIMENTS / 'orin_coslam_runtime_v1', EXPERIMENTS / 'orin_coslam_runtime_v1/run_records',
                         EXPERIMENTS / 'orin_paper_night_v2',
                         EXPERIMENTS / 'orin_continuous_v4']
        for context in context_roots:
            for path in sorted(context.iterdir()):
                if not path.is_file() or not permitted(path.name):
                    continue
                if any(word in path.name.upper() for word in ['ENVIRONMENT', 'CLOCK', 'POWER', 'COMPLETE', 'VERIFIED', 'READINESS', 'VALIDATION', 'STATUS', 'INVENTORY']):
                    add_path(archive, 'series_context/' + context.name + '/' + path.name, path)
        for path in sorted(ROOT.iterdir()):
            if path.is_file() and permitted(path.name):
                add_path(archive, 'export_controller/' + path.name, path)
        add_path(archive, 'MAXN_MISSING_EXPORT_REQUEST.json', config['request'])
        master = dict(utc=now(), mode='MAXN', benchmarks=benchmark_index, profiles=profile_index,
            benchmark_count=292, previous_audited_benchmarks=172, incremental_benchmarks=120,
            profile_count=5, benchmark_seeds=[0, 1], profile_seeds=[0],
            older_DP_runs_included=old_eight, no_duplicate_benchmark_ids=True,
            previous_archive=dict(path=str(previous_path), sha256=request['previous_archive_sha256']),
            profile_archive=dict(path=str(profile_zip), sha256=profile_receipt['sha256']),
            diagnostics_separate=True, EC_ENG='excluded', experiments_repeated=0)
        assert len({row['id'] for row in benchmark_index}) == len(benchmark_index) == 292
        assert len(profile_index) == len({row['id'] for row in profile_index}) == 5
        save(ROOT / 'MAXN_MASTER_INDEX.json', master)
        add_bytes(archive, 'MAXN_MASTER_INDEX.json', json.dumps(master, indent=2).encode() + b'\n', str(ROOT / 'MAXN_MASTER_INDEX.json'))
        archive.writestr('SHA256SUMS.json', json.dumps(hashes, indent=2) + '\n')
        archive.writestr('PATH_MAPPING.json', json.dumps(mapping, indent=2) + '\n')
        archive.writestr('RETURN_MANIFEST.json', json.dumps(dict(utc=now(), files=hashes,
            benchmark_count=292, incremental_benchmarks=120, previous_benchmarks=172, profile_count=5,
            older_DP_runs=8, all_requested_identities_complete=True, experiments_repeated=0), indent=2) + '\n')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
        for member, expected in hashes.items():
            assert hashlib.sha256(archive.read(member)).hexdigest() == expected, member
    temporary.replace(target)
    digest = sha(target)
    target.with_suffix('.zip.sha256').write_text(digest + '  ' + target.name + '\n')
    receipt = dict(utc=now(), status='EXPORTED_AND_VERIFIED', path=str(target), sha256=digest,
        bytes=target.stat().st_size, files_verified=len(hashes), incremental_benchmarks=120,
        previous_audited_benchmarks=172, master_benchmarks=292, profiles=5, older_DP_runs=8,
        master_index=str(ROOT / 'MAXN_MASTER_INDEX.json'), master_index_sha256=sha(ROOT / 'MAXN_MASTER_INDEX.json'),
        experiments_repeated=0, power_or_clocks_changed=False)
    save(ROOT / 'EXPORT_COMPLETE.json', receipt)
    return receipt


def main():
    config = read(ROOT / 'INSTALLATION_CONFIG.json')
    support = profile_support()
    verify_freeze()
    with open('/tmp/dasip_round2_missing_120_export.lock', 'a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(ROOT / 'FOLLOWER_IDENTITY.json', support.process_identity(os.getpid()))
        state = dict(phase='WAITING_FOR_ENTIRE_PROFILE_SERIES_AND_EXPORT_PARENT_EXIT', follower_pid=os.getpid(),
                     packaging_started=False, benchmark_runs_launched=0, GPU_work_launched=False)
        gate = read(ROOT / 'FINAL_PROFILE_AND_EXPORT_GATE.json')
        while True:
            ready, reason = support.gate_ready(gate)
            state.update(utc=now(), reason=reason)
            save(ROOT / 'EXPORT_STATUS.json', state)
            print(now(), state['phase'], reason, flush=True)
            if ready:
                break
            time.sleep(30)
        with open(GPU_LOCK, 'a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    state.update(utc=now(), phase='WAITING_FOR_RELEASED_FINAL_OWNERSHIP')
                    save(ROOT / 'EXPORT_STATUS.json', state)
                    time.sleep(15)
            assert support.gate_ready(gate)[0]
            samples = support.fresh_idle_three()
            verify_freeze()
            save(ROOT / 'FINAL_IDLE_BOUNDARY.json', dict(utc=now(), samples=samples, GPU_lock=GPU_LOCK,
                                                       all_profile_parents_exited=True, gates_passed=True))
            state.update(utc=now(), phase='EXPORTING_AT_FINAL_MAXN_IDLE_BOUNDARY', packaging_started=True)
            save(ROOT / 'EXPORT_STATUS.json', state)
            receipt = build_export(config, support)
        state.update(utc=now(), phase='COMPLETE', archive=receipt['path'], sha256=receipt['sha256'])
        save(ROOT / 'EXPORT_STATUS.json', state)
        samples = support.fresh_idle_three()
        save(ROOT / 'EMAIL_CAMPAIGN_STATUS.json', dict(utc=now(), major_series_terminal=True,
             profiling_and_requested_incremental_export_complete=True, fresh_host_idle=samples, email_sent=False))
        wake(config, 'The user-authorized final Orin MAXN programme, recovered five seed-0 Co/EC profiles, and requested 120-missing-identity incremental export have completed. Read ' +
             str(ROOT / 'EXPORT_COMPLETE.json') + ', ' + str(ROOT / 'MAXN_MASTER_INDEX.json') + ', ' + str(PROFILE / 'PROFILING_COMPLETE.json') +
             '. Verify the final archive and master index: previous172 + incremental120 =292 benchmark identities with no duplicates, including all eight older DP/MC_DP runs, plus five profiles and separate diagnostics. ' +
             'Return final ZIP path/checksum and existing source export paths/checksums. Recheck fresh GPU/program idle immediately before notification. Only for this successfully completed major series and verified export with fresh idle, send ONE authorized decision email to youssefhaddouda233@gmail.com via connected Gmail. ' +
             'Check ' + str(ROOT / 'EMAIL_SENT.json') + ', profile/DP real sent receipts and Gmail Sent to prevent duplicates; DEFERRED wake guards are not sent receipts. Save a real sent-message ID only after success. ' +
             'State MAXN completeness truthfully, list any pending work, keep EC ENG excluded and ask the author to decide the manual 15 W switch. Do not switch modes/clocks, launch 15 W or repeat experiments. If Gmail is unavailable, report that without claiming delivery. Fresh idle evidence: ' + str(ROOT / 'EMAIL_CAMPAIGN_STATUS.json'),
             'EMAIL_WAKE_RECEIPT.json')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(ROOT / 'EXPORT_STOPPED.json', dict(utc=now(), error=repr(error), traceback=traceback.format_exc(),
                                               email_sent=False, experiments_repeated=0))
        try:
            config = read(ROOT / 'INSTALLATION_CONFIG.json')
            wake(config, 'The final Orin 120-identity/export follower stopped or is blocked. Inspect ' + str(ROOT / 'EXPORT_STOPPED.json') +
                 '. Preserve valid profiling jobs, all completed measurements and exports. Repair only archival/orchestration errors after the entire series finishes and GPU ownership is released. No experiment reruns, changes to checks/computation/power/clocks, or isolated-failure email.', 'EXPORT_REPAIR_WAKE_RECEIPT.json')
        except Exception:
            pass
        raise
