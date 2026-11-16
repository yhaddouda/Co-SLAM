"""Reproducible lightweight local installation metadata; no dataset/GPU workload."""
import copy
import json
import re
import shlex
from local_support import *


def check(pointer, value):
    return dict(pointer=pointer, equals=value)


def condition(path, fields):
    return dict(file=str(path), checks=[check(k, v) for k, v in fields.items()])


def main():
    for name in ['manifests', 'gates', 'results', '15W_deferred']:
        (ROOT / name).mkdir(exist_ok=True)
    state = read(DP / 'DP_COMPLETION_STATUS.json')
    dpconfig = read(DP / 'INSTALLATION_CONFIG.json')
    missing_path = DP / 'manifests/RESOLVED_MISSING_MAXN.json'
    missing = read(missing_path)['jobs']
    all_path = DP / 'bundle/manifests/orin_dp_mc_dp_all_targets.json'
    all_jobs = read(all_path)['jobs']
    assert len(missing) == 88 and len(all_jobs) == 96
    assert [j['id'] for j in state['jobs']] == [j['id'] for j in all_jobs]
    processes = [process_identity(state[k]) for k in ['follower_pid', 'worker_pid']]
    assert all(processes)
    for key in ['scannet_parent_identity', 'exporter_identity']:
        processes.append(dpconfig[key])
    native = read(EXPERIMENTS / 'orin_continuous_v4/CONTINUOUS_STATUS.json')
    natives = {j['id']: j for j in native['jobs']}
    manifest = read(ROOT / 'bundle/manifests/orin_component_profile_maxn_v1.json')
    controls = []
    for job in manifest['jobs']:
        match = re.fullmatch(r'(Replica_office0|Tum_fr1_desk)_H(16|19)', job['workload'])
        assert match
        native_id = match[1] + '_BASE_H' + match[2] + '_seed' + str(job['seed'])
        control = natives[native_id]
        assert control['status'] in ['done', 'reused'] and control['system'] == 'Co-SLAM'
        out = Path(control['output'])
        cfgpath, summarypath = out / 'config.json', out / 'summary.json'
        cfg, summary = read(cfgpath), read(summarypath)
        done = out.parent / (control['reuse_provenance']['original_job'] + '.done.json')
        assert read(done)['returncode'] == 0 and sha(done) == control['reuse_provenance']['completion_sha256']
        assert sha(cfgpath) == control['reuse_provenance']['effective_config_sha256']
        assert summary['condition'] == 'BASE' and summary['seed'] == job['seed'] and summary['full_sequence']
        assert summary['frames'] == (2000 if match[1].startswith('Replica') else 592)
        assert cfg['grid']['hash_size'] == int(match[2]) and not cfg['round2']['engineering']
        assert sha(out / 'source_manifest.json') == '9786e843d6856dcc443fb08ebb9646a5d759384e19ab3456552b51e2281e616d'
        job['datadir'] = cfg['data']['datadir']
        assert Path(job['datadir']).is_dir()
        controls.append(dict(id=native_id, output=str(out), datadir=job['datadir'],
                             config_path=str(cfgpath), config_sha256=sha(cfgpath),
                             summary_path=str(summarypath), summary_sha256=sha(summarypath),
                             completion_path=str(done), completion_sha256=sha(done)))
    python = '/home/yh279050/miniforge3/envs/coslam/bin/python'
    manifest.update(idle_check_command=[python, '-B', str(ROOT / 'idle_check.py')],
                    clock_policy_record='Retain author-selected MAXN clock settings; no changes. Prior GPU min/max 1300500000 Hz; full jetson_clocks lock state unknown. Fresh sysfs/environment snapshots are collected by every profile.',
                    path_status='Resolved from six completed native BASE effective configs and successful completion evidence')
    save(ROOT / 'manifests/MAXN_RESOLVED.json', manifest)
    save(ROOT / 'NATIVE_BASE_PATH_PROVENANCE.json', controls)
    protocol = read(ROOT / 'bundle/runtime/PROTOCOL.json')
    selections = {w['id']: {stage: [f for t in w['thirds'] for f in t[stage]['selected_frames']]
                              for stage in ['tracking', 'bundle_adjustment']} for w in protocol['workloads']}
    for selection in selections.values():
        assert len(selection['tracking']) == 300 and len(selection['bundle_adjustment']) == 60
        assert min(selection['tracking']) >= 20 and min(selection['bundle_adjustment']) >= 20
    save(ROOT / 'FROZEN_SELECTIONS.json', dict(protocol_sha256=sha(ROOT / 'bundle/runtime/PROTOCOL.json'),
                                             selections=selections,
                                             selection_sha256=hashlib.sha256(json.dumps(selections, sort_keys=True).encode()).hexdigest()))
    fields = {'/phase': 'COMPLETE_ADMITTED_DP_SERIES', '/active_child_pid': None,
              '/counts/done': 88, '/counts/reused': 8,
              '/native_engineering_completed': 196, '/pruning_completed': 96,
              '/expanded_completed': 292, '/EC_ENG': 'excluded'}
    missing_ids = {j['id'] for j in missing}
    for index, target in enumerate(all_jobs):
        for key in ['id', 'dataset', 'scene', 'condition', 'seed', 'hash_size', 'expected_native_frames']:
            fields[f'/jobs/{index}/{key}'] = target[key]
        fields[f'/jobs/{index}/status'] = 'done' if target['id'] in missing_ids else 'reused'
        fields[f'/jobs/{index}/source_manifest_sha256'] = '9786e843d6856dcc443fb08ebb9646a5d759384e19ab3456552b51e2281e616d'
    conditions = [condition(DP / 'DP_COMPLETION_STATUS.json', fields),
        condition(DP / 'DP_COMPLETION_COMPLETE.json', {'/major_admitted_series_terminal': True,
                  '/new_full_completed': 88, '/counts/done': 88, '/counts/reused': 8, '/expanded_completed': 292,
                  '/power': 'NV Power Mode: MAXN\n0'}),
        condition(DP / 'WORKER_EXIT.json', {'/returncode': 0}),
        condition(DP / 'FOLLOWER_COMPLETE.json', {'/worker_returncode': 0, '/major_admitted_series_terminal': True}),
        condition(DP / 'RETURN_BUNDLE_VERIFIED.json', {'/path': str(TOP / 'ORIN_DP_COMPLETION_MAXN_RESULTS.zip')}),
        condition(OLD_EXPORT / 'EXPORT_RECEIPT.json', {'/status': 'EXPORTED_AND_VERIFIED', '/full_runs': 8, '/diagnostics': 2, '/csv_validation': 'PASS'}),
        condition(OLD_EXPORT / 'EXPORT_STATUS.json', {'/phase': 'COMPLETE'}),
        condition(SCAN / 'SCANNET_COMPLETE.json', {'/major_admitted_series_terminal': True, '/counts/done': 24,
                  '/power': 'NV Power Mode: MAXN\n0'})]
    scanstate = read(SCAN / 'SCANNET_STATUS.json')
    scanfields = {'/phase': 'COMPLETE_ADMITTED_SCANNET_SERIES', '/active_child_pid': None, '/counts/done': 24}
    assert len(scanstate['jobs']) == 24
    for index, target in enumerate(scanstate['jobs']):
        assert target['status'] == 'done'
        for key in ['id', 'dataset', 'scene', 'condition', 'seed', 'hash_size', 'expected_native_frames', 'source_manifest_sha256']:
            scanfields[f'/jobs/{index}/{key}'] = target[key]
        scanfields[f'/jobs/{index}/status'] = 'done'
    conditions.append(condition(SCAN / 'SCANNET_STATUS.json', scanfields))
    gate = dict(conditions=conditions, failure_files=[str(p) for p in [DP / 'WORKER_STOPPED.json',
                DP / 'FOLLOWER_STOPPED.json', SCAN / 'STOPPED.json', OLD_EXPORT / 'EXPORT_STOPPED.json']],
                wait_processes=processes,
                note='Exact 96 target identities and successful existing 292-slot programme only; no clock-time or expanded profiling ALL_REQUIRED flag gate')
    save(ROOT / 'gates/UPSTREAM_SUCCESS_AND_EXPORTS.json', gate)
    controller_gate = copy.deepcopy(gate)
    controller_gate['conditions'].append(condition(ROOT / 'UPSTREAM_VERIFIED.json',
         {'/status': 'VERIFIED', '/exact_dp_identities': 96, '/new_completed': 88, '/reused': 8,
          '/benchmark_slots_complete': 292, '/native_runtime_manifest_sha256': '9786e843d6856dcc443fb08ebb9646a5d759384e19ab3456552b51e2281e616d'}))
    save(ROOT / 'gates/CONTROLLER_GATE.json', controller_gate)
    config = dict(created_utc=now(), expected_dp_jobs=all_jobs, expected_missing_jobs=missing,
                  native_profile_controls=controls, native_runtime_manifest_sha256='9786e843d6856dcc443fb08ebb9646a5d759384e19ab3456552b51e2281e616d',
                  notification=dpconfig['notification'], upstream_processes=processes,
                  session='dasip_r2_component_profile_maxn_v2',
                  planned_archive=str(TOP / 'ORIN_COMPONENT_PROFILE_MAXN_RESULTS.zip'),
                  exact_current_programme_only=True, current_runtime_unchanged=True)
    save(ROOT / 'INSTALLATION_CONFIG.json', config)
    save(ROOT / 'COMPONENT_PROFILE_ACTIVE.json', dict(queue=str(ROOT / 'results/maxn'),
         runtime=str(ROOT / 'bundle/runtime'), manifest=str(ROOT / 'manifests/MAXN_RESOLVED.json')))
    template = read(ROOT / 'bundle/manifests/orin_component_profile_15w_v1.json')
    confpath = Path('/etc/nvpmodel.conf').resolve()
    assert re.search(r'<\s*POWER_MODEL\s+ID=1\s+NAME=MODE_15W\s*>', confpath.read_text())
    for job, resolved in zip(template['jobs'], manifest['jobs']):
        assert job['id'] == resolved['id']
        job['datadir'] = resolved['datadir']
    template.update(power_mode='MODE_15W', idle_check_command=[python, '-B', str(ROOT / 'idle_check.py'), '--expected-mode', 'MODE_15W'],
                    power_mode_source=str(confpath) + ': configured ID 1; re-query actual selected mode after author manual switch',
                    author_manual_switch_and_actual_mode_verification_required=True,
                    do_not_launch_automatically=True, launch_only_after_future_15W_benchmarks_and_exports_complete=True,
                    upstream_gate_status='UNBOUND: actual future 15 W programme identities and export receipts must be bound; never use MAXN gates',
                    path_status='Same verified native datadirs; recheck against completed future native 15 W controls',
                    clock_policy_record='Retain author-selected 15 W policy; record actual state; no automatic changes')
    save(ROOT / '15W_deferred/15W_TEMPLATE_NOT_LAUNCHED.json', template)
    save(ROOT / '15W_deferred/CONTINUATION_PLAN.json', dict(status='PREPARED_NOT_ARMED', profiles=6, diagnostics=3,
        supplied_controller=str(ROOT / 'bundle/run_component_profile_campaign.py'),
        manifest=str(ROOT / '15W_deferred/15W_TEMPLATE_NOT_LAUNCHED.json'),
        requirements=['Actual author-selected MODE_15W verified by nvpmodel -q', 'Future 15 W native controls and all expected ready identities complete normally',
                      'All future 15 W planned exports verified and captured parents exited', 'Shared GPU lock and three fresh validated MODE_15W idle checks',
                      'Distinct results/15w and ORIN_COMPONENT_PROFILE_15W_RESULTS.zip; same selections, runtime and validation tolerances',
                      'Final export, fresh idle, one decision notification; no mode changes'],
        missing_binding='Future 15 W benchmark controllers/manifests/export receipts do not yet exist; cannot truthfully bind now'))
    (ROOT / 'launch_follow_maxn.sh').write_text('#!/bin/bash\nset -euo pipefail\nsource /home/yh279050/miniforge3/etc/profile.d/conda.sh\nconda activate coslam\nexport PYTHONDONTWRITEBYTECODE=1\ncd /home/yh279050/work/Co-SLAM\nexec python -B -u ' + shlex.quote(str(ROOT / 'continuation.py')) + ' >> ' + shlex.quote(str(ROOT / 'follower.log')) + ' 2>&1\n')
    os.chmod(ROOT / 'launch_follow_maxn.sh', 0o755)
    base_seconds = sum(read(c['summary_path'])['slam_work_wall_seconds'] for c in controls)
    save(ROOT / 'ETA_CONTEXT.json', dict(utc=now(), completed_native_BASE_seconds_for_six_profiles=base_seconds,
         note='Six new profiles plus three diagnostics extend the previous 292-slot end-of-MAXN decision point. Uninstrumented BASE proxy is not a profiling ETA; event overhead must be measured locally.',
         provisional_extra_hours=[1.5, 3.0], refresh='After local diagnostics/first full profile; upstream forecast excludes this block'))
    files = {str(p): sha(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and p.name not in ['INSTALLATION_FREEZE.json'] and p.suffix not in ['.log']}
    for freeze in ['EXECUTION_FREEZE.json', 'INSTALLATION_FREEZE.json']:
        files.update(read(DP / freeze)['files'])
    for c in controls:
        for key in ['config_path', 'summary_path', 'completion_path']:
            files[c[key]] = sha(c[key])
    files[str(TOP / 'orin_component_profile_runtime_v2.zip')] = sha(TOP / 'orin_component_profile_runtime_v2.zip')
    save(ROOT / 'INSTALLATION_FREEZE.json', dict(utc=now(), files=files,
         scope='Small source/config/identity files only; bulk results and archive audit deferred until exclusive idle'))
    print('SETUP_CONFIGURED', ROOT, '96 exact DP + 24 ScanNet identities; six full profiles; future 15 W unarmed')


if __name__ == '__main__':
    main()
