"""CPU-only single-seed Co+EC installation; existing benchmark files are read-only."""
import copy
import shlex
from local_support import *

OLD = EXPERIMENTS / 'orin_component_profile_runtime_v2'
PYTHON = '/home/yh279050/miniforge3/envs/coslam/bin/python'


def condition(path, fields):
    return dict(file=str(path), checks=[dict(pointer=k, equals=v) for k, v in fields.items()])


def main():
    for name in ['manifests', 'gates', 'results', '15W_deferred', 'preserved_previous']:
        (ROOT / name).mkdir(exist_ok=True)
    old_state = read(OLD / 'FOLLOWER_STATUS.json')
    old_id = read(OLD / 'FOLLOWER_IDENTITY.json')
    assert old_state['phase'] == 'WAITING_FOR_SUCCESSFUL_MAXN_PROGRAMME_AND_EXPORTS'
    assert not old_state['GPU_work_launched'] and not (OLD / 'CONTROLLER_LAUNCH.json').exists()
    assert still_alive(old_id)
    old_config = read(OLD / 'INSTALLATION_CONFIG.json')
    config = copy.deepcopy(old_config)
    config['created_utc'] = now()
    config['session'] = 'dasip_r2_components_maxn_single_seed'
    config['retired_waiting_follower'] = old_id
    config['planned_archive'] = str(TOP / 'ORIN_COMPONENT_PROFILE_MAXN_RESULTS.zip')
    config['native_profile_controls'] = [c for c in old_config['native_profile_controls'] if c['id'].endswith('seed0')]
    config['native_ec_controls'] = []
    native = {j['id']: j for j in read(EXPERIMENTS / 'orin_continuous_v4/CONTINUOUS_STATUS.json')['jobs']}
    ecjobs = read(ROOT / 'bundle/manifests/orin_ec_component_profile_maxn_single_seed_v1.json')['jobs']
    for job in ecjobs:
        cap = int(job['workload'].rsplit('H', 1)[1])
        control = native[f'EC_Replica_office0_BASE_H{cap}_seed0']
        assert control['system'] == 'EC-SLAM' and control['status'] in ['done', 'reused']
        output = Path(control['output'])
        cfg = read(output / 'config.json')
        summary = read(output / 'summary.json')
        assert summary['full_sequence'] and summary['finite_parameters_and_poses'] and summary['frames'] == 2000
        assert summary['initial_power'] == summary['final_power'] == 'NV Power Mode: MAXN\n0'
        assert cfg['HashGrid']['hash_size'] == cap and not cfg['round2']['engineering']
        if control['status'] == 'reused':
            completion = output.parent / (control['reuse_provenance']['original_job'] + '.done.json')
            assert sha(completion) == control['reuse_provenance']['completion_sha256']
            assert sha(output / 'config.json') == control['reuse_provenance']['effective_config_sha256']
        else:
            completion = EXPERIMENTS / 'orin_continuous_v4/controller_records' / (control['id'] + '.controller_done.json')
            assert sha(output / 'config.json') == control['actual_config_sha256']
        assert read(completion)['returncode'] == 0
        job['datadir'] = cfg['data']['input_folder']
        assert Path(job['datadir']).is_dir()
        config['native_ec_controls'].append(dict(id=control['id'], datadir=job['datadir'], output=str(output),
            config_path=str(output / 'config.json'), config_sha256=sha(output / 'config.json'),
            summary_path=str(output / 'summary.json'), summary_sha256=sha(output / 'summary.json'),
            source_path=str(output / 'source_manifest.json'), source_sha256=sha(output / 'source_manifest.json'),
            completion_path=str(completion), completion_sha256=sha(completion)))
    cojobs = read(ROOT / 'bundle/manifests/orin_component_profile_maxn_single_seed_v1.json')['jobs']
    for job in cojobs:
        match = next(c for c in config['native_profile_controls'] if c['id'] == job['workload'].replace('_H', '_BASE_H') + '_seed0')
        job['datadir'] = match['datadir']
    clock = 'Retain author-selected MAXN policy; no clock/power changes. GPU min/max previously verified 1300500000 Hz. Full jetson_clocks lock state unknown; collect fresh environment/sysfs snapshots.'
    for system, prefix, jobs in [('CO', 'orin_component_profile', cojobs), ('EC', 'orin_ec_component_profile', ecjobs)]:
        for mode, platform, mode_label in [('maxn', 'ORIN_MAXN', 'MAXN'), ('15w', 'ORIN_15W', 'MODE_15W')]:
            manifest = read(ROOT / 'bundle/manifests' / f'{prefix}_{mode}_single_seed_v1.json')
            manifest.update(jobs=copy.deepcopy(jobs), power_mode=mode_label,
                idle_check_command=[PYTHON, '-B', str(ROOT / 'idle_check.py'), '--expected-mode', mode_label],
                clock_policy_record=clock if mode == 'maxn' else 'Retain actual future author-selected 15 W policy; re-query mode/clocks after manual switch',
                path_status='Resolved from matching completed native BASE effective configs; EC native layout is preserved')
            if mode == 'maxn':
                save(ROOT / 'manifests' / f'{system}_MAXN_SINGLE_SEED.json', manifest)
            else:
                manifest.update(do_not_launch_automatically=True, author_manual_switch_and_actual_mode_verification_required=True,
                    upstream_gate_status='UNBOUND: actual future 15 W benchmark and export identities required',
                    run_only_after_future_15W_benchmark_programme_and_exports=True)
                save(ROOT / '15W_deferred' / f'{system}_15W_SINGLE_SEED_NOT_LAUNCHED.json', manifest)
    upstream = read(OLD / 'gates/UPSTREAM_SUCCESS_AND_EXPORTS.json')
    upstream['wait_processes'].append(old_id)
    save(ROOT / 'gates/UPSTREAM_SUCCESS_AND_EXPORTS.json', upstream)
    cogate = copy.deepcopy(upstream)
    cogate['conditions'].append(condition(ROOT / 'UPSTREAM_VERIFIED.json',
        {'/status': 'VERIFIED', '/exact_dp_identities': 96, '/benchmark_slots_complete': 292}))
    save(ROOT / 'gates/CO_CONTROLLER_GATE.json', cogate)
    copath = ROOT / 'results/co_maxn'
    ecgate = dict(conditions=[
        condition(copath / 'status.json', {'/status': 'COMPLETE', '/completed': 3, '/diagnostic_complete': 3, '/failed': 0, '/current': None}),
        condition(copath / 'series_complete.json', {'/status': 'COMPLETE', '/completed': 3, '/diagnostic_complete': 3, '/failed': 0, '/gpu_idle': True}),
        condition(copath / 'identity.json', {'/manifest_sha256': sha(ROOT / 'manifests/CO_MAXN_SINGLE_SEED.json'),
                                             '/runtime_manifest_sha256': sha(ROOT / 'co_bundle/runtime/RUNTIME_MANIFEST.json')}),
        condition(ROOT / 'CO_PARENT_EXIT.json', {'/returncode': 0, '/normal_parent_exit': True, '/verified_full_identities': 3}),
        condition(ROOT / 'UPSTREAM_VERIFIED.json', {'/status': 'VERIFIED', '/benchmark_slots_complete': 292})],
        failure_files=[str(copath / 'series_failure.json'), str(ROOT / 'CO_STAGE_FAILED.json'), str(ROOT / 'FOLLOWER_STOPPED.json')],
        wait_processes=[old_id], note='Real Co completion/identity/done records and normal reaped parent exit; no clock/global-flag gate')
    for job in cojobs:
        ecgate['conditions'].append(condition(copath / (job['id'] + '.done.json'), {'/id': job['id']}))
        ecgate['conditions'].append(condition(copath / job['id'] / 'summary.json',
            {'/workload': job['workload'], '/seed': 0, '/condition': 'BASE', '/full_sequence': True, '/diagnostic': False,
             '/finite_parameters_and_poses': True, '/selected_calls/tracking': 300, '/selected_calls/bundle_adjustment': 60,
             '/frames': 592 if job['workload'].startswith('Tum') else 2000,
             '/source_manifest_sha256': sha(ROOT / 'co_bundle/runtime/RUNTIME_MANIFEST.json'),
             '/protocol_sha256': sha(ROOT / 'co_bundle/runtime/PROTOCOL.json'),
             '/initial_power': 'NV Power Mode: MAXN\n0', '/final_power': 'NV Power Mode: MAXN\n0'}))
    save(ROOT / 'gates/EC_AFTER_CO_SUCCESS.json', ecgate)
    save(ROOT / 'INSTALLATION_CONFIG.json', config)
    for system, bundle, queue in [('CO', 'co_bundle', 'co_maxn'), ('EC', 'bundle', 'ec_maxn')]:
        save(ROOT / f'{system}_COMPONENT_PROFILE_ACTIVE.json', dict(queue=str(ROOT / 'results' / queue),
            runtime=str(ROOT / bundle / 'runtime'), manifest=str(ROOT / 'manifests' / f'{system}_MAXN_SINGLE_SEED.json')))
    selections = {}
    for system, bundle in [('CO', 'co_bundle'), ('EC', 'bundle')]:
        protocol = read(ROOT / bundle / 'runtime/PROTOCOL.json')
        selections[system] = dict(protocol_sha256=sha(ROOT / bundle / 'runtime/PROTOCOL.json'), workloads=protocol['workloads'])
    save(ROOT / 'FROZEN_SELECTIONS.json', selections)
    save(ROOT / '15W_deferred/CONTINUATION_PLAN.json', dict(status='PREPARED_NOT_ARMED', mode='MODE_15W',
        manual_mode_query_required=True, benchmark_slots=292, Co_profiles=3, EC_profiles=2, total_full_slots=297,
        diagnostics=dict(Co=3, EC=2), order=['future 15 W benchmarks and planned exports', 'Co preflights/full profiles',
        'Co normal parent exit', 'EC preflights/full profiles', 'combined idle export and one decision email'],
        missing_binding='Actual future 15 W native controls, successful terminal/export records and parent identities do not yet exist'))
    for name in ['FOLLOWER_STATUS.json', 'FOLLOWER_IDENTITY.json', 'INSTALLATION_CONFIG.json', 'INSTALLATION_FREEZE.json',
                 'INSTALL_VERIFIED.json', 'UPSTREAM_NOTIFICATION_DEFERRED.json', '15W_deferred/15W_TEMPLATE_NOT_LAUNCHED.json']:
        path = OLD / name
        if path.exists():
            dst = ROOT / 'preserved_previous' / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(path.read_bytes())
    text = '#!/bin/bash\nset -euo pipefail\nsource /home/yh279050/miniforge3/etc/profile.d/conda.sh\nconda activate coslam\nexport PYTHONDONTWRITEBYTECODE=1\ncd /home/yh279050/work/Co-SLAM\nexec python -B -u ' + shlex.quote(str(ROOT / 'continuation.py')) + ' >> ' + shlex.quote(str(ROOT / 'follower.log')) + ' 2>&1\n'
    (ROOT / 'launch_follow_maxn.sh').write_text(text)
    os.chmod(ROOT / 'launch_follow_maxn.sh', 0o755)
    durations = [read(c['summary_path'])['slam_work_wall_seconds'] for c in config['native_profile_controls'] + config['native_ec_controls']]
    save(ROOT / 'ETA_CONTEXT.json', dict(utc=now(), native_BASE_proxy_hours=sum(durations) / 3600,
        provisional_extra_profiling_hours=[1.5, 3.0], note='Five full profiles plus five bounded local diagnostics; proxy is not a calibrated profiling ETA. Refresh from actual Orin profiles; benchmark forecast excludes this extension.'))
    files = {str(p): sha(p) for p in ROOT.rglob('*') if p.is_file() and p.suffix != '.log' and '__pycache__' not in p.parts and p.name != 'INSTALLATION_FREEZE.json'}
    files.update(read(DP / 'EXECUTION_FREEZE.json')['files'])
    files.update(read(DP / 'INSTALLATION_FREEZE.json')['files'])
    for control in config['native_profile_controls'] + config['native_ec_controls']:
        for key in ['config_path', 'summary_path', 'completion_path']:
            files[control[key]] = sha(control[key])
    for name in ['orin_component_profile_runtime_v2.zip', 'orin_ec_component_profile_runtime_v2_single_seed.zip']:
        files[str(TOP / name)] = sha(TOP / name)
    save(ROOT / 'INSTALLATION_FREEZE.json', dict(utc=now(), files=files))
    print('Single-seed setup resolved: Co 3 + EC 2; current benchmark gates preserved')


if __name__ == '__main__':
    main()
