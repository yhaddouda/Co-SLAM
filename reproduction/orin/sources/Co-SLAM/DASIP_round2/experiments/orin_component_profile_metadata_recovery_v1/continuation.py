"""292-slot programme/exports -> Co seed0 profiles -> EC seed0 profiles -> combined export."""
import sys
import traceback
from local_support import *
from audit_upstream import audit_upstream
from export_profiles import validate_series, package


def launch(system):
    co = system == 'CO'
    controller = ROOT / ('co_bundle/run_component_profile_campaign.py' if co else 'bundle/run_ec_component_profile_campaign.py')
    runtime = ROOT / ('co_bundle/runtime' if co else 'bundle/runtime')
    gate = ROOT / 'gates' / ('CO_CONTROLLER_GATE.json' if co else 'EC_AFTER_CO_SUCCESS.json')
    queue = ROOT / 'results' / ('co_maxn' if co else 'ec_maxn')
    command = [sys.executable, '-B', '-u', str(controller), '--runtime', str(runtime), '--manifest',
               str(ROOT / 'manifests' / f'{system}_MAXN_SINGLE_SEED.json'), '--gate', str(gate), '--output', str(queue)]
    save(ROOT / f'{system}_CONTROLLER_LAUNCH.json', dict(utc=now(), command=command, GPU_lock_inherited=False))
    with (ROOT / f'{system}_controller.log').open('x') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, close_fds=True)
    save(ROOT / f'{system}_CONTROLLER_IDENTITY.json', process_identity(child.pid))
    return child


def audit_ec_controls(config):
    records = []
    for c in config['native_ec_controls']:
        for path, expected in [(c['config_path'], c['config_sha256']), (c['summary_path'], c['summary_sha256']),
                               (c['completion_path'], c['completion_sha256']), (c['source_path'], c['source_sha256'])]:
            assert sha(path) == expected, path
        assert read(c['completion_path'])['returncode'] == 0
        assert read(c['config_path'])['data']['input_folder'] == c['datadir']
        assert Path(c['datadir']).is_dir()
        # Full native layout/count checks are performed by the supplied Orin diagnostics.
        records.append(c)
    save(ROOT / 'EC_NATIVE_CONTROL_AUDIT.json', dict(utc=now(), status='VERIFIED', controls=records))


def main():
    config = read(ROOT / 'INSTALLATION_CONFIG.json')
    gate = read(ROOT / 'gates/UPSTREAM_SUCCESS_AND_EXPORTS.json')
    with open('/tmp/dasip_round2_components_single_seed_maxn_v2.lock', 'a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(ROOT / 'FOLLOWER_IDENTITY.json', process_identity(os.getpid()))
        state = dict(phase='WAITING_FOR_SETUP_HANDOFF', follower_pid=os.getpid(), co_controller_pid=None,
            ec_controller_pid=None, full_required=dict(CO=3, EC=2), diagnostics_required=dict(CO=3, EC=2),
            final_combined_export_and_notification_armed=True, GPU_work_launched=False, current_benchmarks_modified=False)
        def snapshot(reason):
            state.update(utc=now(), reason=reason)
            save(ROOT / 'FOLLOWER_STATUS.json', state)
            print(now(), state['phase'], reason, flush=True)
        frozen_sources_ok()
        # A real detached EC controller waits on Co's exact normal completion/exit.
        ec = launch('EC')
        state['ec_controller_pid'] = ec.pid
        while not (ROOT / 'ARMED.json').exists():
            snapshot('New waiting controllers installed; old waiting Co follower must be verified retired before admission')
            time.sleep(5)
        assert read(ROOT / 'ARMED.json')['status'] == 'VERIFIED_HANDOFF'
        assert not still_alive(config['retired_waiting_follower'])
        state['phase'] = 'WAITING_FOR_SUCCESSFUL_MAXN_PROGRAMME_AND_EXPORTS'
        while True:
            ok, reason = gate_ready(gate)
            dp = read(DP / 'DP_COMPLETION_STATUS.json')
            state['upstream_counts'] = dp['counts']
            if dp['phase'] == 'COMPLETE_ADMITTED_DP_SERIES' and any(j['status'] not in ['done', 'reused'] for j in dp['jobs']):
                raise RuntimeError('Original MAXN programme ended with unresolved targets; profiles are not admitted')
            if ec.poll() is not None:
                raise RuntimeError('Detached EC waiting controller stopped; preserved EC_controller.log and series_failure.json')
            if ok:
                break
            snapshot(reason)
            time.sleep(30)
        with open(GPU_LOCK, 'a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    snapshot('Successful original programme/export gates passed; waiting for exclusive ownership')
                    time.sleep(15)
            assert gate_ready(gate)[0]
            samples = fresh_idle_three()
            state['phase'] = 'IDLE_UPSTREAM_EXPORT_AND_IDENTITY_AUDIT'
            snapshot('All old MAXN benchmark/DP parents and planned exports finished normally')
            proof = audit_upstream(config)
            audit_ec_controls(config)
            save(ROOT / 'IDLE_ADMISSION.json', dict(utc=now(), shared_lock=GPU_LOCK, samples=samples, proof=proof))
            for bundle, verifier in [('co_bundle', 'verify_component_profile_handoff.py'), ('bundle', 'verify_ec_component_profile_handoff.py')]:
                result = subprocess.run([sys.executable, '-B', str(ROOT / bundle / verifier)], capture_output=True, text=True)
                save(ROOT / (bundle + '_boundary_verification.json'), dict(utc=now(), returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
                assert result.returncode == 0, bundle + ' boundary verification failed'
        co = launch('CO')
        state.update(phase='CO_PROFILING_CONTROLLER_RUNNING', co_controller_pid=co.pid, GPU_work_launched=True)
        snapshot('Three local Co checks then three full Co profiles; EC remains gated')
        while co.poll() is None:
            if (ROOT / 'results/co_maxn/status.json').exists():
                state['co_status'] = read(ROOT / 'results/co_maxn/status.json')
            save(ROOT / 'FOLLOWER_STATUS.json', dict(state, utc=now()))
            time.sleep(10)
        code = co.wait()
        success = code == 0
        if success:
            try:
                records = validate_series('CO')
            except Exception as error:
                success = False
                save(ROOT / 'CO_STAGE_FAILED.json', dict(utc=now(), error=repr(error), traceback=traceback.format_exc()))
        if success:
            assert not still_alive(read(ROOT / 'CO_CONTROLLER_IDENTITY.json'))
            # Written only after wait() reaps the actual successful Co parent and identities validate.
            save(ROOT / 'CO_PARENT_EXIT.json', dict(utc=now(), returncode=0, normal_parent_exit=True,
                verified_full_identities=3, parent=read(ROOT / 'CO_CONTROLLER_IDENTITY.json'), runs=records))
            state.update(phase='EC_PROFILING_CONTROLLER_RUNNING', co_controller_pid=None)
            snapshot('All three Co profiles complete and parent reaped; two local EC checks then two full EC profiles')
        else:
            if not (ROOT / 'CO_STAGE_FAILED.json').exists():
                save(ROOT / 'CO_STAGE_FAILED.json', dict(utc=now(), returncode=code, reason='Supplied Co controller failed; exact logs preserved'))
            snapshot('Co failed; EC gate rejects launch; preserve failure evidence without relaxing checks')
        while ec.poll() is None:
            if (ROOT / 'results/ec_maxn/status.json').exists():
                state['ec_status'] = read(ROOT / 'results/ec_maxn/status.json')
            save(ROOT / 'FOLLOWER_STATUS.json', dict(state, utc=now()))
            time.sleep(10)
        ec_code = ec.wait()
        save(ROOT / 'EC_PARENT_EXIT.json', dict(utc=now(), returncode=ec_code,
            normal_parent_exit=ec_code == 0, parent=read(ROOT / 'EC_CONTROLLER_IDENTITY.json')))
        success = success and ec_code == 0
        state.update(phase='WAITING_FOR_IDLE_COMBINED_EXPORT', co_controller_pid=None, ec_controller_pid=None)
        snapshot('Both profiling parents exited; all raw evidence retained')
        with open(GPU_LOCK, 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            save(ROOT / 'FINAL_EXPORT_IDLE.json', dict(utc=now(), samples=fresh_idle_three(), shared_lock=GPU_LOCK))
            receipt = package(config, success)
        state['phase'] = 'COMPLETE' if receipt['campaign_success'] else 'PROFILING_FAILED_EVIDENCE_EXPORTED'
        snapshot('Combined Co+EC export verified; no intermediate Co email or automatic mode change')
        if not receipt['campaign_success']:
            wake(config, 'The authorized combined Orin Co/EC single-seed profiling block failed. Read ' + str(ROOT / 'PROFILING_COMPLETE.json') +
                ', ' + str(ROOT / 'EXPORT_COMPLETE.json') + ' and original failure logs. Preserve exact failed checks and successful upstream work. Do not relax tolerances, change computation or repeat completed benchmark/profile jobs. Do not send an isolated-failure email or change power/clocks.', 'FAILURE_WAKE_RECEIPT.json')
            return
        samples = fresh_idle_three()
        save(ROOT / 'EMAIL_CAMPAIGN_STATUS.json', dict(utc=now(), major_series_terminal=True,
            combined_Co_EC_profiles_and_export_complete=True, fresh_host_idle=samples, email_sent=False))
        message = ('The current authorized Orin MAXN programme and updated seed-0-only Co+EC component profiling series have finished normally, with final combined export. Read ' +
            str(ROOT / 'PROFILING_COMPLETE.json') + ', ' + str(ROOT / 'EXPORT_COMPLETE.json') + ', ' + str(ROOT / 'UPSTREAM_VERIFIED.json') +
            ', both reports under ' + str(ROOT / 'results') + ', and ' + str(DP / 'DP_COMPLETION_STATUS.json') +
            '. Verify the ZIP, digest, entry hashes and prior export references. Recheck fresh host GPU idle and experiment programme status immediately before notification. ' +
            'State actual counts truthfully: unchanged benchmarks 292, Co profiles 3, EC profiles 2 = 297 full slots; five bounded diagnostics separate. All Orin profiles seed 0 only; no across-seed SD. EC ENG remains excluded. List all pending/blocked work if any. ' +
            'Only for a successfully completed major combined series, verified export and fresh idle, send ONE authorized concise decision email to youssefhaddouda233@gmail.com via connected Gmail. ' +
            'Check ' + str(ROOT / 'EMAIL_SENT.json') + ', earlier profiling/DP sent receipts and Gmail Sent to prevent duplicates. Wake guards/DEFERRED receipts are not delivery. Save a real sent message ID only after Gmail success. ' +
            'Report the combined results ZIP, completion counts and failures and whether MAXN is finished for the manual 15 W decision. Do not start any next series, change modes/clocks, run extra quality/mesh/kernel experiments or launch 15 W. ' +
            'The matching five-profile 15 W template is prepared but must follow the END of the future author-selected 15 W benchmarks/exports with actual future gates. If Gmail transport is unavailable, state it without claiming delivery. Idle evidence: ' + str(ROOT / 'EMAIL_CAMPAIGN_STATUS.json'))
        wake(config, message, 'EMAIL_WAKE_RECEIPT.json')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(ROOT / 'FOLLOWER_STOPPED.json', dict(utc=now(), error=repr(error), traceback=traceback.format_exc(), email_sent=False))
        try:
            wake(read(ROOT / 'INSTALLATION_CONFIG.json'), 'The authorized Orin single-seed Co+EC profiling continuation has an orchestration/gate failure. Read ' +
                str(ROOT / 'FOLLOWER_STOPPED.json') + '. Preserve all running valid benchmark/profile jobs and completed evidence. Repair only justified orchestration without bypassing normal gates, relaxing checks, changing numerical behavior, repeating successful jobs or changing mode/clocks. No isolated-failure email.', 'ORCHESTRATION_FAILURE_WAKE_RECEIPT.json')
        except Exception:
            pass
        raise
