"""Durable normal-upstream -> supplied controller -> idle export -> decision wake."""
import argparse
import sys
import traceback
from local_support import *
from export_profiles import audit_upstream, export_profiles


def main():
    config = read(ROOT / 'INSTALLATION_CONFIG.json')
    gate = read(ROOT / 'gates/UPSTREAM_SUCCESS_AND_EXPORTS.json')
    with open('/tmp/dasip_round2_component_profile_maxn_v2.lock', 'a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(ROOT / 'FOLLOWER_IDENTITY.json', process_identity(os.getpid()))
        state = dict(phase='WAITING_FOR_SUCCESSFUL_MAXN_PROGRAMME_AND_EXPORTS',
                     follower_pid=os.getpid(), controller_pid=None, profile_full_required=6,
                     profile_diagnostics_required=3, GPU_work_launched=False,
                     final_export_and_notification_armed=True,
                     current_benchmarks_modified=False, automatic_power_or_clock_changes=False)
        def snapshot(reason):
            state.update(utc=now(), reason=reason)
            save(ROOT / 'FOLLOWER_STATUS.json', state)
            print(now(), state['phase'], reason, flush=True)
        frozen_sources_ok()
        while True:
            ok, reason = gate_ready(gate)
            dp = read(DP / 'DP_COMPLETION_STATUS.json')
            state['upstream_counts'] = dp['counts']
            if dp['phase'] == 'COMPLETE_ADMITTED_DP_SERIES':
                if any(j['status'] not in ['done', 'reused'] for j in dp['jobs']):
                    raise RuntimeError('Upstream ended with unresolved identities; profiling is not admitted')
            if ok:
                break
            snapshot(reason)
            time.sleep(30)
        # Bulk archive/data-evidence audit is explicitly deferred until this boundary.
        with open(GPU_LOCK, 'a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    snapshot('All success gates passed; waiting for exclusive GPU ownership')
                    time.sleep(15)
            assert gate_ready(gate)[0]
            samples = fresh_idle_three()
            state['phase'] = 'IDLE_UPSTREAM_EXPORT_AND_IDENTITY_AUDIT'
            snapshot('All current benchmarks, expected exports and captured parents finished normally')
            proof = audit_upstream(config)
            save(ROOT / 'IDLE_ADMISSION.json', dict(utc=now(), idle=samples, upstream_proof=proof,
                                                   shared_lock=GPU_LOCK, power=require_maxn()))
            verify = subprocess.run([sys.executable, '-B', str(ROOT / 'bundle/verify_component_profile_handoff.py')],
                                    capture_output=True, text=True)
            save(ROOT / 'BOUNDARY_PACKAGE_VERIFICATION.json', dict(utc=now(), returncode=verify.returncode,
                                                                   stdout=verify.stdout, stderr=verify.stderr))
            assert verify.returncode == 0, 'Profiling package verification failed at idle boundary'
        # Release before invoking the supplied controller, which acquires the same lock itself.
        command = [sys.executable, '-B', '-u', str(ROOT / 'bundle/run_component_profile_campaign.py'),
                   '--runtime', str(ROOT / 'bundle/runtime'),
                   '--manifest', str(ROOT / 'manifests/MAXN_RESOLVED.json'),
                   '--gate', str(ROOT / 'gates/CONTROLLER_GATE.json'),
                   '--output', str(ROOT / 'results/maxn')]
        save(ROOT / 'CONTROLLER_LAUNCH.json', dict(utc=now(), command=command, inherited_gpu_lock=False))
        with (ROOT / 'controller.log').open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            save(ROOT / 'CONTROLLER_IDENTITY.json', process_identity(child.pid))
            state.update(phase='SUPPLIED_PROFILING_CONTROLLER_RUNNING', controller_pid=child.pid,
                         GPU_work_launched=True)
            snapshot('Three supplied local fixed-state/50-frame checks, then all six BASE full profiles')
            while child.poll() is None:
                state['profile_status'] = read(ROOT / 'results/maxn/status.json') if (ROOT / 'results/maxn/status.json').exists() else None
                save(ROOT / 'FOLLOWER_STATUS.json', dict(state, utc=now()))
                time.sleep(10)
            code = child.wait()
        save(ROOT / 'CONTROLLER_EXIT.json', dict(utc=now(), returncode=code))
        state.update(phase='WAITING_FOR_IDLE_FINAL_PROFILE_EXPORT', controller_pid=None)
        snapshot('Supplied controller exited; exact preflight/full failures remain preserved')
        with open(GPU_LOCK, 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            samples = fresh_idle_three()
            save(ROOT / 'FINAL_EXPORT_IDLE.json', dict(utc=now(), samples=samples,
                                                      controller_exited=True, shared_lock=GPU_LOCK))
            receipt = export_profiles(config, code == 0)
        state.update(phase='COMPLETE' if receipt['campaign_success'] else 'PROFILING_FAILED_EVIDENCE_EXPORTED')
        snapshot('Export verified; no automatic mode switch or next series')
        if not receipt['campaign_success']:
            wake(config, 'The separate authorized Orin component profiling block failed. Read ' + str(ROOT / 'PROFILING_COMPLETE.json') +
                 ', ' + str(ROOT / 'EXPORT_COMPLETE.json') + ' and controller/diagnostic logs. Preserve exact errors; do not relax tolerances, alter numerical behavior, repeat completed benchmarks, change power/clocks, or email for this isolated failure. Report the failure and pending profiles to the author.', 'FAILURE_WAKE_RECEIPT.json')
            return
        samples = fresh_idle_three()
        save(ROOT / 'EMAIL_CAMPAIGN_STATUS.json', dict(utc=now(), major_series_terminal=True,
             profiling_and_export_successful=True, fresh_host_idle=samples, email_sent=False))
        message = ('The complete user-authorized Orin MAXN benchmark programme AND its separate component-profiling continuation and final export have finished normally. Read ' +
                   str(ROOT / 'PROFILING_COMPLETE.json') + ', ' + str(ROOT / 'EXPORT_COMPLETE.json') + ', ' +
                   str(ROOT / 'UPSTREAM_VERIFIED.json') + ', ' + str(ROOT / 'results/component_profile_analysis_v1/REPORT.md') +
                   ' and ' + str(DP / 'DP_COMPLETION_STATUS.json') + '. Verify the profile ZIP, SHA256SUMS/path mapping, and referenced prior exports. ' +
                   'Recheck fresh host GPU idle and experiment program status immediately before notifying. State benchmark 292/292 and profiling 6/6 plus diagnostics 3/3 truthfully, list any unfinished MAXN work and keep EC ENG excluded. ' +
                   'Only if this complete major series, successful final export and fresh idle are confirmed, send ONE authorized concise decision email to youssefhaddouda233@gmail.com via connected Gmail. ' +
                   'Check ' + str(ROOT / 'EMAIL_SENT.json') + ', ' + str(DP / 'EMAIL_SENT.json') + ' and Gmail Sent for duplicates. The DP EMAIL_WAKE_RECEIPT.json is an explicit deferral guard, NOT a sent receipt. Save a real Gmail sent-message ID/receipt only after success. ' +
                   'Report the ZIP, completion counts/failures and whether all MAXN work is finished so the author can switch manually. The six-profile 15 W template is deferred until the END of the future author-authorized 15 W benchmark programme; do not launch it, switch modes/clocks or start any next series. If Gmail transport is unavailable, state that without claiming delivery. Idle evidence: ' + str(ROOT / 'EMAIL_CAMPAIGN_STATUS.json'))
        wake(config, message, 'EMAIL_WAKE_RECEIPT.json')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        save(ROOT / 'FOLLOWER_STOPPED.json', dict(utc=now(), error=repr(error), traceback=traceback.format_exc(),
                                                major_series_complete=False, email_sent=False))
        try:
            wake(read(ROOT / 'INSTALLATION_CONFIG.json'), 'The Orin component-profiling continuation is blocked/stopped. Inspect ' +
                 str(ROOT / 'FOLLOWER_STOPPED.json') + '. Preserve the exact failure, all existing benchmark jobs/exports and profiling evidence. Do not send an isolated-failure email, bypass normal completion/export/idle gates, change computation/tolerances or power/clocks. Report actual pending work and repair only orchestration if justified.', 'ORCHESTRATION_FAILURE_WAKE_RECEIPT.json')
        except Exception:
            pass
        raise
