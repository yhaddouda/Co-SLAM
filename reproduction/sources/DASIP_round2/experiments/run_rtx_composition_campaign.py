#!/usr/bin/env python3
"""Detached continuation: missing MC+DP full runs, then all new mesh scores."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from checked_reconstruction import save_json
from report_support_reference import gpu_idle, running


def main():
    base = Path(__file__).resolve().parents[1]
    output = base / 'results/composition_campaign_v1'
    output.mkdir(exist_ok=False)
    state = dict(status='WAITING_FOR_IDLE_GPU', training_completed=0, training_total=72,
                 reconstruction_completed=0, reconstruction_total=45, automatic_email=False)

    def snapshot():
        state['updated_paris'] = datetime.now(ZoneInfo('Europe/Paris')).isoformat()
        state['training_completed'] = len(list((base / 'results/composition_v1').glob('*.done.json')))
        state['reconstruction_completed'] = len(list((base / 'results/reconstruction_composition_v1').glob('*.done.json')))
        save_json(output / 'status.json', state)
        text = '# RTX MC+DP completion campaign\n\n'
        text += f"Updated {state['updated_paris']}. **{state['status']}**. Full MC+DP runs: **{state['training_completed']}/72**; checked new meshes: **{state['reconstruction_completed']}/45**.\n\n"
        text += 'The earlier RTX BASE/DP/MC, EC, engineering and 279-score queues are complete. This closes the remaining declared composition gap: all 24 configured native-capacity scenes, seeds 0/1/2, unchanged runtime v1, with existing matched BASE/MC/DP controls. Both timing scopes and ATE are collected; only Replica/Synthetic export and score meshes. No new sampling variant, H19 pruning or ENG+DP is introduced. Kernel profiling remains author-owned.\n\n'
        text += 'The detached serial parent continues through training, raw-output audit and scoring without this chat. Status: results/composition_campaign_v1/status.json. The paper updater incorporates complete three-seed comparisons; pending/negative results remain visible.\n\n'
        text += 'Orin is separate: the author reports its MAXN ScanNet series and waiting DP continuation; see ../ORIN_15W_DEADLINE_PLAN.md. No automatic mode switch or routine email.\n'
        (output / 'LIVE_STATUS.md').write_text(text)
        (base / 'results/LIVE_STATUS.md').write_text(text)

    def run(argv, label):
        with (output / (label + '.log')).open('w') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                state['child_pid'] = child.pid
                snapshot()
                time.sleep(10)
        state.pop('child_pid', None)
        if child.returncode:
            raise RuntimeError(f'{label} failed with {child.returncode}; original evidence retained')

    try:
        while gpu_idle()['idle'] is not True or running():
            snapshot()
            time.sleep(15)
        state['status'] = 'RUNNING_FULL_MC_DP'; snapshot()
        run([sys.executable, str(base / 'experiments/run_manifest_queue.py'),
             '--runtime', str(base / 'experiments/coslam_reference_runtime_v1'),
             '--manifest', str(base / 'experiments/manifests/rtx_composition_all_scenes.json'),
             '--output', str(base / 'results/composition_v1')], 'training')
        state['status'] = 'AUDITING_FULL_MC_DP'; snapshot()
        run([sys.executable, str(base / 'experiments/report_composition_results.py')], 'audit')
        save_json(output / 'campaign_complete.json', dict(training_complete=True, policy='Upstream marker for serial reconstruction; full series status remains in status.json'))
        state['status'] = 'SCORING_45_NEW_MESHES'; snapshot()
        run([sys.executable, str(base / 'experiments/run_reconstruction_campaign.py'),
             '--after', str(output),
             '--runtime', str(base / 'experiments/reconstruction_runtime_v2'),
             '--manifest', str(base / 'experiments/manifests/rtx_reconstruction_composition.json'),
             '--output', str(base / 'results/reconstruction_composition_v1'),
             '--cache', str(base / 'results/reconstruction_v1/gt_cache')], 'reconstruction')
        scored = json.loads((base / 'results/reconstruction_composition_v1/campaign_complete.json').read_text())
        state['status'] = 'COMPLETE' if scored['failed'] == 0 else 'COMPLETE_WITH_SCORING_FAILURES'
        snapshot()
        save_json(output / 'series_complete.json', dict(**state, gpu_snapshot=gpu_idle()))
        (output / 'DECISIONS_READY.md').write_text('The RTX MC+DP series has finished. Review status.json, raw-output composition report, reconstruction failures if any and the updated paper. No unattended email transport is configured; no email delivery is claimed.\n')
    except BaseException as error:
        state.update(status='STOPPED: ' + str(error))
        snapshot()
        save_json(output / 'campaign_failure.json', state)
        raise


if __name__ == '__main__':
    main()
