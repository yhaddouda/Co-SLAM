#!/usr/bin/env python3
"""Audit completed runtime-v1 outputs and report paired means without GPU work."""
import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from zoneinfo import ZoneInfo


def read_json(path):
    for attempt in range(3):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(.1)


def running():
    with open('/tmp/dasip_round2_gpu0.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    return False


def rigid_rmse(truth, estimate):
    import numpy as np
    truth = truth[:, :3, 3].astype(np.float64)
    estimate = estimate[:, :3, 3].astype(np.float64)
    valid = np.isfinite(truth).all(axis=1) & np.isfinite(estimate).all(axis=1)
    assert valid.sum() >= 3
    target = truth[valid] - truth[valid].mean(axis=0)
    source = estimate[valid] - estimate[valid].mean(axis=0)
    # Column-vector Kabsch formulation, independently read from saved poses.
    u, _, vt = np.linalg.svd(target.T @ source)
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    errors = source @ rotation.T - target
    return float(np.sqrt(np.mean(np.einsum('ij,ij->i', errors, errors))))


def audit_job(queue, job, identity):
    import numpy as np
    directory = queue / job['id']
    result = read_json(directory / 'summary.json')
    config = read_json(directory / 'config.json')
    assert hashlib.sha256((directory / 'source_manifest.json').read_bytes()).hexdigest() == identity['runtime_manifest_sha256']
    assert result['condition'] == job['condition'] and result['seed'] == job['seed']
    assert result['dataset'] == job['dataset'] and result['scene'] == job['scene']
    assert result['full_sequence'] and not result['diagnostic'] and result['finite_parameters_and_poses']
    assert result['quality'] and config['round2']['engineering'] is False
    if job.get('export_mesh'):
        assert (directory / 'mesh_export.json').is_file()
        assert list((directory / 'artifacts').glob('mesh_track*.ply'))
    poses = np.load(directory / 'trajectories.npz', allow_pickle=False)
    assert np.array_equal(poses['frame_ids'], np.arange(result['frames']))
    for key, field in [('estimated', 'ate_rmse_m'), ('corrected', 'corrected_ate_rmse_m')]:
        assert np.isfinite(poses[key]).all()
        recomputed = rigid_rmse(poses['gt'], poses[key]) / config['data']['sc_factor']
        assert math.isclose(recomputed, result[field], rel_tol=1e-7, abs_tol=1e-9), (job['id'], field)
    with (directory / 'calls.csv').open() as file:
        calls = list(csv.DictReader(file))
    row = dict(id=job['id'], dataset=job['dataset'], scene=job['scene'], seed=job['seed'],
               condition=job['condition'], frames=result['frames'],
               ate_cm=100 * result['ate_rmse_m'], corrected_ate_cm=100 * result['corrected_ate_rmse_m'])
    for stage, prefix in [('tracking', 'tracking'), ('bundle_adjustment', 'ba')]:
        selected = [r for r in calls if r['stage'] == stage and int(r['frame_id']) >= result['warmup_frames']]
        expected = set(range(max(1, result['warmup_frames']), result['frames']))
        if stage == 'bundle_adjustment':
            expected = {i for i in expected if i % config['mapping']['map_every'] == 0}
        assert len(selected) == len(expected) and {int(r['frame_id']) for r in selected} == expected
        wall = statistics.mean(float(r['wall_ms']) for r in selected)
        cuda = statistics.mean(float(r['cuda_ms']) for r in selected)
        assert wall > 0 and math.isfinite(wall) and cuda >= 0 and math.isfinite(cuda)
        assert math.isclose(wall, result[stage]['mean_wall_ms'], rel_tol=1e-9)
        row[prefix + '_wall_ms'] = wall
        row[prefix + '_cuda_ms'] = cuda
        row[prefix + '_pruned_pct'] = result[stage]['queries_pruned_pct']
        counts = result[stage]
        assert 0 < counts['actual_queries'] <= counts['dense_queries']
        if job['condition'] in ('BASE', 'MC'):
            assert counts['actual_queries'] == counts['dense_queries']
        else:
            assert counts['actual_queries'] < counts['dense_queries']
    clean_config = json.loads(json.dumps(config))
    clean_config.pop('round2')
    clean_config['data'].pop('output')
    row['effective_config_sha256'] = hashlib.sha256(json.dumps(clean_config, sort_keys=True).encode()).hexdigest()
    return row


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def gpu_idle():
    try:
        result = subprocess.run(['nvidia-smi', '--id=0', '--query-compute-apps=pid,process_name',
                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
        if result.returncode:
            return dict(idle=None, error=result.stderr.strip())
        processes = [line for line in result.stdout.splitlines() if line.strip()]
        return dict(idle=not processes, compute_processes=processes)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(idle=None, error=str(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    queue = args.queue.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = read_json(queue / 'job_manifest.json')['jobs']
    assert all(j['condition'] in ('BASE', 'DP') and j['quality'] for j in jobs)
    if args.wait:
        while not (queue / 'queue_complete.json').is_file():
            if not running():
                break
            time.sleep(15)
    identity = read_json(queue / 'queue_identity.json')
    completed = [j for j in jobs if (queue / (j['id'] + '.done.json')).is_file()]
    full = len(completed) == len(jobs) and (queue / 'queue_complete.json').is_file()
    rows = [audit_job(queue, j, identity) for j in completed]
    grouped = {}
    for row in rows:
        grouped.setdefault((row['dataset'], row['scene']), {}).setdefault(row['condition'], {})[row['seed']] = row
    comparisons = []
    for (dataset, scene), conditions in sorted(grouped.items()):
        if set(conditions.get('BASE', {})) != {0, 1, 2} or set(conditions.get('DP', {})) != {0, 1, 2}:
            continue
        for seed in (0, 1, 2):
            assert conditions['BASE'][seed]['effective_config_sha256'] == conditions['DP'][seed]['effective_config_sha256']
            assert conditions['BASE'][seed]['frames'] == conditions['DP'][seed]['frames']
        comparison = dict(dataset=dataset, scene=scene, paired_seeds=3)
        for metric in ('ate_cm', 'corrected_ate_cm', 'tracking_wall_ms', 'ba_wall_ms',
                       'tracking_cuda_ms', 'ba_cuda_ms', 'tracking_pruned_pct', 'ba_pruned_pct'):
            for condition in ('BASE', 'DP'):
                values = [conditions[condition][seed][metric] for seed in (0, 1, 2)]
                comparison[condition + '_' + metric] = statistics.mean(values)
                comparison[condition + '_' + metric + '_sd'] = statistics.stdev(values)
            base = comparison['BASE_' + metric]
            candidate = comparison['DP_' + metric]
            if metric in ('ate_cm', 'corrected_ate_cm'):
                comparison[metric + '_increase_pct'] = 100 * (candidate / base - 1) if base else None
            elif metric.endswith('_ms'):
                comparison[metric + '_reduction_pct'] = 100 * (1 - candidate / base)
        comparison['tracking_tolerated_target_met'] = comparison['tracking_wall_ms_reduction_pct'] >= 5
        comparison['ba_tolerated_target_met'] = comparison['ba_wall_ms_reduction_pct'] >= 25
        comparison['direct_ate_target_met'] = comparison['ate_cm_increase_pct'] is not None and comparison['ate_cm_increase_pct'] <= 15
        comparison['corrected_ate_target_met'] = comparison['corrected_ate_cm_increase_pct'] is not None and comparison['corrected_ate_cm_increase_pct'] <= 15
        comparisons.append(comparison)
    now = datetime.now(ZoneInfo('Europe/Paris')).isoformat()
    gpu = gpu_idle() if full else dict(idle=None, reason='Major series incomplete; no email eligibility')
    status = dict(updated_paris=now, completed_jobs=len(completed), total_jobs=len(jobs),
                  full_series_complete=full, scenes_with_three_paired_seeds=len(comparisons),
                  trajectory_ate_and_call_csv_audits_passed=True, reconstruction_scoring='pending',
                  final_ate_reporting_convention='direct and corrected both reported; primary table convention remains to freeze',
                  gpu_snapshot=gpu, email_review_point=full and gpu['idle'] is True,
                  email_sent=False, unattended_email_transport_configured=False, queue_identity=identity)
    report = '# Restored-reference BASE versus conservative DP\n\n'
    report += f'Updated {now} (Europe/Paris). **{len(completed)}/{len(jobs)} jobs complete.** '
    report += 'Full major series finished.\n\n' if full else 'Partial results; the series is not complete.\n\n'
    report += 'Three paired seeds per reported scene; means and sample SD. Original pose files were independently realigned to audit both saved ATE values. Whole-call wall/CUDA means and post-warmup frame coverage were checked against raw calls. Matched effective configs and source manifests were verified. Final meshes exist where requested; reconstruction metric scoring remains pending.\n\n'
    report += '| Dataset / scene | BASE direct ATE cm | DP direct ATE cm | Change | BASE corrected ATE cm | DP corrected ATE cm | Change |\n|---|---:|---:|---:|---:|---:|---:|\n'
    def value(c, condition, metric):
        return f"{c[condition + '_' + metric]:.3f} ± {c[condition + '_' + metric + '_sd']:.3f}"
    def change(number):
        return 'undefined (zero baseline)' if number is None else f'{number:+.2f}%'
    for c in comparisons:
        report += f"| {c['dataset']} / {c['scene']} | {value(c,'BASE','ate_cm')} | {value(c,'DP','ate_cm')} | {change(c['ate_cm_increase_pct'])} | {value(c,'BASE','corrected_ate_cm')} | {value(c,'DP','corrected_ate_cm')} | {change(c['corrected_ate_cm_increase_pct'])} |\n"
    report += '\n| Dataset / scene | BASE tracking ms | DP tracking ms | Reduction | BASE BA ms | DP BA ms | Reduction | DP queried-point reduction TR / BA |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for c in comparisons:
        report += f"| {c['dataset']} / {c['scene']} | {value(c,'BASE','tracking_wall_ms')} | {value(c,'DP','tracking_wall_ms')} | {c['tracking_wall_ms_reduction_pct']:+.2f}% | {value(c,'BASE','ba_wall_ms')} | {value(c,'DP','ba_wall_ms')} | {c['ba_wall_ms_reduction_pct']:+.2f}% | {c['DP_tracking_pruned_pct']:.2f}% / {c['DP_ba_pruned_pct']:.2f}% |\n"
    report += '\nPrimary timing omits frames 0–19 and includes preprocessing through pose finalization. Positive reduction means faster. Query reduction covers the RGB/SDF field-query path, not unchanged dense feature smoothness. Per-run and per-scene CSVs retain CUDA timing and SD.\n\n'
    report += 'The research targets are at most 15% per-scene mean ATE increase, tracking about 10% and BA about 30% wall-time reduction, with five percentage points of timing tolerance (lower bounds 5% / 25%). The CSV records each screen separately. They are not an accepted paper claim: both ATE conventions are visible, all scenes matter, reconstruction scoring is pending, and restored-reference results do not inherit older acceptance figures. Negative results are retained.\n\n'
    report += 'This automated step performs no new experiment selection, code modification, power-mode change or email delivery. It stops for author review after this series. A completion email additionally requires a freshly verified idle GPU and the agreed decision summary; no unattended mail transport is configured.\n'
    write_csv(output / 'per_run.csv', rows)
    write_csv(output / 'per_scene.csv', comparisons)
    (output / 'REPORT.md').write_text(report)
    (output / 'ANALYSIS_STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
