"""Generate final CPU audits and reports after the serial GPU queue completes."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/execution_experiments'


def wait_for(marker):
    while not marker.exists():
        print('WAIT ' + str(marker.relative_to(ROOT)), flush=True)
        time.sleep(30)


def command(script, *args):
    subprocess.run([sys.executable, str(ROOT / 'scripts/Validation' / script),
                    *map(str, args)], cwd=ROOT, check=True)


def aggregate(root):
    import csv
    with (root / 'comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for variant in sorted({r['variant'] for r in rows} - {'baseline'}):
        selected = [r for r in rows if r['variant'] == variant]
        result[variant] = dict(pairs=len(selected),
            mean_ate_change_pct=mean(float(r['ate_change_pct']) for r in selected),
            worst_ate_change_pct=max(float(r['ate_change_pct']) for r in selected))
        for phase in ('tracking_invocations', 'tracking_frames', 'mapping_invocations', 'mapping_frames'):
            key = phase + '_reduction_pct'
            result[variant][key] = mean(float(r[key]) for r in selected) if selected[0][key] else None
    return result


def main():
    co = RESULTS / 'coslam/full'
    ec = RESULTS / 'ec_slam/full'
    repeat = RESULTS / 'coslam/repeat_full'
    ec_repeat = RESULTS / 'ec_slam/repeat_full'
    validation = RESULTS / 'gpu_checks/final_validation'
    wait_for(ec / 'SUITE_COMPLETE')
    # The earlier telemetry process ends with the EC suite. Cover the separate
    # baseline repeat without any GPU queries or additional heavy CPU workload.
    repeat_telemetry = RESULTS / 'hardware_telemetry_repeat.csv'
    with (RESULTS / 'hardware_telemetry_repeat.log').open('w') as stream:
        monitor = subprocess.Popen([sys.executable,
            str(ROOT / 'scripts/Validation/monitor_execution_hardware.py'),
            '--output', str(repeat_telemetry), '--until', str(repeat / 'SUITE_COMPLETE')],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        wait_for(validation / 'VALIDATION_COMPLETE')
        assert monitor.wait(timeout=10) == 0
    roots = [co, ec, repeat, ec_repeat]
    command('audit_execution_results.py', '--roots', *roots, '--require-complete')
    for root in roots:
        command('summarize_execution_experiments.py', '--root', root, '--plot')
    command('summarize_execution_diagnostics.py', '--roots', co, ec,
        '--telemetry', RESULTS / 'hardware_telemetry.csv', repeat_telemetry,
        '--coslam-full', co, '--coslam-repeat', repeat, '--ec-full', ec,
        '--ec-repeat', ec_repeat, '--output', RESULTS / 'diagnostics')
    original = json.loads((ROOT / 'experiments/ec_slam_execution/UPSTREAM_SNAPSHOT.json').read_text())
    changed = [name for name, sha in original['sha256'].items()
        if hashlib.sha256((Path(original['source']) / name).read_bytes()).hexdigest() != sha]
    (RESULTS / 'original_ec_source_check.json').write_text(json.dumps(dict(
        checked_files=len(original['sha256']), changed_files=changed), indent=2) + '\n')
    assert not changed, ('Original EC sources changed since the snapshot', changed)
    aggregates = dict(coslam=aggregate(co), ec_slam=aggregate(ec))
    (RESULTS / 'full_aggregates.json').write_text(json.dumps(aggregates, indent=2) + '\n')
    lines = ['# Full execution experiments on Orin', '',
        'Complete 2,000-frame Replica trajectories on office1, room1 and office0, paired '
        'seed 0. Co-SLAM changes tracking only; its BA remains dense baseline and is not '
        'timed. EC-SLAM tests tracking-only, mapping-only and combined changes. '
        'All experimental flags remain optional and default to baseline.', '',
        '| System | Variant | Tracking call reduction | Tracking frame reduction | Mapping call reduction | Mapping frame reduction | Mean ATE change | Worst ATE change |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):
        return '—' if value is None else f'{value:+.2f}%'
    for system, variants in aggregates.items():
        for variant, values in variants.items():
            keys = ('tracking_invocations_reduction_pct', 'tracking_frames_reduction_pct',
                    'mapping_invocations_reduction_pct', 'mapping_frames_reduction_pct',
                    'mean_ate_change_pct', 'worst_ate_change_pct')
            lines.append('| ' + ' | '.join([system, variant, *[fmt(values[key]) for key in keys]]) + ' |')
    lines += ['', 'Positive time reduction is faster; positive ATE change is worse. '
        'Percentages average paired scene changes. Absolute values, medians, p95s, counts, '
        'relative ATE and individual trajectories are retained in the detailed reports.', '',
        '- [Co-SLAM full results](coslam/full/REPORT.md) and [tradeoff plot](coslam/full/tradeoff.png).',
        '- [EC-SLAM full results](ec_slam/full/REPORT.md) and [tradeoff plot](ec_slam/full/tradeoff.png).',
        '- [Same-source baseline repeat and telemetry](diagnostics/REPORT.md).',
        '- [Separate profiler evidence and CUDA checks](gpu_checks/TRACE_REPORT.md); '
        'final pose-specific gradient checks are in `gpu_checks/final_validation`.',
        '- [Implementation, protocol, architecture hypotheses and reproduction](../../docs/execution_experiments.md).', '',
        'All four full roots passed independent trajectory/ATE/count/config/source audits '
        '(`AUDIT.json`). The adjacent original EC source files match the recorded snapshot. '
        'Five short traces and 42 prefix screening runs are separate from these full-run '
        'timing comparisons. Final frozen CUDA checks test pose gradients separately from '
        'map gradients; CPU semantic/loss tests passed 25 checks.', '',
        'A single paired seed and one baseline repeat per system do not establish statistical zero '
        'degradation. Depth and reconstruction quality were not measured in this experiment. '
        'The RTX 4070 is unavailable, so architectural extrapolations remain hypotheses '
        'and no RTX gain is reported. These are execution optimizations; two implementations '
        'on one platform alone do not establish a new estimator or universal hash-grid SLAM speedup.']
    (RESULTS / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (RESULTS / 'REPORTS_COMPLETE').write_text('Full timing suites, repeat, final CUDA checks and CPU result audits complete.\n')
    print('REPORTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
