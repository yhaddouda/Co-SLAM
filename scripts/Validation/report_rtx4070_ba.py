"""Compare audited RTX BA dense-fast observations against the packaged Orin runs."""
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.Validation.audit_rtx4070_ba_results import audit_run
from scripts.Validation.preflight_rtx4070_tracking import differences

RTX = ROOT / 'results/ba_experiments/rtx4070_dense_fast_full'
META = ROOT / 'results/ba_experiments/rtx4070_metadata'
ORIN = ROOT / 'transfer/rtx4070/staging/coslam_experiments/results/ba_experiments/full'
OUT = ROOT / 'results/ba_experiments/rtx4070_comparison'
SCENES = ('office1', 'room1', 'office0')
VARIANTS = ('baseline', 'baseline_fast')
PHASES = ('ba_invocations', 'ba_frames')


def reduction(value, baseline):
    return 100 * (1 - value / baseline)


def csv_file(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    audit = json.loads((RTX / 'AUDIT.json').read_text())
    assert audit['completed_runs'] == 6 and audit['suite_complete']
    OUT.mkdir(exist_ok=False)
    data, records, config_differences, windows, missing_snapshots = {}, [], [], [], []
    for platform, root in (('RTX', RTX), ('Orin', ORIN)):
        for scene in SCENES:
            for variant in VARIANTS:
                directory = root / scene / f'{variant}_seed0'
                summary, cfg, frames, signature = audit_run(directory, require_snapshot=platform == 'RTX')
                data[platform, scene, variant] = (summary, cfg)
                if signature is None:
                    missing_snapshots.append(str(directory))
                measured = [r for r in frames if int(r['frame_id']) >= 20 and int(r['has_ba'])]
                for bucket in range(20):
                    selected = [r for r in measured if int(r['frame_id']) // 100 == bucket]
                    windows.append(dict(platform=platform, scene=scene, variant=variant,
                                        window_start_frame=bucket * 100, count=len(selected),
                                        ba_call_mean_ms=mean(float(r['ba_wall_ms']) for r in selected),
                                        ba_frame_mean_ms=mean(float(r['wall_ms']) for r in selected)))
    for scene in SCENES:
        for variant in VARIANTS:
            row = dict(scene=scene, variant=variant, seed=0, frames=2000, warmup_frames=20)
            rtx, rcfg = data['RTX', scene, variant]
            orin, ocfg = data['Orin', scene, variant]
            config_differences.append(dict(scene=scene, variant=variant, differences=differences(rcfg, ocfg)))
            for platform, summary in (('rtx', rtx), ('orin', orin)):
                baseline = data['RTX' if platform == 'rtx' else 'Orin', scene, 'baseline'][0]
                row[platform + '_ate_mm'] = 1000 * summary['ate_rmse_m']
                row[platform + '_relative_ate_mm'] = 1000 * summary['ate_relative_rmse_m']
                row[platform + '_ate_change_mm'] = 1000 * (summary['ate_rmse_m'] - baseline['ate_rmse_m'])
                row[platform + '_ate_change_pct'] = 100 * (summary['ate_rmse_m'] / baseline['ate_rmse_m'] - 1)
                for phase in PHASES:
                    for key, value in summary[phase].items():
                        row[f'{platform}_{phase}_{key}'] = value
                    row[f'{platform}_{phase}_reduction_pct'] = reduction(
                        summary[phase]['mean_ms'], baseline[phase]['mean_ms'])
            for phase in PHASES:
                row[f'{phase}_rtx_minus_orin_reduction_pp'] = (
                    row[f'rtx_{phase}_reduction_pct'] - row[f'orin_{phase}_reduction_pct'])
                row[f'{phase}_orin_over_rtx_time_ratio'] = orin[phase]['mean_ms'] / rtx[phase]['mean_ms']
            records.append(row)
    csv_file(OUT / 'comparison.csv', records)
    csv_file(OUT / 'runtime_windows.csv', windows)
    treatments = [r for r in records if r['variant'] == 'baseline_fast']
    aggregates = {}
    for platform in ('rtx', 'orin'):
        aggregates[f'{platform}_mean_ate_change_pct'] = mean(r[f'{platform}_ate_change_pct'] for r in treatments)
        aggregates[f'{platform}_worst_ate_change_pct'] = max(r[f'{platform}_ate_change_pct'] for r in treatments)
        for phase in PHASES:
            aggregates[f'{platform}_{phase}_mean_reduction_pct'] = mean(r[f'{platform}_{phase}_reduction_pct'] for r in treatments)
    for phase in PHASES:
        aggregates[f'{phase}_rtx_minus_orin_reduction_pp'] = (
            aggregates[f'rtx_{phase}_mean_reduction_pct'] - aggregates[f'orin_{phase}_mean_reduction_pct'])
    with (META / 'gpu_telemetry.csv').open() as stream:
        telemetry = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
    def number(row, column):
        return float(row[column].split()[0])
    active = [r for r in telemetry if number(r, 'utilization.gpu [%]') > 5]
    gpu = dict(total_samples=len(telemetry), active_samples=len(active),
               first_sample=telemetry[0]['timestamp'], last_sample=telemetry[-1]['timestamp'],
               active_definition='GPU utilization >5%; includes initialization, baseline tracking, and BA',
               active_clock_event_flags=sorted({r['clocks_event_reasons.active'] for r in active}))
    for column in ('utilization.gpu [%]', 'temperature.gpu', 'power.draw [W]', 'power.limit [W]',
                   'memory.used [MiB]', 'clocks.current.graphics [MHz]', 'clocks.current.memory [MHz]'):
        values = [number(r, column) for r in active]
        gpu[column] = dict(min=min(values), mean=mean(values), median=float(np.median(values)), max=max(values))
    result = dict(aggregates=aggregates, configurations=config_differences,
                  telemetry=gpu, preflight=json.loads((META / 'PREFLIGHT.json').read_text()),
                  independently_revalidated_runs=12,
                  orin_runs_without_source_snapshots=missing_snapshots,
                  isolated_mechanism='BA batch preloading plus final optimized-pose batch reuse; tracking baseline')
    (OUT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# RTX 4070 BA dense-fast versus Orin', '',
             'Completed six serial Co-SLAM runs: fresh BA baseline and BA `baseline_fast` on office1, room1, '
             'and office0. Every trajectory has 2,000 frames, seed 0, and a 20-frame timing warmup. '
             'Tracking stays baseline. This is the BA dense-fast mechanism, separate from the earlier '
             'tracking `baseline_fast` results.', '',
             'Dense-fast preloads the exact baseline ray batches and reuses the already updated optimized '
             'pose batch at finalization. It retains 43 depth samples per ray, the baseline ray counts, '
             '10 BA iterations, and the optimizer schedule. No bucket, lattice packing, Morton sorting, '
             'tracking optimization, or depth cache is enabled.', '',
             '| Platform | Mean BA-call reduction | Mean complete BA-frame reduction | Mean ATE change | Worst ATE change |',
             '|---|---:|---:|---:|---:|']
    for platform in ('rtx', 'orin'):
        lines.append(f"| {platform.upper()} | {aggregates[f'{platform}_ba_invocations_mean_reduction_pct']:.2f}% | "
                     f"{aggregates[f'{platform}_ba_frames_mean_reduction_pct']:.2f}% | "
                     f"{aggregates[f'{platform}_mean_ate_change_pct']:+.2f}% | {aggregates[f'{platform}_worst_ate_change_pct']:+.2f}% |")
    lines += ['', 'Percentages are equal-weight means of paired scene changes. Positive timing reduction '
              'means faster; positive ATE change means worse. ATE is final raw-trajectory rigid-aligned '
              'translation RMSE in float64, with no scale fitting. Relative-pose ATE is retained in CSV/JSON.', '',
              '## Per-scene paired results', '',
              '| Platform | Scene | Baseline ATE mm | Dense-fast ATE mm | ATE change | BA-call reduction | Complete BA-frame reduction |',
              '|---|---|---:|---:|---:|---:|---:|']
    for platform in ('rtx', 'orin'):
        for r in treatments:
            baseline = data['RTX' if platform == 'rtx' else 'Orin', r['scene'], 'baseline'][0]
            lines.append(f"| {platform.upper()} | {r['scene']} | {1000*baseline['ate_rmse_m']:.3f} | "
                         f"{r[f'{platform}_ate_mm']:.3f} | {r[f'{platform}_ate_change_pct']:+.2f}% | "
                         f"{r[f'{platform}_ba_invocations_reduction_pct']:.2f}% | {r[f'{platform}_ba_frames_reduction_pct']:.2f}% |")
    lines += ['', '## Synchronized timing statistics', '',
              'BA calls include setup, ray selection/transfers, the complete optimization loop, and pose '
              'finalization. Complete BA frames additionally include loading, baseline tracking, and '
              'keyframe insertion. All runs retain 396 BA-call samples and 396 BA-frame samples '
              '(frames 20, 25, …, 1995), plus 1,584 pure tracking-frame samples. CUDA is synchronized at '
              'both timing boundaries. Evaluation, meshes, checkpoints, and profilers are disabled.', '',
              '| Platform | Scene | Variant | BA mean ms | Median | p95 | SD | BA-frame mean ms | Median | p95 | SD |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for platform in ('rtx', 'orin'):
        for r in records:
            values = ' | '.join(f'{r[f"{platform}_{phase}_{metric}"]:.3f}'
                                for phase in PHASES for metric in ('mean_ms', 'median_ms', 'p95_ms', 'std_ms'))
            lines.append(f"| {platform.upper()} | {r['scene']} | {r['variant']} | {values} |")
    lines += ['', '## Platform comparison and interpretation', '',
              '| Scene | RTX minus Orin BA saving, pp | RTX minus Orin frame saving, pp | Baseline Orin/RTX BA time | Dense-fast Orin/RTX BA time |',
              '|---|---:|---:|---:|---:|']
    for r in treatments:
        baseline = next(b for b in records if b['scene'] == r['scene'] and b['variant'] == 'baseline')
        lines.append(f"| {r['scene']} | {r['ba_invocations_rtx_minus_orin_reduction_pp']:+.2f} | "
                     f"{r['ba_frames_rtx_minus_orin_reduction_pp']:+.2f} | "
                     f"{baseline['ba_invocations_orin_over_rtx_time_ratio']:.3f}x | "
                     f"{r['ba_invocations_orin_over_rtx_time_ratio']:.3f}x |")
    lines += ['', 'These are whole-platform observations. RTX uses a Ryzen 5 9600X, native x86 builds, '
              'PyTorch CUDA 11.8, and driver 580.173.02; the packaged Orin uses Arm CPUs and PyTorch '
              'CUDA 12.4. CPU dispatch, transfers, GPU kernels, clocks, and software builds all differ '
              'together. The relative savings and absolute ratios cannot isolate Ampere-versus-Ada '
              'architecture or assign the measured gain to a particular component without profiling '
              'and controlled follow-up measurements. One seed does not establish accuracy equivalence.', '',
              '## Source, configuration, and provenance', '',
              'The BA dense-fast helper and harness match all three Orin dense-fast snapshots exactly. '
              'Six of their eight recorded source files match RTX. `coslam.py` retains local guards and '
              'instance configuration lookups and adds later optional tracking paths; `scene_rep.py` adds '
              'an optional depth cache. Those tracking/cache paths are off. The BA function AST matches '
              'the Orin dense-fast source after normalizing the retained configuration references. '
              'The preloading RNG/batch and pose-commit semantic tests passed; all 10 packaged BA CPU '
              'tests and the two-condition CUDA smoke suite passed before measurement.', '',
              'The original Orin office1 baseline lacks a per-run source snapshot/hash manifest. '
              'Room1/office0 baseline snapshots have historical source-list/docstring differences in '
              'the benchmark/ranking scripts. Every available snapshot hash is checked. All 12 RTX/Orin '
              'trajectories, ATEs, raw timing statistics, and isolation configurations were independently '
              'revalidated; missing historical provenance is recorded rather than assumed identical.', '',
              'Resolved configuration differences include local dataset/output paths, inactive inner '
              'timer/profiler fields, an explicit baseline tracking key on RTX, and the collect-samples '
              'flag missing from the earliest Orin baseline. The latter affects lattice diagnostics '
              'and is inactive for both dense conditions. Full differences and source patches are in '
              '`../rtx4070_metadata/PREFLIGHT.json` and `comparison.json`.', '',
              'Datasets use `../datasets/data/Replica/{office1,room1,office0}`. All 2,000 RGB/depth '
              'files and poses are present; the loaded ground-truth matrices match Orin exactly. '
              'Native tiny-cuda-nn source commit and Python wrapper hashes match the transferred '
              'evidence; native GPU libraries were retained. `TORCH_CUDA_ARCH_LIST=8.9` and the '
              'previously warmed RTX extension cache were used. Power/clock settings were not changed.', '',
              'Run order was office1/room1/office0 baseline, then office1/room1/office0 dense-fast. '
              'This follows the original Orin baseline-then-fast grouping while omitting its other '
              'bucket/lattice conditions. Prior tracking results and Orin files were preserved. '
              'Plotting uses the previously staged isolated `packaging` dependency after timing; '
              'the native conda environment was not changed.', '',
              '## Desktop telemetry', '',
              f"One-second telemetry spans {gpu['first_sample']}–{gpu['last_sample']} in Europe/Paris time, "
              f"with {gpu['total_samples']} samples and {gpu['active_samples']} above 5% utilization. "
              'Active statistics include initialization, baseline tracking, and BA.', '',
              '| Quantity | Minimum | Mean | Median | Maximum |', '|---|---:|---:|---:|---:|']
    for column, values in gpu.items():
        if isinstance(values, dict):
            lines.append(f"| {column} | {values['min']:.2f} | {values['mean']:.2f} | {values['median']:.2f} | {values['max']:.2f} |")
    lines += ['', f"Active clock-event flags: {gpu['active_clock_event_flags']}.", '',
              '## Artifacts', '',
              '- [RTX suite report](../rtx4070_dense_fast_full/REPORT.md) and [independent audit](../rtx4070_dense_fast_full/AUDIT.json).',
              '- [Full precision comparison CSV](comparison.csv), [comparison JSON](comparison.json), and [100-frame timing windows](runtime_windows.csv).',
              '- [Environment](../rtx4070_metadata/environment.json), [hardware](../rtx4070_metadata/hardware.json), '
              '[preflight](../rtx4070_metadata/PREFLIGHT.json), [session](../rtx4070_metadata/SESSION.json), '
              'and [raw desktop telemetry](../rtx4070_metadata/gpu_telemetry.csv).',
              '- Per-run trajectories, raw frame times, configs, source snapshots, and hashes are in '
              '`../rtx4070_dense_fast_full/<scene>/<variant>_seed0/`.',
              '- Outer logs: `../../execution_experiments/logs/rtx4070/ba_dense_fast/`.', '',
              '![Paired BA time reductions](ba_time_reductions.png)', '',
              '![BA runtime across the trajectory](ba_runtime_by_frame.png)', '']
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/coslam-rtx-ba-plots')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for ax, phase, title in zip(axes, PHASES, ('Complete BA call', 'Complete BA frame')):
        x = np.arange(3)
        ax.bar(x-.18, [r[f'rtx_{phase}_reduction_pct'] for r in treatments], .36, label='RTX 4070', color='#2166ac')
        ax.bar(x+.18, [r[f'orin_{phase}_reduction_pct'] for r in treatments], .36, label='Orin', color='#d95f02')
        ax.set_xticks(x, SCENES)
        ax.set_title(title)
        ax.set_ylabel('Mean time reduction (%)')
        ax.grid(axis='y', alpha=.2)
        ax.legend()
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'ba_time_reductions.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, scene in zip(axes, SCENES):
        for platform, color in (('RTX', '#2166ac'), ('Orin', '#d95f02')):
            for variant, style in (('baseline', '--'), ('baseline_fast', '-')):
                selected = [w for w in windows if w['platform'] == platform and w['scene'] == scene and w['variant'] == variant]
                ax.plot([w['window_start_frame']+50 for w in selected], [w['ba_call_mean_ms'] for w in selected],
                        color=color, linestyle=style, label=platform + ' ' + variant)
        ax.set_title(scene)
        ax.set_xlabel('Frame index (100-frame windows)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Complete BA-call time (ms)')
    axes[-1].legend(fontsize=7)
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / f'ba_runtime_by_frame.{suffix}', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(aggregates=aggregates, paired_scenes=treatments, report=str(OUT/'REPORT.md')), indent=2))


if __name__ == '__main__':
    main()
