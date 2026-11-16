"""Compare audited RTX tracking results with the packaged Co-SLAM Orin runs."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.Validation.execution_benchmark_common import rigid_ate, statistics
from scripts.Validation.preflight_rtx4070_tracking import differences

META = ROOT / 'results/execution_experiments/rtx4070_metadata'
STAGE = ROOT / 'transfer/rtx4070/staging/coslam_experiments'
OUTPUT = ROOT / 'results/execution_experiments/rtx4070_comparison'
SCENES = ('office1', 'room1', 'office0')
VARIANTS = ('baseline', 'baseline_fast', 'cached_fast')
PHASES = ('tracking_invocations', 'tracking_frames')


def saved(value, baseline):
    return 100 * (1 - value / baseline)


def load(root, scene, variant):
    directory = root / scene / f'{variant}_seed0'
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['frames'] == 2000 and summary['warmup_frames'] == 20
    assert not summary['diagnostic'] and summary['finite_parameters_and_poses']
    assert summary['mapping_variant'] == 'baseline' and summary['seed'] == 0
    cfg = json.loads((directory / 'config.json').read_text())
    assert cfg['ba_experiment']['variant'] == 'baseline' and not cfg['ba_bucket']['enabled']
    assert cfg['tracking_experiment']['variant'] == variant
    assert all(not cfg[n]['enabled'] for n in
               ('tracking_bucket', 'tracking_depth_spatial', 'tracking_naive_pruning'))
    assert not cfg['grid']['morton_sort'] and cfg['grid']['hash'] == 'CoherentPrime'
    assert cfg['grid']['hash_size'] == 19
    assert not cfg['profiling']['enabled'] and cfg['timing']['disable_eval']
    trajectories = np.load(directory / 'trajectories.npz')
    for name, key in (('estimated', 'ate_rmse_m'), ('relative', 'ate_relative_rmse_m')):
        assert trajectories[name].shape == (2000, 4, 4) and np.isfinite(trajectories[name]).all()
        ate = rigid_ate(trajectories['ground_truth'], trajectories[name]) / cfg['data']['sc_factor']
        assert abs(ate - summary[key]) < 1e-12
    with (directory / 'frames.csv').open() as stream:
        frames = list(csv.DictReader(stream))
    assert [int(f['frame_id']) for f in frames] == list(range(1, 2000))
    measured = [f for f in frames if int(f['frame_id']) >= 20]
    for phase, values in (
        ('tracking_invocations', [float(f['tracking_wall_ms']) for f in measured]),
        ('tracking_frames', [float(f['wall_ms']) for f in measured if not int(f['has_mapping'])])):
        assert len(values) == (1980 if phase == 'tracking_invocations' else 1584)
        assert np.isfinite(values).all() and min(values) > 0
        recomputed = statistics(values)
        for metric in recomputed:
            assert abs(recomputed[metric] - summary[phase][metric]) < 1e-9
    for name, value in json.loads((directory / 'source_hashes.json').read_text()).items():
        assert hashlib.sha256((directory / 'source' / name).read_bytes()).hexdigest() == value
    return summary, cfg


def write_csv(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    full = ROOT / 'results/execution_experiments/coslam/rtx4070_full'
    repeat = ROOT / 'results/execution_experiments/coslam/rtx4070_repeat_full'
    orin = ROOT / 'results/execution_experiments/coslam/full'
    orin_repeat = ROOT / 'results/execution_experiments/coslam/repeat_full'
    session = json.loads((META / 'SESSION.json').read_text())
    assert session['status'] == 'complete_and_audited'
    for root, count in ((full, 9), (repeat, 1)):
        audit = json.loads((root / 'AUDIT.json').read_text())
        assert audit['suite_complete'] and audit['completed_runs'] == count
        assert not audit['differing_source_snapshots']
    OUTPUT.mkdir(exist_ok=False)
    rows, config_changes = [], []
    samples = {}
    source_signatures = {}
    for scene in SCENES:
        for variant in VARIANTS:
            samples['RTX', scene, variant] = load(full, scene, variant)
            samples['Orin', scene, variant] = load(orin, scene, variant)
            for platform, root in (('RTX', full), ('Orin', orin)):
                signature = json.loads((root / scene / f'{variant}_seed0/source_hashes.json').read_text())
                source_signatures.setdefault(platform, signature)
                assert signature == source_signatures[platform]
            rtx, rtx_cfg = samples['RTX', scene, variant]
            orin_sample, orin_cfg = samples['Orin', scene, variant]
            config_changes.append(dict(scene=scene, variant=variant,
                                       differences=differences(rtx_cfg, orin_cfg)))
            expected = json.loads((META / f'{scene}_expected_config.json').read_text())
            expected['data']['output'] = rtx_cfg['data']['output']
            expected['tracking_experiment']['variant'] = variant
            assert rtx_cfg == expected
            row = dict(scene=scene, variant=variant, seed=0, frames=2000, timing_warmup_frames=20)
            for platform, data in (('rtx', rtx), ('orin', orin_sample)):
                baseline = samples['RTX' if platform == 'rtx' else 'Orin', scene, 'baseline'][0]
                row[platform + '_ate_mm'] = 1000 * data['ate_rmse_m']
                row[platform + '_relative_ate_mm'] = 1000 * data['ate_relative_rmse_m']
                row[platform + '_ate_change_mm'] = 1000 * (data['ate_rmse_m'] - baseline['ate_rmse_m'])
                row[platform + '_ate_change_pct'] = 100 * (data['ate_rmse_m'] / baseline['ate_rmse_m'] - 1)
                for phase in PHASES:
                    for metric, value in data[phase].items():
                        row[f'{platform}_{phase}_{metric}'] = value
                    row[f'{platform}_{phase}_reduction_pct'] = saved(
                        data[phase]['mean_ms'], baseline[phase]['mean_ms'])
            for phase in PHASES:
                row[phase + '_reduction_rtx_minus_orin_pp'] = (
                    row[f'rtx_{phase}_reduction_pct'] - row[f'orin_{phase}_reduction_pct'])
                row[phase + '_orin_over_rtx_mean_ratio'] = (
                    orin_sample[phase]['mean_ms'] / rtx[phase]['mean_ms'])
            rows.append(row)
    write_csv(OUTPUT / 'comparison.csv', rows)
    aggregates = []
    for variant in VARIANTS[1:]:
        selected = [r for r in rows if r['variant'] == variant]
        record = dict(variant=variant, paired_scenes=3)
        for platform in ('rtx', 'orin'):
            record[platform + '_mean_ate_change_pct'] = mean(r[platform + '_ate_change_pct'] for r in selected)
            record[platform + '_worst_ate_change_pct'] = max(r[platform + '_ate_change_pct'] for r in selected)
            for phase in PHASES:
                record[f'{platform}_{phase}_mean_reduction_pct'] = mean(
                    r[f'{platform}_{phase}_reduction_pct'] for r in selected)
        for phase in PHASES:
            record[f'{phase}_reduction_rtx_minus_orin_pp'] = (
                record[f'rtx_{phase}_mean_reduction_pct'] - record[f'orin_{phase}_mean_reduction_pct'])
        aggregates.append(record)
    repeat_rows = []
    for platform, root in (('RTX', repeat), ('Orin', orin_repeat)):
        value, repeat_config = load(root, 'office1', 'baseline')
        signature = json.loads((root / 'office1/baseline_seed0/source_hashes.json').read_text())
        assert signature == source_signatures[platform], 'Baseline repeat source changed'
        baseline = samples[platform, 'office1', 'baseline'][0]
        first_config = json.loads(json.dumps(samples[platform, 'office1', 'baseline'][1]))
        first_config['data'].pop('output')
        repeat_config['data'].pop('output')
        assert first_config == repeat_config, 'Baseline repeat configuration changed'
        item = dict(platform=platform, scene='office1', seed=0,
                    first_ate_mm=1000 * baseline['ate_rmse_m'], repeat_ate_mm=1000 * value['ate_rmse_m'],
                    ate_change_mm=1000 * (value['ate_rmse_m'] - baseline['ate_rmse_m']),
                    ate_change_pct=100 * (value['ate_rmse_m'] / baseline['ate_rmse_m'] - 1))
        for phase in PHASES:
            item[f'first_{phase}_mean_ms'] = baseline[phase]['mean_ms']
            item[f'repeat_{phase}_mean_ms'] = value[phase]['mean_ms']
            item[f'{phase}_reduction_pct'] = saved(value[phase]['mean_ms'], baseline[phase]['mean_ms'])
            for metric, measurement in value[phase].items():
                item[f'repeat_{phase}_{metric}'] = measurement
        repeat_rows.append(item)
    write_csv(OUTPUT / 'baseline_repeats.csv', repeat_rows)
    with (META / 'gpu_telemetry.csv').open() as stream:
        telemetry = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
    def numeric(row, key):
        return float(row[key].split()[0])
    active = [r for r in telemetry if numeric(r, 'utilization.gpu [%]') > 5]
    telemetry_stats = dict(total_samples=len(telemetry), active_samples=len(active),
                           active_definition='GPU utilization > 5%; includes startup/mapping and tracking',
                           first_sample=telemetry[0]['timestamp'], last_sample=telemetry[-1]['timestamp'])
    for column in ('utilization.gpu [%]', 'utilization.memory [%]', 'memory.used [MiB]',
                   'temperature.gpu', 'power.draw [W]', 'power.limit [W]',
                   'clocks.current.graphics [MHz]', 'clocks.current.sm [MHz]', 'clocks.current.memory [MHz]'):
        values = [numeric(r, column) for r in active]
        telemetry_stats[column] = dict(min=min(values), mean=mean(values),
                                        median=float(np.median(values)), max=max(values))
    telemetry_stats['active_clock_event_flags'] = sorted({r['clocks_event_reasons.active'] for r in active})
    rtx_environment = json.loads((META / 'environment.json').read_text())
    orin_environment = json.loads((STAGE / 'results/execution_experiments/gpu_checks/environments/coslam.json').read_text())
    preflight = json.loads((META / 'PREFLIGHT.json').read_text())
    result = dict(aggregates=aggregates, baseline_repeats=repeat_rows, telemetry=telemetry_stats,
                  environment_differences=differences(rtx_environment['packages'], orin_environment['packages']),
                  resolved_config_differences=config_changes, preflight=preflight,
                  revalidated_results='All 20 RTX and packaged Co-SLAM Orin runs: trajectories, ATE, '
                  'sample counts, raw timing statistics, source snapshots and isolation settings passed.')
    (OUTPUT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# RTX 4070 Co-SLAM tracking reproduction', '',
             'Completed and audited: nine matched runs plus a separate office1 baseline repeat. '
             'Every run uses seed 0, all 2,000 frames, a 20-frame timing warmup, and dense baseline BA. '
             'No EC-SLAM experiment or BA variant suite was run.', '',
             'The name `baseline_fast` is used by two separate experiments. All `baseline_fast` rows '
             'in this report refer to **tracking**, selected by `tracking_experiment.variant`. '
             'The earlier **BA dense-fast** result is selected by `ba_experiment.variant` and has '
             'its own baseline, timing boundary, and report.', '',
             '| Experiment | Mechanism | Orin mean call reduction | Orin mean frame reduction | RTX mean call reduction | RTX mean frame reduction |',
             '|---|---|---:|---:|---:|---:|',
             '| BA `baseline_fast` (dense-fast) | Preload the same BA ray batches; reuse the final optimized pose batch | 47.73% | 34.30% of complete BA-frame time | Unmeasured | Unmeasured |',
             '| Tracking `baseline_fast` (this report) | Reuse the current pose matrix; allocate pose constants on device; defer best-loss selection | 6.14% | 6.04% of pure tracking-frame time | 8.94% | 8.41% of pure tracking-frame time |', '',
             'The 49.44% / 45.62% / 48.13% BA-call reductions for office1 / room1 / office0 '
             'remain valid Orin observations in [the packaged BA protocol](../../../docs/replica_ba_experiments.md). '
             'This RTX tracking suite deliberately sets `ba_experiment.variant=baseline`; it does not '
             'measure the BA dense-fast mechanism. Comparing its 8.94% tracking gain with the earlier '
             '47.73% BA gain would mix stages and transformations.', '',
             'For the matched tracking baseline, Orin/RTX call-time ratios are 6.116 (office1), '
             '6.115 (room1), and 5.981 (office0). These are observed whole-platform ratios. CPU, '
             'memory, clocks, drivers, and CUDA/dependency builds differ together with the GPU; '
             'this comparison cannot isolate Ampere-versus-Ada architecture. A matched RTX BA '
             'baseline/dense-fast suite would be required to compare the earlier BA mechanism across platforms.', '',
             'Positive timing reduction means faster; positive ATE change means worse. '
             'ATE is the final raw trajectory translation RMSE after float64 Horn rigid alignment, '
             'without scale fitting. Relative trajectories are retained in the JSON/CSV artifacts. '
             'Aggregate percentages give equal weight to each paired scene.', '',
             '| Variant | RTX call saved | Orin call saved | Difference pp | RTX frame saved | Orin frame saved | Difference pp | RTX mean ATE change | RTX worst ATE change |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in aggregates:
        lines.append(f"| {a['variant']} | {a['rtx_tracking_invocations_mean_reduction_pct']:+.2f}% | "
                     f"{a['orin_tracking_invocations_mean_reduction_pct']:+.2f}% | "
                     f"{a['tracking_invocations_reduction_rtx_minus_orin_pp']:+.2f} | "
                     f"{a['rtx_tracking_frames_mean_reduction_pct']:+.2f}% | "
                     f"{a['orin_tracking_frames_mean_reduction_pct']:+.2f}% | "
                     f"{a['tracking_frames_reduction_rtx_minus_orin_pp']:+.2f} | "
                     f"{a['rtx_mean_ate_change_pct']:+.2f}% | {a['rtx_worst_ate_change_pct']:+.2f}% |")
    lines += ['', '## Per-scene paired changes', '',
              '| Scene | Variant | RTX ATE mm | RTX ATE change | RTX call saved | RTX frame saved | Orin ATE mm | Orin ATE change | Orin call saved | Orin frame saved |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['scene']} | {r['variant']} | {r['rtx_ate_mm']:.3f} | {r['rtx_ate_change_pct']:+.2f}% | "
                     f"{r['rtx_tracking_invocations_reduction_pct']:+.2f}% | {r['rtx_tracking_frames_reduction_pct']:+.2f}% | "
                     f"{r['orin_ate_mm']:.3f} | {r['orin_ate_change_pct']:+.2f}% | "
                     f"{r['orin_tracking_invocations_reduction_pct']:+.2f}% | {r['orin_tracking_frames_reduction_pct']:+.2f}% |")
    lines += ['', '## Synchronized timing statistics', '',
              'Complete tracking calls include ray selection/transfer, the pose loop, best-pose selection, '
              'and final storage. Pure tracking frames include loading and all common frame work, '
              'and exclude frames running BA. Both use CUDA synchronization at timer boundaries. '
              'All runs have 1,980 call samples and 1,584 pure-frame samples (frames 20–1999). '
              'Evaluation, meshes, checkpoints, active profilers, experimental samplers, BA buckets, and Morton sorting '
              'are disabled. Phase/frame seeds and rotated run order match Orin.', '',
              '| Platform | Scene | Variant | Call mean ms | Median | p95 | SD | Pure frame mean ms | Median | p95 | SD |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for platform in ('rtx', 'orin'):
        for r in rows:
            values = ' | '.join(f'{r[f"{platform}_{phase}_{metric}"]:.3f}'
                                for phase in PHASES for metric in ('mean_ms', 'median_ms', 'p95_ms', 'std_ms'))
            lines.append(f"| {platform.upper()} | {r['scene']} | {r['variant']} | {values} |")
    lines += ['', '## Separate office1 baseline repeats', '',
              '| Platform | First ATE mm | Repeat ATE mm | ATE change mm | ATE change | First call ms | Repeat call ms | Call saved | First frame ms | Repeat frame ms | Frame saved |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in repeat_rows:
        lines.append(f"| {r['platform']} | {r['first_ate_mm']:.3f} | {r['repeat_ate_mm']:.3f} | "
                     f"{r['ate_change_mm']:+.3f} | {r['ate_change_pct']:+.2f}% | "
                     f"{r['first_tracking_invocations_mean_ms']:.3f} | {r['repeat_tracking_invocations_mean_ms']:.3f} | "
                     f"{r['tracking_invocations_reduction_pct']:+.2f}% | {r['first_tracking_frames_mean_ms']:.3f} | "
                     f"{r['repeat_tracking_frames_mean_ms']:.3f} | {r['tracking_frames_reduction_pct']:+.2f}% |")
    lines += ['', 'The repeat is a separate observation and is never substituted for the primary baseline. '
              'A single matched seed and one baseline repeat cannot establish zero expected quality degradation '
              'or isolate GPU architecture from CPU, memory, driver, and dependency differences.', '',
              '## Source, configuration, and datasets', '',
              'Archive SHA-256 `96a528046d9899563319f53d838edb5c66e4d4828ac38813faf362b28e3657f0` '
              'passed; all 2,297 manifest files verified after staging extraction. '
              'The full checkout comparison and application record are under `../rtx4070_metadata/`. '
              'Original local edits and Git diff are backed up in `../../../transfer/rtx4070/pre_apply_backup/`.', '',
              'Eight of the nine per-run source hashes match Orin exactly. `coslam.py` preserves local '
              'dataset/BA pool guards, `self.config` lookups, max-frame validation, and the input-folder override. '
              'The BA AST matches the packaged source after normalizing its three retained configuration lookups; '
              'the harness sets the same global and instance config. The packaged scene model and experiment '
              'helpers are used. The local ScanNet checks and native Morton default of 8.9 remain; Replica '
              'loading matches Orin. The extended runtime snapshots cover configuration, loaders, encoders, '
              'CUDA sources, and scripts, and stayed unchanged throughout the suite.', '',
              'Local Replica defaults are retained; optional baseline experiment keys were added. '
              'All active experimental configuration matches Orin after harness overrides. The remaining '
              'differences are dataset/output paths and inactive profiling/timing output, header, and inner '
              'warmup settings. Inner timing is off; the measured warmup is 20 frames. Exact differences '
              'for every condition are in `comparison.json`.', '',
              'Dataset roots are `../datasets/data/Replica/{office1,room1,office0}`. All 2,000 RGB/depth '
              'filenames, sizes, and poses were checked, three sample pairs decoded per scene, and all '
              'loaded ground-truth matrices match Orin exactly. The package does not include Orin image '
              'hashes, so RGB/depth byte identity across machines cannot be verified from this handoff.', '',
              'Five CPU tracking semantic tests and the Horn ATE agreement test passed. The packaged '
              'BA snapshot test skips because that older BA source was not installed; the separate '
              'preflight performs the normalized BA AST comparison against the staged Orin source. '
              'The 20-frame three-variant GPU smoke suite passed with finite-loss diagnostics and is '
              'excluded from all reported timings. Native PyTorch3D transforms and tiny-cuda-nn '
              'forward/backward operations passed. Independent final audits recompute trajectories, '
              'ATE, raw timing statistics, counts, isolation settings, and source hashes.', '',
              '## Hardware and environment', '',
              'RTX platform: AMD Ryzen 5 9600X (6 cores / 12 threads), RTX 4070 compute capability 8.9 '
              '(46 SMs, CUDA-addressable memory about 11,864 MiB), NVIDIA driver 580.173.02, 200 W GPU '
              'power limit. Power and clock settings were not changed. An idle inhibitor held through the '
              'persistent supervised run; desktop AC idle-suspend timeout was 0 (disabled). '
              'The Xorg/GNOME display processes are recorded in the before/after NVIDIA snapshots.', '',
              'Both use Python 3.10.18, PyTorch 2.3.0, PyTorch3D 0.7.8, and tiny-cuda-nn 2.0. '
              'RTX PyTorch uses CUDA 11.8; Orin PyTorch uses CUDA 12.4. The local tiny-cuda-nn source '
              'commit `4bf7700a9c4bb111c07c709e91cbca7d6c2a5ca0` matches the manifest; installed Python '
              'wrapper hashes match Orin. The RTX native `_89_C` binding and Orin Arm `_87_C` binding '
              'are separately hashed. No Arm environment or native binary was installed. '
              '`TORCH_CUDA_ARCH_LIST=8.9` was set for all GPU experiment commands and a fresh local '
              'CUDA extension cache was compiled before measurement.', '',
              'Postprocessing initially found that `coslam` lacks the pure-Python `packaging` dependency '
              'needed by Matplotlib. Plotting used an isolated copy of installed base-environment '
              '`packaging` 24.2 through `PYTHONPATH`, only after the measured runs. The conda environment '
              'and runtime source stayed unchanged. `../rtx4070_metadata/report_dependencies/PROVENANCE.json` '
              'records that copy; the original failure and recovered logs are retained.', '',
              '| Package/runtime | RTX | Orin |', '|---|---|---|']
    for d in result['environment_differences']:
        lines.append(f"| {d['path'].lstrip('.')} | {d['rtx']} | {d['orin']} |")
    lines += ['', '## Desktop telemetry', '',
              f"One-second telemetry covers {telemetry_stats['total_samples']} samples from "
              f"{telemetry_stats['first_sample']} to {telemetry_stats['last_sample']} (host Europe/Paris time). "
              f"There are {len(active)} samples with GPU utilization above 5%; these include startup, tracking, "
              'and dense BA. Statistics below use those active samples, not isolated tracking kernels.', '',
              '| Quantity | Minimum | Mean | Median | Maximum |', '|---|---:|---:|---:|---:|']
    for column, values in telemetry_stats.items():
        if isinstance(values, dict):
            lines.append(f"| {column} | {values['min']:.2f} | {values['mean']:.2f} | "
                         f"{values['median']:.2f} | {values['max']:.2f} |")
    lines += ['', f"Active clock-event flag values: {telemetry_stats['active_clock_event_flags']}. "
              'Raw timestamps, utilization, clocks, temperature, memory, power, and clock-event masks '
              'are retained in `../rtx4070_metadata/gpu_telemetry.csv`.', '',
              '## Artifacts', '',
              '- [RTX full report](../coslam/rtx4070_full/REPORT.md) and [independent audit](../coslam/rtx4070_full/AUDIT.json).',
              '- [RTX repeat report](../coslam/rtx4070_repeat_full/REPORT.md) and [independent audit](../coslam/rtx4070_repeat_full/AUDIT.json).',
              '- [Full precision comparison CSV](comparison.csv), [repeat CSV](baseline_repeats.csv), and [comparison JSON](comparison.json).',
              '- [Session record](../rtx4070_metadata/SESSION.json), [preflight](../rtx4070_metadata/PREFLIGHT.json), '
              '[environment](../rtx4070_metadata/environment.json), and [hardware](../rtx4070_metadata/hardware.json).',
              '- Outer logs are in `../logs/rtx4070/`; per-condition logs and raw frames/trajectories/config/source snapshots '
              'are in each RTX suite root. Orin reference roots and all earlier local results are preserved.', '',
              '![Paired timing reductions on RTX and Orin](timing_reductions.png)', '']
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/coslam-rtx-report-plots')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for i, variant in enumerate(VARIANTS[1:]):
        selected = [r for r in rows if r['variant'] == variant]
        for j, phase in enumerate(PHASES):
            ax = axes[i, j]
            x = np.arange(3)
            ax.bar(x - .18, [r[f'rtx_{phase}_reduction_pct'] for r in selected], .36, label='RTX 4070', color='#2166ac')
            ax.bar(x + .18, [r[f'orin_{phase}_reduction_pct'] for r in selected], .36, label='Orin', color='#d95f02')
            ax.set_xticks(x, SCENES)
            ax.axhline(0, color='black', linewidth=.7)
            ax.set_ylabel('Mean time reduction (%)')
            ax.set_title(variant + ' — ' + ('tracking call' if j == 0 else 'pure tracking frame'))
            ax.grid(axis='y', alpha=.2)
            ax.legend()
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(OUTPUT / f'timing_reductions.{suffix}', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(output=str(OUTPUT), aggregates=aggregates, baseline_repeats=repeat_rows,
                          independently_revalidated_runs=20), indent=2))


if __name__ == '__main__':
    main()
