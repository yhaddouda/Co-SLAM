"""Summarize diagnostic trace counts and frozen CUDA checks, without headline times."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    labels = ('coslam_baseline_trace','coslam_baseline_fast_trace','coslam_cached_fast_trace',
              'ec_slam_baseline_trace','ec_slam_both_fast_trace')
    lines = ['# Execution traces and CUDA validation', '',
             'These are separate 30-frame office1 diagnostic runs on Orin. They include profiler overhead '
             'and an additional end fence, so their wall times are not benchmark speedups. '
             'The analysis retains 26 tracking calls after dropping the first three and six EC mapping '
             'calls after dropping initial first-frame mapping. Complete experiment calls do not overlap; '
             'all CUDA host threads in the same process are included.', '',
             '| Trace | Stage | Calls | Kernels/call | Sync APIs/call | H2D copies/call | H2D bytes/call | D2H copies/call | GPU busy ms/call |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    summaries = {}
    for label in labels:
        path = args.root/(label+'_analysis/summary.json')
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        summaries[label] = summary
        for stage, values in summary['stages'].items():
            m = {key:value['mean'] for key,value in values['metrics'].items()}
            lines.append(f"| {label} | {stage.replace('Experiment','').replace('Call','')} | {values['calls']} | "
                         f"{m['kernel_count']:.2f} | {m['synchronize_api_count']:.2f} | {m['h2d_count']:.2f} | "
                         f"{m['h2d_bytes']:.0f} | {m['d2h_count']:.2f} | {m['gpu_busy_union_ms']:.3f} |")
    lines += ['', 'Count interpretation:', '',
              '- Co `baseline_fast` removes 40 H2D calls per tracking call while reducing H2D bytes by '
              'only 2,000. That is primarily repeated small identity transfers and associated dispatch, '
              'not a large-bandwidth transfer optimization.',
              '- Co `cached_fast` removes an additional 27 H2D calls and 1,180,440 H2D bytes per call. '
              'It also removes repeated fixed depth-mask/lattice/bin work. CPU random perturbation '
              'draws and their per-iteration transfers are retained.',
              '- EC tracking removes 18 H2D calls but only 568 H2D bytes; 16 net D2H scalar copies and '
              '34 synchronization API calls are removed. Hundreds of remaining field/loss operations '
              'are unchanged, explaining why counts alone cannot promise a large stage speedup.',
              '- EC mapping removes 19 H2D calls and one D2H call in these short invocations. '
              'GPU kernel counts and busy time barely change. Fixed-reference caching chiefly removes '
              'host/transfer work; repeated CPU inverse savings grow with the keyframe history and '
              'are understated by a 30-frame trace.',
              '- GPU busy time is the union of kernel/copy/memset intervals. Summed CUDA API durations '
              'can overlap GPU work and include waiting; do not add them to GPU time or label them '
              'exclusive CPU overhead. Count reductions are evidence of removed operations, not '
              'a guarantee of unprofiled speedup.', '',
              'Pose-only microchecks (30 repetitions, warmup performed, CUDA synchronized):', '',
              '| System | Scope | Original mean ms | Device variant mean ms | Difference ms |',
              '|---|---|---:|---:|---:|']
    for system, fast_key in (('coslam','device_constants'),('ec_slam','device_allocation')):
        data = json.loads((args.root/(system+'_equivalence.json')).read_text())
        micro = data['pose_microcheck']
        original,fast = micro['baseline']['mean_ms'],micro[fast_key]['mean_ms']
        lines.append(f"| {system} | {micro['scope']} | {original:.3f} | {fast:.3f} | {original-fast:.3f} |")
    lines += ['', 'The microchecks include pose conversion only; they exclude field evaluation, backward '
              'and optimizer work. They must not be reported as complete-stage gains.', '',
              'Frozen CUDA checks:', '',
              '- Trained Co field, real frames 0/300/900: pose matrices and losses match exactly; '
              'maximum aggregate map/rotation/translation gradient relative L2 difference is about 1.0e-5.',
              '- Initialized real EC field, tracking and mapping: pose matrices and losses match exactly; '
              'gradient relative L2 differences are about 1.4e-8. Mapping reference values and pose gradients match exactly.',
              '- Co depth cache: nine comparisons (three changing poses on each of frames 0/300/900) '
              'have exact depth samples and forward values, and identical final CPU/CUDA RNG states. '
              'Small gradient differences are retained in coslam_depth_cache.json.',
              '- These checks are local numerical evidence. Full trajectories establish observed ATE; '
              'neither microchecks nor a single seed establish statistical accuracy equivalence.', '',
              'Raw artifacts include *.nsys-rep, *.sqlite, each *_analysis/calls.csv, '
              'per-API/kernel/copy/NVTX tables, source snapshots in trace_runs, and equivalence JSON files. '
              'The original trace_status.json and cache_trace_status.json record tool completion.']
    (args.root/'TRACE_REPORT.md').write_text('\n'.join(lines)+'\n')
    (args.root/'trace_summaries.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print('Wrote '+str(args.root/'TRACE_REPORT.md'))


if __name__ == '__main__':
    main()
