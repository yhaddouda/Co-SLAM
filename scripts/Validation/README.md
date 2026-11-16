# Experiment commands

Run commands from the Co-SLAM repository root. GPU benchmarks run serially in
the system's conda environment. The scripts keep their existing paths to preserve
reproduction commands and historical source snapshots.

| Purpose | Main commands |
|---|---|
| Sample pruning: 17% ATE / ≥30% BA on at least one scene | `run_ba30_pruning_study.py`; gate: `validate_ba30_pruning.py`; report: `write_ba30_report.py`; plot: `plot_ba30_pruning.py` |
| Accepted sample-pruning study | `run_accepted_pruning_study.py`; independent gate: `validate_pruning_acceptance.py`; plot: `plot_accepted_pruning.py` |
| Co-SLAM tracking suite | `run_replica_execution_suite.py --system coslam`; worker: `benchmark_replica_tracking.py` |
| EC tracking/mapping suite | `run_replica_execution_suite.py --system ec_slam`; worker: `benchmark_ec_replica_execution.py` |
| Co-SLAM BA suite | `run_replica_ba_suite.py`; worker: `benchmark_replica_ba.py` |
| Tracking/mapping report and plots | `summarize_execution_experiments.py` |
| BA report and independent ATE evaluation | `summarize_replica_ba.py --reevaluate --plot` |
| Audit saved tracking/mapping results | `audit_execution_results.py --require-complete` |
| Environment and hardware metadata | `capture_execution_environment.py`, `capture_execution_hardware.py` |
| Frozen-field / pose CUDA checks | `check_execution_gpu.py` |
| Dense tracking cache CUDA checks | `check_tracking_depth_cache_gpu.py` |
| BA lattice CUDA checks | `check_ba_lattice_gpu.py` |
| Separate Nsight traces and analysis | `trace_replica_execution.py`, `analyze_execution_trace.py`, `summarize_execution_traces.py` |
| Baseline repeat / telemetry analysis | `summarize_execution_diagnostics.py` |
| EC workspace preparation / patch export | `prepare_ec_execution_workspace.py`, `export_ec_execution_patch.py` |
| Initial scene ranking and bucket diagnostics | `rank_replica_ba_scenes.py`, `check_ba_bucket.py`, `compare_ba_bucket.py`, `summarize_ba_bucket.py` |

`execution_benchmark_common.py` supplies shared timing statistics, source
snapshots, phase seeds and ATE alignment. `check_ec_tracking_invocation_gpu.py`
is the iterative EC pose probe used by the GPU checker.

The original Orin queue used `run_execution_gpu_checks.py`,
`run_tracking_cache_checks.py`, `run_final_execution_validation.py`,
`watch_execution_reports.py` and `finalize_execution_experiments.py` to manage
that completed queue. These use its recorded result roots and dependencies;
use the explicit suite/report/audit commands in the RTX handoff for new RTX
runs. `monitor_execution_hardware.py` captures Orin sysfs telemetry; use
`nvidia-smi` for desktop GPU telemetry.

The `patch_validation/` directory and `test_smoothness.py` predate these suites
and remain available for their original checks.

New outer job transcripts belong under
`results/execution_experiments/logs/<machine>/`, not the repository root.
Each suite already writes per-scene/per-variant process logs inside its output
directory. Diagnostic timings are separate from the reported full-run timings.
