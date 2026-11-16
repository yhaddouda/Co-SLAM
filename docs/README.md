# Experiment index

Start here for the experiments added during the BA and tracking investigation.
Existing setup/profiling guides remain in [README/Documentation_Index.md](../README/Documentation_Index.md).

| Task | Entry point |
|---|---|
| Genuine sample pruning with scene-mean accuracy and full-stage runtime gates | [Accepted report](../results/pruning_accepted/REPORT.md), [method](interval_support_pruning.md) |
| Transfer the experiments to the RTX 4070 | [RTX transfer instructions](rtx4070_coslam_handoff.md) |
| Understand and reproduce tracking / EC mapping variants | [Execution protocol](execution_experiments.md) |
| Read completed Orin tracking / EC mapping results | [Execution report](../results/execution_experiments/REPORT.md) |
| Understand and reproduce BA variants | [BA protocol](replica_ba_experiments.md) |
| Read completed Orin BA results | [BA report](../results/ba_experiments/full/REPORT.md) |
| Understand BA `baseline_fast` in detail | [Explanation PDF](baseline_fast_explained/baseline_fast_explained.pdf) |
| Find the benchmark/check/report scripts | [Validation script index](../scripts/Validation/README.md) |
| Explain the log files | [Archived driver log index](../results/execution_experiments/logs/orin/README.md) |
| Rebuild the transfer archive | [Packaging tool](../scripts/Experiments/README.md) |

The structure is:

```text
optimization/                       Optional algorithm/execution helpers
scripts/Validation/                 Benchmark, correctness and report commands
scripts/Experiments/                Transfer-package tool
tests/                              Semantic and loss checks
docs/                               Protocols, explanation PDFs and this index
experiments/ec_slam_execution/       Isolated EC source workspace and patch
results/ba_experiments/              BA observations, with source snapshots
results/execution_experiments/       Tracking/mapping observations and diagnostics
  logs/orin/                        Archived top-level job transcripts
transfer/rtx4070/                    Generated Co-SLAM transfer archive
```

Runtime files and historical result directories keep their original paths so
saved source hashes and existing commands remain valid. Old `.back` files,
earlier profiling/paper assets and unrelated user configuration changes remain
in their existing locations. The cleanup archives experiment transcripts rather
than deleting evidence.

For the latest Co-SLAM comparison, use `baseline`, `baseline_fast` and
`cached_fast` from the tracking suite. BA has a separate `baseline_fast` variant;
consult the matching protocol instead of combining the two stage changes.
