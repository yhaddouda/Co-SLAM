#!/usr/bin/env python3
"""Reproduce the aggregate runtime and quality values used in Section 6.

The script keeps component timing separate from the frame-wise aggregate and
records the source and normalization used for every reported result.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import fmean, stdev


SCENES = (
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
)
DEPTH_VALID_SCENES = tuple(scene for scene in SCENES if scene != "office4")
LOWER_IS_BETTER = {"Accuracy", "Completion", "DepthL1", "ATE_RMSE"}
RUNTIME_RUN_IDS = (1, 2, 3)
RUNTIME_EXPERIMENTS = (
    ("15 W", Path("Replica_cfgE_15W_csvs")),
    ("MAXN", Path("frame_total_MAXN") / "Replica"),
)
RUNTIME_MODE_TOKENS = {
    "baseline": "baseline",
    "pruning": "pruning",
    "combined_e": "pruningD",
}


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty result file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def reduction_percent(baseline: float, optimized: float) -> float:
    return 100.0 * (baseline - optimized) / baseline


def collect_repeated_runtime(
    repo_root: Path,
    timing_relative_root: Path,
    mode_tokens: dict[str, str],
) -> dict[str, dict[str, list[float]]]:
    """Return run-level Replica macro means after strict CSV validation."""

    timing_root = repo_root / timing_relative_root
    run_macros: dict[str, dict[str, list[float]]] = {
        metric: {mode: [] for mode in mode_tokens}
        for metric in ("tracking", "bundle_adjustment", "frame")
    }

    for run in RUNTIME_RUN_IDS:
        for mode, token in mode_tokens.items():
            scene_means = {
                metric: [] for metric in ("tracking", "bundle_adjustment", "frame")
            }
            for scene in SCENES:
                source = timing_root / (
                    f"{scene}_frame_total_timing_complete_{token}"
                    f"_Size19_Run{run}.csv"
                )
                totals_by_frame: dict[int, tuple[float, float, float]] = {}
                with source.open(newline="", encoding="utf-8") as csv_file:
                    for row in csv.reader(csv_file):
                        if len(row) != 4:
                            raise ValueError(
                                f"Expected four columns in {source}, got: {row}"
                            )
                        frame_id = int(row[0])
                        if frame_id in totals_by_frame:
                            raise ValueError(f"Duplicate frame {frame_id} in {source}")
                        total_ms, tracking_ms, ba_ms = map(float, row[1:])
                        values = (total_ms, tracking_ms, ba_ms)
                        if any(not isfinite(value) or value < 0.0 for value in values):
                            raise ValueError(
                                f"Expected finite nonnegative timings at frame "
                                f"{frame_id} in {source}"
                            )
                        if abs(total_ms - tracking_ms - ba_ms) > 1e-5:
                            raise ValueError(
                                f"Component total mismatch at frame {frame_id} "
                                f"in {source}"
                            )
                        totals_by_frame[frame_id] = (
                            total_ms,
                            tracking_ms,
                            ba_ms,
                        )

                if sorted(totals_by_frame) != list(range(2000)):
                    raise ValueError(f"Expected frame ids 0--1999 in {source}")
                if totals_by_frame[0] != (0.0, 0.0, 0.0):
                    raise ValueError(f"Expected zero initialization timing in {source}")

                noninitial = [totals_by_frame[frame] for frame in range(1, 2000)]
                tracking_ids = [
                    frame for frame in range(1, 2000)
                    if totals_by_frame[frame][1] > 0.0
                ]
                ba_ids = [
                    frame for frame in range(1, 2000)
                    if totals_by_frame[frame][2] > 0.0
                ]
                if tracking_ids != list(range(1, 2000)):
                    raise ValueError(
                        f"Unexpected tracking-frame support in {source}"
                    )
                if ba_ids != list(range(5, 2000, 5)):
                    raise ValueError(
                        f"Unexpected bundle-adjustment support in {source}"
                    )

                tracking = [totals_by_frame[frame][1] for frame in tracking_ids]
                bundle_adjustment = [
                    totals_by_frame[frame][2] for frame in ba_ids
                ]
                tracking_mean = fmean(tracking)
                ba_mean = fmean(bundle_adjustment)
                frame_mean = fmean(row[0] for row in noninitial)
                expected_frame_mean = tracking_mean + len(ba_ids) / 1999 * ba_mean
                if abs(frame_mean - expected_frame_mean) > 1e-5:
                    raise ValueError(
                        f"Frame/component mean mismatch in {source}"
                    )

                scene_means["tracking"].append(tracking_mean)
                scene_means["bundle_adjustment"].append(ba_mean)
                scene_means["frame"].append(frame_mean)

            for metric, values in scene_means.items():
                if len(values) != len(SCENES):
                    raise ValueError(
                        f"Expected {len(SCENES)} scene means for {mode}, "
                        f"run {run}, metric {metric}"
                    )
                run_macros[metric][mode].append(fmean(values))

    for metric, by_mode in run_macros.items():
        for mode, values in by_mode.items():
            if len(values) != len(RUNTIME_RUN_IDS):
                raise ValueError(
                    f"Expected {len(RUNTIME_RUN_IDS)} run macros for "
                    f"{metric}/{mode}"
                )
    return run_macros


def runtime_metadata() -> dict[str, tuple[str, str, int, object, str]]:
    return {
        "tracking": (
            "Tracking stage",
            "ms per stage",
            1999,
            10,
            "sum of ten tracking-iteration profile ranges per noninitial frame; "
            "outer pose setup and finalization excluded",
        ),
        "bundle_adjustment": (
            "Bundle-adjustment invocation",
            "ms per invocation",
            399,
            10,
            "sum of ten bundle-adjustment-iteration profile ranges per "
            "invocation; outside-range work including pruning bucket preparation "
            "excluded",
        ),
        "frame": (
            "Frame-wise Replica average",
            "ms per frame",
            1999,
            "10 tracking; +10 bundle adjustment every fifth frame",
            "tracking plus bundle-adjustment profile ranges per noninitial frame; "
            "outside-range work excluded",
        ),
    }


def source_glob(
    timing_relative_root: Path,
    tokens: tuple[str, ...],
) -> str:
    token_group = ",".join(tokens)
    return (
        f"{timing_relative_root.as_posix()}/"
        "*_frame_total_timing_complete_"
        f"{{{token_group}}}_Size19_Run{{1,2,3}}.csv"
    )


def common_runtime_fields(
    timing_relative_root: Path,
    *,
    observation_count: int,
    iterations_per_observation: object,
    timer_boundary: str,
    tokens: tuple[str, ...],
) -> dict[str, object]:
    return {
        "Run_Count": len(RUNTIME_RUN_IDS),
        "Run_IDs": "1;2;3",
        "Scene_Count": len(SCENES),
        "Observations_Per_Scene_Run": observation_count,
        "Iterations_Per_Observation": iterations_per_observation,
        "Hash_Size_Log2": 19,
        "Timer_Boundary": timer_boundary,
        "Source": source_glob(timing_relative_root, tokens),
        "Selection_Policy": (
            "runs 1--3 only; run 4 excluded because scene/config coverage "
            "is incomplete"
        ),
        "Aggregation": (
            "observation mean within scene; 8-scene macro mean within run; "
            "mean and sample SD across runs"
        ),
        "Reduction_Definition": "computed from reported aggregate means",
    }


def write_component_timing(repo_root: Path, output_path: Path) -> None:
    """Write repeated-run baseline/pruning component timing for both power modes."""

    mode_tokens = {
        key: RUNTIME_MODE_TOKENS[key] for key in ("baseline", "pruning")
    }
    metadata = runtime_metadata()
    output_rows: list[dict[str, object]] = []
    for power_mode, timing_relative_root in RUNTIME_EXPERIMENTS:
        run_macros = collect_repeated_runtime(
            repo_root, timing_relative_root, mode_tokens
        )
        for metric in ("tracking", "bundle_adjustment"):
            observations = run_macros[metric]
            baseline = fmean(observations["baseline"])
            pruning = fmean(observations["pruning"])
            (
                component,
                unit,
                observation_count,
                iterations_per_observation,
                timer_boundary,
            ) = metadata[metric]
            output_rows.append(
                {
                    "Power_Mode": power_mode,
                    "Component": component,
                    "Unit": unit,
                    "Baseline_Mean": f"{baseline:.9f}",
                    "Baseline_Sample_SD": (
                        f"{stdev(observations['baseline']):.9f}"
                    ),
                    "Pruning_Mean": f"{pruning:.9f}",
                    "Pruning_Sample_SD": (
                        f"{stdev(observations['pruning']):.9f}"
                    ),
                    "Reduction_pct": (
                        f"{reduction_percent(baseline, pruning):.9f}"
                    ),
                    **common_runtime_fields(
                        timing_relative_root,
                        observation_count=observation_count,
                        iterations_per_observation=iterations_per_observation,
                        timer_boundary=timer_boundary,
                        tokens=("baseline", "pruning"),
                    ),
                }
            )
    write_rows(output_path, output_rows)


def write_frame_execution_time(repo_root: Path, output_path: Path) -> None:
    """Write repeated-run baseline/pruning frame-wise timing for both modes."""

    mode_tokens = {
        key: RUNTIME_MODE_TOKENS[key] for key in ("baseline", "pruning")
    }
    metadata = runtime_metadata()
    output_rows: list[dict[str, object]] = []
    for power_mode, timing_relative_root in RUNTIME_EXPERIMENTS:
        run_macros = collect_repeated_runtime(
            repo_root, timing_relative_root, mode_tokens
        )
        observations = run_macros["frame"]
        baseline = fmean(observations["baseline"])
        pruning = fmean(observations["pruning"])
        (
            scope,
            unit,
            observation_count,
            iterations_per_observation,
            timer_boundary,
        ) = metadata["frame"]
        output_rows.append(
            {
                "Power_Mode": power_mode,
                "Scope": scope,
                "Unit": unit,
                "Baseline_Mean": f"{baseline:.9f}",
                "Baseline_Sample_SD": (
                    f"{stdev(observations['baseline']):.9f}"
                ),
                "Pruning_Mean": f"{pruning:.9f}",
                "Pruning_Sample_SD": (
                    f"{stdev(observations['pruning']):.9f}"
                ),
                "Reduction_pct": f"{reduction_percent(baseline, pruning):.9f}",
                **common_runtime_fields(
                    timing_relative_root,
                    observation_count=observation_count,
                    iterations_per_observation=iterations_per_observation,
                    timer_boundary=timer_boundary,
                    tokens=("baseline", "pruning"),
                ),
            }
        )
    write_rows(output_path, output_rows)


def write_combined_execution_time(repo_root: Path, output_path: Path) -> None:
    """Write repeated baseline/pruning/config-E timing for both power modes."""

    metadata = {
        "tracking": (
            "Tracking stage",
            "ms per stage",
            1999,
            10,
            "sum of ten tracking-iteration profile ranges per noninitial frame; "
            "outer pose setup/finalization excluded; E's Morton work inside "
            "the forward pass included",
        ),
        "bundle_adjustment": (
            "Bundle-adjustment invocation",
            "ms per invocation",
            399,
            10,
            "sum of ten bundle-adjustment-iteration profile ranges per "
            "invocation; outside-range work including pruning bucket preparation "
            "excluded; E's Morton work inside the forward pass included",
        ),
        "frame": (
            "Frame-wise Replica average",
            "ms per frame",
            1999,
            "10 tracking; +10 bundle adjustment every fifth frame",
            "tracking plus bundle-adjustment profile ranges per noninitial frame; "
            "outside-range work excluded; E's Morton work inside the forward pass "
            "included",
        ),
    }
    output_rows: list[dict[str, object]] = []
    for power_mode, timing_relative_root in RUNTIME_EXPERIMENTS:
        run_macros = collect_repeated_runtime(
            repo_root, timing_relative_root, RUNTIME_MODE_TOKENS
        )
        for metric, observations in run_macros.items():
            baseline = fmean(observations["baseline"])
            pruning = fmean(observations["pruning"])
            combined_e = fmean(observations["combined_e"])
            (
                scope,
                unit,
                observation_count,
                iterations_per_observation,
                timer_boundary,
            ) = metadata[metric]
            output_rows.append(
                {
                    "Power_Mode": power_mode,
                    "Scope": scope,
                    "Unit": unit,
                    "Baseline_Mean": f"{baseline:.9f}",
                    "Baseline_Sample_SD": (
                        f"{stdev(observations['baseline']):.9f}"
                    ),
                    "Pruning_Mean": f"{pruning:.9f}",
                    "Pruning_Sample_SD": (
                        f"{stdev(observations['pruning']):.9f}"
                    ),
                    "Combined_E_Mean": f"{combined_e:.9f}",
                    "Combined_E_Sample_SD": (
                        f"{stdev(observations['combined_e']):.9f}"
                    ),
                    "Pruning_Reduction_vs_A_pct": (
                        f"{reduction_percent(baseline, pruning):.9f}"
                    ),
                    "Combined_E_Reduction_vs_A_pct": (
                        f"{reduction_percent(baseline, combined_e):.9f}"
                    ),
                    "Combined_E_Reduction_vs_Pruning_pct": (
                        f"{reduction_percent(pruning, combined_e):.9f}"
                    ),
                    **common_runtime_fields(
                        timing_relative_root,
                        observation_count=observation_count,
                        iterations_per_observation=iterations_per_observation,
                        timer_boundary=timer_boundary,
                        tokens=("baseline", "pruning", "pruningD"),
                    ),
                }
            )
    write_rows(output_path, output_rows)


def complete_rows_by_run(
    rows: list[dict[str, str]],
    *,
    run_key: str,
    variants: tuple[str, str],
    variant_key: str,
) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row[variant_key] in variants:
            grouped[int(row[run_key])].append(row)
    expected = {(scene, variant) for scene in SCENES for variant in variants}
    return {
        run: run_rows
        for run, run_rows in grouped.items()
        if {(row["Scene"], row[variant_key]) for row in run_rows} == expected
    }


def macro_pair(
    rows: list[dict[str, str]],
    *,
    metric: str,
    variant_key: str,
    variants: tuple[str, str],
    scale: float,
    scenes: tuple[str, ...],
) -> tuple[float, float]:
    values: dict[str, dict[str, float]] = {variant: {} for variant in variants}
    for row in rows:
        scene = row["Scene"]
        variant = row[variant_key]
        if scene in scenes and variant in values:
            key = (scene, variant)
            if scene in values[variant]:
                raise ValueError(f"Duplicate quality row for {key}")
            values[variant][scene] = float(row[metric]) * scale
    for variant in variants:
        if set(values[variant]) != set(scenes):
            raise ValueError(f"Incomplete {metric} macro for variant {variant}")
    return tuple(fmean(values[variant].values()) for variant in variants)  # type: ignore[return-value]


def locality_csv_runs(repo_root: Path) -> dict[int, list[dict[str, str]]]:
    root = repo_root / "Quality" / "hash grid optim"
    runs: dict[int, list[dict[str, str]]] = {}
    for path in sorted(root.glob("replica_results_setups_run*.csv")):
        match = re.search(r"run(\d+)\.csv$", path.name)
        if match is None:
            continue
        run = int(match.group(1))
        rows = read_dict_rows(path)
        if {int(row["HashSize"]) for row in rows} != {16}:
            raise ValueError(f"Unexpected locality quality hash size in {path}")
        expected = {(scene, setup) for scene in SCENES for setup in ("A", "B", "C", "D")}
        if {(row["Scene"], row["Setup"]) for row in rows} != expected:
            raise ValueError(f"Incomplete locality quality run: {path}")
        runs[run] = rows
    return runs


def locality_ate_runs(repo_root: Path) -> dict[int, list[dict[str, str]]]:
    """Read the latest complete 2,000-pose A/D evaluation in each run file."""

    runs: dict[int, list[dict[str, str]]] = {}
    for run in range(1, 10):
        run_rows: list[dict[str, str]] = []
        complete = True
        for scene in SCENES:
            for setup in ("A", "D"):
                path = (
                    repo_root
                    / "output"
                    / "Replica"
                    / scene
                    / f"run_{run}"
                    / f"{setup}_Size19_Run{run}"
                    / "output.txt"
                )
                if not path.exists():
                    complete = False
                    continue
                latest: dict[str, object] | None = None
                with path.open(encoding="utf-8") as output_file:
                    for line in output_file:
                        if not line.strip():
                            continue
                        record = ast.literal_eval(line)
                        if (
                            record.get("compared_pose_pairs") == 2000
                            and record.get("Name") == 1999
                        ):
                            latest = record
                if latest is None:
                    complete = False
                    continue
                run_rows.append(
                    {
                        "Scene": scene,
                        "Setup": setup,
                        "ATE_RMSE": str(latest["absolute_translational_error.rmse"]),
                    }
                )
        if complete and len(run_rows) == 2 * len(SCENES):
            runs[run] = run_rows
    return runs


def append_quality_metric(
    run_macro_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    *,
    optimization: str,
    metric: str,
    unit: str,
    hash_size: int,
    run_rows: dict[int, list[dict[str, str]]],
    variant_key: str,
    variants: tuple[str, str],
    scale: float,
    scenes: tuple[str, ...],
    source: str,
    selection_policy: str,
) -> None:
    macros: list[tuple[int, float, float]] = []
    for run, rows in sorted(run_rows.items()):
        baseline, optimized = macro_pair(
            rows,
            metric=metric,
            variant_key=variant_key,
            variants=variants,
            scale=scale,
            scenes=scenes,
        )
        macros.append((run, baseline, optimized))
        run_macro_rows.append(
            {
                "Optimization": optimization,
                "Metric": metric,
                "Unit": unit,
                "Log2_Hash_Size": hash_size,
                "Run": run,
                "Scene_Count": len(scenes),
                "Baseline_Macro": f"{baseline:.9f}",
                "Optimized_Macro": f"{optimized:.9f}",
                "Source": source,
                "Selection_Policy": selection_policy,
            }
        )

    baseline_values = [row[1] for row in macros]
    optimized_values = [row[2] for row in macros]
    baseline_mean = fmean(baseline_values)
    optimized_mean = fmean(optimized_values)
    absolute_change = optimized_mean - baseline_mean
    relative_change = 100.0 * absolute_change / baseline_mean
    summary_rows.append(
        {
            "Optimization": optimization,
            "Metric": metric,
            "Direction": "lower" if metric in LOWER_IS_BETTER else "higher",
            "Unit": unit,
            "Log2_Hash_Size": hash_size,
            "Baseline_Mean": f"{baseline_mean:.9f}",
            "Baseline_Sample_SD": f"{stdev(baseline_values):.9f}",
            "Optimized_Mean": f"{optimized_mean:.9f}",
            "Optimized_Sample_SD": f"{stdev(optimized_values):.9f}",
            "Absolute_Change": f"{absolute_change:.9f}",
            "Relative_Change_pct": (
                "" if metric == "CompletionRatio" else f"{relative_change:.9f}"
            ),
            "Completion_Ratio_Change_pp": (
                f"{absolute_change:.9f}" if metric == "CompletionRatio" else ""
            ),
            "Run_Count": len(macros),
            "Scene_Count": len(scenes),
            "Excluded_Scenes": "office4" if metric == "DepthL1" else "",
            "Source": source,
            "Selection_Policy": selection_policy,
        }
    )


def write_quality_results(repo_root: Path, output_dir: Path) -> None:
    run_macro_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    locality_runs = locality_csv_runs(repo_root)
    for metric, unit, scale in (
        ("Accuracy", "cm", 1.0),
        ("Completion", "cm", 1.0),
        ("CompletionRatio", "%", 1.0),
        ("DepthL1", "mm", 1000.0),
    ):
        append_quality_metric(
            run_macro_rows,
            summary_rows,
            optimization="hash_locality",
            metric=metric,
            unit=unit,
            hash_size=16,
            run_rows=locality_runs,
            variant_key="Setup",
            variants=("A", "D"),
            scale=scale,
            scenes=DEPTH_VALID_SCENES if metric == "DepthL1" else SCENES,
            source="Quality/hash grid optim/replica_results_setups_run0--5.csv",
            selection_policy="all six complete run CSVs",
        )

    append_quality_metric(
        run_macro_rows,
        summary_rows,
        optimization="hash_locality",
        metric="ATE_RMSE",
        unit="cm",
        hash_size=19,
        run_rows=locality_ate_runs(repo_root),
        variant_key="Setup",
        variants=("A", "D"),
        scale=100.0,
        scenes=SCENES,
        source="output/Replica/<scene>/run_1--5/{A,D}_Size19_Run*/output.txt",
        selection_policy="latest complete 2,000-pose evaluation per run directory",
    )

    depth_root = repo_root / "Quality" / "depth sampling"
    depth_metric_rows = read_dict_rows(
        depth_root / "replica_tracking_ba_ablation_results.csv"
    )
    ate_rows = read_dict_rows(depth_root / "replica_tracking_ba_ate_results.csv")
    complete_metric_runs = complete_rows_by_run(
        depth_metric_rows,
        run_key="Run",
        variants=("A", "D"),
        variant_key="Config",
    )
    complete_ate_runs = complete_rows_by_run(
        ate_rows,
        run_key="Run",
        variants=("A", "D"),
        variant_key="Config",
    )
    if set(complete_metric_runs) != set(range(1, 8)):
        raise ValueError(f"Unexpected complete pruning metric runs: {complete_metric_runs}")
    if set(complete_ate_runs) != set(range(1, 8)):
        raise ValueError(f"Unexpected complete pruning ATE runs: {complete_ate_runs}")

    for metric, unit, scale, source_rows, source_name in (
        (
            "Accuracy",
            "cm",
            1.0,
            complete_metric_runs,
            "Quality/depth sampling/replica_tracking_ba_ablation_results.csv",
        ),
        (
            "Completion",
            "cm",
            1.0,
            complete_metric_runs,
            "Quality/depth sampling/replica_tracking_ba_ablation_results.csv",
        ),
        (
            "CompletionRatio",
            "%",
            1.0,
            complete_metric_runs,
            "Quality/depth sampling/replica_tracking_ba_ablation_results.csv",
        ),
        (
            "DepthL1",
            "mm",
            1000.0,
            complete_metric_runs,
            "Quality/depth sampling/replica_tracking_ba_ablation_results.csv",
        ),
        (
            "ATE_RMSE",
            "cm",
            100.0,
            complete_ate_runs,
            "Quality/depth sampling/replica_tracking_ba_ate_results.csv",
        ),
    ):
        append_quality_metric(
            run_macro_rows,
            summary_rows,
            optimization="depth_pruning",
            metric=metric,
            unit=unit,
            hash_size=19,
            run_rows=source_rows,
            variant_key="Config",
            variants=("A", "D"),
            scale=scale,
            scenes=DEPTH_VALID_SCENES if metric == "DepthL1" else SCENES,
            source=source_name,
            selection_policy="complete runs 1--7; internal A=unpruned, D=tracking+BA pruning",
        )

    combined_metric_rows = read_dict_rows(
        repo_root / "output" / "replica_tracking_ba_ablation_results.csv"
    )
    combined_ate_rows = read_dict_rows(
        repo_root / "output" / "replica_tracking_ba_ate_results.csv"
    )
    complete_combined_metric_runs = complete_rows_by_run(
        combined_metric_rows,
        run_key="Run",
        variants=("A", "E"),
        variant_key="Config",
    )
    complete_combined_ate_runs = complete_rows_by_run(
        combined_ate_rows,
        run_key="Run",
        variants=("A", "E"),
        variant_key="Config",
    )
    expected_combined_runs = set(range(1, 6))
    if set(complete_combined_metric_runs) != expected_combined_runs:
        raise ValueError(
            "Unexpected complete combined metric runs: "
            f"{complete_combined_metric_runs}"
        )
    if set(complete_combined_ate_runs) != expected_combined_runs:
        raise ValueError(
            f"Unexpected complete combined ATE runs: {complete_combined_ate_runs}"
        )

    for metric, unit, scale, source_rows, source_name in (
        (
            "Accuracy",
            "cm",
            1.0,
            complete_combined_metric_runs,
            "output/replica_tracking_ba_ablation_results.csv",
        ),
        (
            "Completion",
            "cm",
            1.0,
            complete_combined_metric_runs,
            "output/replica_tracking_ba_ablation_results.csv",
        ),
        (
            "CompletionRatio",
            "%",
            1.0,
            complete_combined_metric_runs,
            "output/replica_tracking_ba_ablation_results.csv",
        ),
        (
            "DepthL1",
            "mm",
            1000.0,
            complete_combined_metric_runs,
            "output/replica_tracking_ba_ablation_results.csv",
        ),
        (
            "ATE_RMSE",
            "cm",
            100.0,
            complete_combined_ate_runs,
            "output/replica_tracking_ba_ate_results.csv",
        ),
    ):
        append_quality_metric(
            run_macro_rows,
            summary_rows,
            optimization="combined_optimizations",
            metric=metric,
            unit=unit,
            hash_size=19,
            run_rows=source_rows,
            variant_key="Config",
            variants=("A", "E"),
            scale=scale,
            scenes=DEPTH_VALID_SCENES if metric == "DepthL1" else SCENES,
            source=source_name,
            selection_policy=(
                "complete runs 1--5; A=baseline, "
                "E=depth-aware reduction + Morton hashing and sorting"
            ),
        )

    write_rows(output_dir / "section6_quality_run_macros.csv", run_macro_rows)
    write_rows(output_dir / "section6_quality_summary.csv", summary_rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory for derived CSV files (default: script directory).",
    )
    parser.add_argument(
        "--section",
        choices=(
            "all",
            "component-timing",
            "frame-timing",
            "combined-runtime",
            "quality",
        ),
        default="all",
        help=(
            "Select one derived result when only its source data are available "
            "(default: all)."
        ),
    )
    args = parser.parse_args()

    if args.section in ("all", "component-timing"):
        write_component_timing(
            repo_root, args.output_dir / "section6_depth_pruning_execution_time.csv"
        )
    if args.section in ("all", "frame-timing"):
        write_frame_execution_time(
            repo_root, args.output_dir / "section6_frame_execution_time.csv"
        )
    if args.section in ("all", "combined-runtime"):
        write_combined_execution_time(
            repo_root, args.output_dir / "section6_combined_execution_time.csv"
        )
    if args.section in ("all", "quality"):
        write_quality_results(repo_root, args.output_dir)


if __name__ == "__main__":
    main()
