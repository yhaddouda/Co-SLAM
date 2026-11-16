#!/usr/bin/env python3
"""Sweep depth-spatial tracking hyperparameters and evaluate Replica quality."""

import argparse
import csv
import itertools
import sys
from pathlib import Path

if __package__:
    from . import auto_eval_coslam_all_datasets_bucket as common
else:
    import auto_eval_coslam_all_datasets_bucket as common


DEFAULT_SCENES = ["office0"]
DEFAULT_SIZES = list(range(13, 25))
DEFAULT_MIN_COVERAGES = [0.75]
DEFAULT_GRID_HEIGHTS = [8]
DEFAULT_GRID_WIDTHS = [8]
DEFAULT_POOL_FACTORS = [4]

# Populated in main and read by apply_depth_spatial_configuration when the
# shared run/cull/evaluate workflow prepares each experiment configuration.
HYPERPARAMETER_DEFINITIONS = {}


def value_label(value: float) -> str:
    """Return a compact filesystem-safe label for a numeric value."""
    return format(value, "g").replace("-", "m").replace(".", "p")


def build_mode_name(
    min_coverage: float,
    grid_h: int,
    grid_w: int,
    pool_factor: int,
) -> str:
    return (
        f"depth_spatial_cov{value_label(min_coverage)}"
        f"_grid{grid_h}x{grid_w}_pool{pool_factor}"
    )


def apply_depth_spatial_configuration(cfg: dict, mode_name: str) -> None:
    params = HYPERPARAMETER_DEFINITIONS[mode_name]

    cfg.setdefault("grid", {})
    cfg.setdefault("training", {})
    cfg.setdefault("tracking_bucket", {})
    cfg.setdefault("tracking_naive_pruning", {})
    cfg.setdefault("tracking_depth_spatial", {})
    cfg.setdefault("ba_bucket", {})

    # Keep the non-hyperparameter parts of the depth_spatial mode fixed.
    cfg["grid"]["hash"] = "CoherentPrime"
    cfg["grid"]["morton_sort"] = False
    cfg["training"]["uniform_samples_until_depth"] = False
    cfg["tracking_bucket"]["enabled"] = False
    cfg["tracking_bucket"]["morton_inside_bucket"] = False
    cfg["tracking_naive_pruning"]["enabled"] = False
    cfg["tracking_depth_spatial"]["enabled"] = True
    cfg["ba_bucket"]["enabled"] = True
    cfg["ba_bucket"]["morton_inside_bucket"] = False

    cfg["tracking_depth_spatial"]["min_coverage"] = params["min_coverage"]
    cfg["tracking_depth_spatial"]["grid_h"] = params["grid_h"]
    cfg["tracking_depth_spatial"]["grid_w"] = params["grid_w"]
    cfg["tracking_depth_spatial"]["pool_factor"] = params["pool_factor"]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be an integer >= 1")
    return parsed


def coverage(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("coverage must be between 0 and 1")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Cartesian sweep of depth-spatial tracking "
            "hyperparameters on Replica."
        )
    )
    parser.add_argument(
        "--scenes", nargs="+", default=DEFAULT_SCENES, help="Replica scenes"
    )
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=1,
        help="Number of repetitions of the complete experiment grid",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SIZES,
        help="Hash-table size exponents",
    )
    parser.add_argument(
        "--min-coverages",
        "--min-coverage",
        dest="min_coverages",
        nargs="+",
        type=coverage,
        default=DEFAULT_MIN_COVERAGES,
    )
    parser.add_argument(
        "--grid-heights",
        "--grid-h",
        dest="grid_heights",
        nargs="+",
        type=positive_int,
        default=DEFAULT_GRID_HEIGHTS,
    )
    parser.add_argument(
        "--grid-widths",
        "--grid-w",
        dest="grid_widths",
        nargs="+",
        type=positive_int,
        default=DEFAULT_GRID_WIDTHS,
    )
    parser.add_argument(
        "--pool-factors",
        "--pool-factor",
        dest="pool_factors",
        nargs="+",
        type=positive_int,
        default=DEFAULT_POOL_FACTORS,
        help="Candidate-pool factors for tracking_depth_spatial",
    )
    parser.add_argument(
        "--configs-root", type=Path, default=Path("configs/Replica")
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../datasets/vv_data/Replica"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("output/replica_depth_spatial_hyperparameters.csv"),
    )
    parser.add_argument(
        "--output-prefix",
        default="depth_spatial_hyperparameters",
        help="Subdirectory prefix for CoSLAM outputs, logs, and temp configs",
    )
    parser.add_argument("--python-exec", default=sys.executable)
    return parser.parse_args()


def prepare_csv(path: Path, header: list[str]) -> None:
    common.ensure_dir(path.parent)
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerow(header)
        return

    with open(path, "r", newline="", encoding="utf-8") as csv_file:
        existing_header = next(csv.reader(csv_file), [])
    if existing_header != header:
        raise SystemExit(
            f"Existing CSV has an incompatible header: {path}\n"
            f"Expected: {','.join(header)}\n"
            "Use a new --output-csv path or update the existing CSV header."
        )


def main() -> None:
    args = parse_args()
    combinations = list(
        itertools.product(
            args.min_coverages,
            args.grid_heights,
            args.grid_widths,
            args.pool_factors,
        )
    )

    for min_coverage, grid_h, grid_w, pool_factor in combinations:
        mode_name = build_mode_name(min_coverage, grid_h, grid_w, pool_factor)
        params = {
            "min_coverage": min_coverage,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "pool_factor": pool_factor,
        }
        previous = HYPERPARAMETER_DEFINITIONS.setdefault(mode_name, params)
        if previous != params:
            raise SystemExit(f"Hyperparameter label collision: {mode_name}")

    common.SAMPLING_DEFINITIONS = HYPERPARAMETER_DEFINITIONS
    common.apply_sampling_mode = apply_depth_spatial_configuration

    use_run_column = args.runs > 1
    header = []
    if use_run_column:
        header.append("Run")
    header.extend(
        [
            "Scene",
            "Mode",
            "HashSize",
            "MinCoverage",
            "GridH",
            "GridW",
            "PoolFactor",
            "Accuracy",
            "Completion",
            "CompletionRatio",
            "DepthL1",
        ]
    )
    prepare_csv(args.output_csv, header)

    total_runs = (
        len(args.scenes) * len(args.sizes) * len(combinations) * args.runs
    )
    print(
        f"Total Runs Scheduled: {total_runs} "
        f"({args.runs} iterations of the grid)"
    )

    for run_idx in range(1, args.runs + 1):
        if use_run_column:
            print(f"\n=== STARTING RUN ITERATION {run_idx}/{args.runs} ===")

        jobs = itertools.product(args.scenes, combinations, args.sizes)
        for scene, params, hash_size in jobs:
            min_coverage, grid_h, grid_w, pool_factor = params
            mode_name = build_mode_name(
                min_coverage, grid_h, grid_w, pool_factor
            )
            effective_run_idx = run_idx if use_run_column else None
            metrics = common.run_sequence(
                args,
                scene,
                mode_name,
                hash_size,
                run_idx=effective_run_idx,
            )

            if metrics is None:
                run_prefix = f"Run {run_idx} " if use_run_column else ""
                print(
                    f"  > Failed: {run_prefix}{scene} {mode_name} "
                    f"T{hash_size}"
                )
                continue

            row = []
            if use_run_column:
                row.append(run_idx)
            row.extend(
                [
                    scene,
                    "depth_spatial",
                    hash_size,
                    min_coverage,
                    grid_h,
                    grid_w,
                    pool_factor,
                    metrics["Accuracy"],
                    metrics["Completion"],
                    metrics["Completion_Ratio"],
                    metrics["Depth_L1"],
                ]
            )
            print("  > Done: " + ",".join(map(str, row)))
            with open(
                args.output_csv, "a", newline="", encoding="utf-8"
            ) as csv_file:
                csv.writer(csv_file).writerow(row)


if __name__ == "__main__":
    main()
