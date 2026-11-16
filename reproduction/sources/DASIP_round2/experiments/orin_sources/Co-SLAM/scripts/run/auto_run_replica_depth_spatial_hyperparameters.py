#!/usr/bin/env python3
"""Sweep Replica depth-spatial hyperparameters in statistics-only mode."""

import argparse
import csv
import itertools
import math
import subprocess
import sys
from pathlib import Path

import auto_run_replica_iteration_timings as common


DEFAULT_STATS_DIR = Path(
    "results/depth_spatial_sample_count_sweep/Replica"
)
DEFAULT_LOG_FILE = Path(
    "output/Replica_depth_spatial_sample_count_sweep.csv"
)
SAMPLE_STATS_COLUMNS = (
    "scene",
    "spatial_frames",
    "mean_uniform_samples",
    "stddev_uniform_samples",
    "median_uniform_samples",
    "p95_uniform_samples",
    "min_uniform_samples",
    "max_uniform_samples",
)
LOG_COLUMNS = (
    "Run",
    "Scene",
    "Seed",
    "PoolFactor",
    "GridH",
    "GridW",
    "MinCoverage",
    "FrameLimit",
    "WarmupFrames",
    "SpatialFrames",
    "MeanUniformSamples",
    "StddevUniformSamples",
    "MedianUniformSamples",
    "P95UniformSamples",
    "MinUniformSamples",
    "MaxUniformSamples",
    "SampleCountStatsCsv",
)


def float_label(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def parameter_tag(
    pool_factor: int,
    grid_h: int,
    grid_w: int,
    min_coverage: float,
) -> str:
    return (
        f"pf{pool_factor}_gh{grid_h}_gw{grid_w}_"
        f"mc{float_label(min_coverage)}"
    )


def build_run_stem(
    scene: str,
    run_idx: int,
    pool_factor: int,
    grid_h: int,
    grid_w: int,
    min_coverage: float,
) -> str:
    tag = parameter_tag(pool_factor, grid_h, grid_w, min_coverage)
    return f"{scene}_{tag}_Run{run_idx}"


def build_stats_path(
    stats_dir: Path,
    scene: str,
    run_idx: int,
    pool_factor: int,
    grid_h: int,
    grid_w: int,
    min_coverage: float,
) -> Path:
    stem = build_run_stem(
        scene,
        run_idx,
        pool_factor,
        grid_h,
        grid_w,
        min_coverage,
    )
    return stats_dir / f"{stem}_sample_counts.csv"


def override_config(
    base_cfg_path: Path,
    run_idx: int,
    frame_count: int | None,
    warmup_frames: int,
    sample_stats_csv: Path,
    stats_seed: int,
    pool_factor: int,
    grid_h: int,
    grid_w: int,
    min_coverage: float,
) -> Path:
    cfg = common.read_yaml(base_cfg_path)
    tag = parameter_tag(pool_factor, grid_h, grid_w, min_coverage)

    cfg.setdefault("tracking_depth_spatial", {})
    cfg["tracking_depth_spatial"].update({
        "enabled": True,
        "pool_factor": pool_factor,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "min_coverage": min_coverage,
        "collect_fallback_stats": False,
        "collect_sample_count_stats": True,
        "stats_only": True,
        "stats_seed": stats_seed,
        "stats_warmup_frames": warmup_frames,
        "sample_count_stats_csv": common.format_config_path(sample_stats_csv),
    })

    cfg.setdefault("tracking_bucket", {})
    cfg["tracking_bucket"]["enabled"] = False
    cfg.setdefault("tracking_naive_pruning", {})
    cfg["tracking_naive_pruning"]["enabled"] = False
    cfg["tracking_naive_pruning"]["collect_stats"] = False
    cfg.setdefault("training", {})
    cfg["training"]["uniform_samples_until_depth"] = False

    # No optimization or BA is executed in statistics-only mode.
    cfg.setdefault("ba_bucket", {})
    cfg["ba_bucket"]["enabled"] = False
    cfg["ba_bucket"]["morton_inside_bucket"] = False

    cfg.setdefault("mapping", {})
    cfg["mapping"]["first_mesh"] = False
    cfg.setdefault("mesh", {})
    cfg["mesh"]["vis"] = 1_000_000_000
    cfg["mesh"]["visualisation"] = False
    cfg.setdefault("profiling", {})
    cfg["profiling"]["enabled"] = False
    cfg["sync_for_ncu"] = False

    cfg.setdefault("timing", {})
    cfg["timing"]["mode"] = "none"
    cfg["timing"]["warmup_frames"] = warmup_frames
    cfg["timing"]["max_frames"] = frame_count
    cfg["timing"]["disable_eval"] = True

    cfg.setdefault("data", {})
    cfg["data"]["exp_name"] = f"depth_spatial_sweep_{tag}_Run{run_idx}"
    base_output = Path(
        cfg["data"].get("output", f"output/Replica/{base_cfg_path.stem}")
    )
    cfg["data"]["output"] = str(
        base_output / "depth_spatial_hyperparameters" / f"run_{run_idx}"
    )

    temp_path = base_cfg_path.parent / (
        f"temp_{base_cfg_path.stem}_{cfg['data']['exp_name']}.yaml"
    )
    common.write_yaml(temp_path, cfg)
    return temp_path


def validate_sample_stats(stats_csv: Path, expected_scene: str) -> dict:
    if not stats_csv.is_file():
        raise RuntimeError(f"sample-count summary was not created: {stats_csv}")
    with open(stats_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SAMPLE_STATS_COLUMNS:
            raise RuntimeError(f"unexpected sample-count header: {stats_csv}")
        rows = list(reader)
    if len(rows) != 1 or rows[0]["scene"] != expected_scene:
        raise RuntimeError(
            f"sample-count CSV must contain one row for {expected_scene}: "
            f"{stats_csv}"
        )

    row = rows[0]
    try:
        frames = int(row["spatial_frames"])
        values = [float(row[column]) for column in SAMPLE_STATS_COLUMNS[2:]]
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid sample-count summary: {stats_csv}") from error
    if frames <= 0 or any(not math.isfinite(value) or value < 0 for value in values):
        raise RuntimeError(f"invalid sample-count summary: {stats_csv}")
    if not (
        values[-2] <= values[0] <= values[-1]
        and values[-2] <= values[2] <= values[3] <= values[-1]
    ):
        raise RuntimeError(f"inconsistent sample-count quantiles: {stats_csv}")
    return row


def prepare_log_file(log_file: Path) -> None:
    common.ensure_dir(log_file.parent)
    if not log_file.exists():
        with open(log_file, "w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(LOG_COLUMNS)
        return
    with open(log_file, "r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != list(LOG_COLUMNS):
        raise RuntimeError(
            f"Existing log '{log_file}' has an incompatible header. "
            "Use --log-file with a fresh path."
        )


def append_log_row(
    log_file: Path,
    run_idx: int,
    scene: str,
    stats_seed: int,
    pool_factor: int,
    grid_h: int,
    grid_w: int,
    min_coverage: float,
    args: argparse.Namespace,
    stats: dict,
    stats_csv: Path,
) -> None:
    with open(log_file, "a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([
            run_idx,
            scene,
            stats_seed,
            pool_factor,
            grid_h,
            grid_w,
            min_coverage,
            "All" if args.frame_count is None else args.frame_count,
            args.warmup_frames,
            stats["spatial_frames"],
            stats["mean_uniform_samples"],
            stats["stddev_uniform_samples"],
            stats["median_uniform_samples"],
            stats["p95_uniform_samples"],
            stats["min_uniform_samples"],
            stats["max_uniform_samples"],
            stats_csv,
        ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep depth-spatial tracking pool, grid, and coverage settings "
            "over Replica scenes. Parameter lists form a Cartesian product."
        )
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--pool-factors", nargs="+", type=int, default=[4])
    parser.add_argument("--grid-hs", nargs="+", type=int, default=[8])
    parser.add_argument("--grid-ws", nargs="+", type=int, default=[8])
    parser.add_argument(
        "--min-coverages",
        nargs="+",
        type=float,
        default=[0.75],
    )
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument(
        "--configs-root", type=Path, default=Path("configs/Replica")
    )
    parser.add_argument(
        "--frame-count",
        type=common.parse_frame_count_arg,
        default=None,
        metavar="N|all",
    )
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument("--target-script", type=Path, default=Path("coslam.py"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp-configs", action="store_true")
    return parser.parse_args()


def unique(values):
    return list(dict.fromkeys(values))


def validate_args(args: argparse.Namespace) -> None:
    if common.yaml is None:
        raise RuntimeError("PyYAML is required")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative")
    if any(value < 1 for value in args.pool_factors):
        raise ValueError("--pool-factors values must be at least 1")
    if any(value < 1 for value in args.grid_hs + args.grid_ws):
        raise ValueError("grid dimensions must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.min_coverages):
        raise ValueError("--min-coverages values must be between 0 and 1")
    if not args.target_script.is_file():
        raise FileNotFoundError(f"target script not found: {args.target_script}")
    if not args.configs_root.is_dir():
        raise FileNotFoundError(f"configs root not found: {args.configs_root}")


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        scenes = common.discover_scenes(args.configs_root, args.scenes)
        parameter_sets = list(itertools.product(
            unique(args.pool_factors),
            unique(args.grid_hs),
            unique(args.grid_ws),
            unique(args.min_coverages),
        ))
        if not args.dry_run:
            prepare_log_file(args.log_file)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)

    total_runs = len(scenes) * len(parameter_sets) * args.runs
    print(f"Total runs scheduled: {total_runs}")
    print(f"Scenes: {', '.join(scenes)}")
    print(f"Hyperparameter combinations: {len(parameter_sets)}")
    print("-" * 72)

    successes = 0
    failures = 0
    for run_idx in range(1, args.runs + 1):
        combinations = []
        for scene_index, scene in enumerate(scenes):
            shift = (scene_index + run_idx - 1) % len(parameter_sets)
            ordered = parameter_sets[shift:] + parameter_sets[:shift]
            combinations.extend((scene, params) for params in ordered)

        for scene, params in combinations:
            pool_factor, grid_h, grid_w, min_coverage = params
            stats_seed = args.base_seed + run_idx - 1
            base_cfg = args.configs_root / f"{scene}.yaml"
            stats_csv = build_stats_path(
                args.stats_dir,
                scene,
                run_idx,
                pool_factor,
                grid_h,
                grid_w,
                min_coverage,
            )
            staged_stats_csv = common.build_staging_timing_path(stats_csv)
            temp_cfg = override_config(
                base_cfg,
                run_idx,
                args.frame_count,
                args.warmup_frames,
                stats_csv if args.dry_run else staged_stats_csv,
                stats_seed,
                pool_factor,
                grid_h,
                grid_w,
                min_coverage,
            )

            tag = parameter_tag(pool_factor, grid_h, grid_w, min_coverage)
            print(
                f"--> Run={run_idx} Scene={scene} Params={tag}"
            )
            print(f"    Sample counts: {stats_csv}")
            if args.dry_run:
                if not args.keep_temp_configs:
                    temp_cfg.unlink(missing_ok=True)
                continue

            try:
                subprocess.run(
                    [
                        args.python_exec,
                        str(args.target_script),
                        "--config",
                        str(temp_cfg),
                    ],
                    check=True,
                )
                stats = validate_sample_stats(staged_stats_csv, scene)
                common.ensure_dir(stats_csv.parent)
                staged_stats_csv.replace(stats_csv)
                append_log_row(
                    args.log_file,
                    run_idx,
                    scene,
                    stats_seed,
                    pool_factor,
                    grid_h,
                    grid_w,
                    min_coverage,
                    args,
                    stats,
                    stats_csv,
                )
                successes += 1
                print(
                    f"    SUCCESS: frames={stats['spatial_frames']}, "
                    f"mean={stats['mean_uniform_samples']}, "
                    f"std={stats['stddev_uniform_samples']}, "
                    f"median={stats['median_uniform_samples']}, "
                    f"p95={stats['p95_uniform_samples']}"
                )
                if not args.keep_temp_configs:
                    temp_cfg.unlink(missing_ok=True)
            except subprocess.CalledProcessError as error:
                failures += 1
                print(f"    FAILED: Co-SLAM exited with {error.returncode}")
                print(f"    Temp config retained: {temp_cfg}")
            except (OSError, RuntimeError) as error:
                failures += 1
                print(f"    FAILED: output validation error: {error}")
                print(f"    Temp config retained: {temp_cfg}")
            except KeyboardInterrupt:
                print("\n    ABORTED: user interrupted.")
                temp_cfg.unlink(missing_ok=True)
                sys.exit(130)

    if args.dry_run:
        print("\nDry run complete.")
        return

    print(
        f"\nAll scheduled runs attempted: {successes} succeeded, "
        f"{failures} failed. Run log: {args.log_file}"
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
