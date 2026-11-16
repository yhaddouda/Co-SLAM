#!/usr/bin/env python3
"""Compare three Replica sampling/hash modes with one timing run per mode.

Each run writes:
  * an iteration CSV with tracking/BA forward, backward, and complete-iteration
    CUDA times;
  * a frame CSV derived from those rows by summing complete iterations, so the
    tracking and BA sampler preprocessing enclosed by the first iteration is
    included in the corresponding frame total;
  * when enabled in Replica config, one lightweight spatial-fallback summary.
"""

import argparse
import csv
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import auto_run_replica_iteration_timings as common


MODE_DEFINITIONS = {
    "baseline": {
        "tracking_depth_spatial": False,
        "ba_bucket": False,
        "hash": "CoherentPrime",
        "morton_sort": False,
    },
    "depth_spatial": {
        "tracking_depth_spatial": True,
        "ba_bucket": True,
        "hash": "CoherentPrime",
        "morton_sort": False,
    },
    "depth_spatial_morton": {
        "tracking_depth_spatial": True,
        "ba_bucket": True,
        "hash": "Morton",
        "morton_sort": True,
    },
}

DEFAULT_TIMING_DIR = Path("results/depth_spatial_three_modes_15W_run2/Replica")
DEFAULT_LOG_FILE = Path("output/Replica_depth_spatial_three_modes_runs_15W_run2.csv")
DEFAULT_FALLBACK_LOG_FILE = Path(
    "output/Replica_depth_spatial_fallback_stats_15W_run2.csv"
)
FRAME_COLUMNS = (
    "frame_id",
    "total_iter_profile_ms",
    "tr_iter_profile_total_ms",
    "ba_iter_profile_total_ms",
)
FALLBACK_COLUMNS = (
    "scene",
    "spatial_frames",
    "fallback_frames",
    "fallback_percent",
    "full_uniform_fallback_frames",
    "full_uniform_fallback_percent",
    "uniform_samples",
)
FALLBACK_LOG_COLUMNS = (
    "Run",
    "Mode",
    "Scene",
    "HashSizeLog2",
    "SpatialFrames",
    "FallbackFrames",
    "FallbackPercent",
    "FullUniformFallbackFrames",
    "FullUniformFallbackPercent",
    "UniformSamples",
    "FallbackStatsCsv",
)
LOG_COLUMNS = (
    "Run",
    "Mode",
    "Scene",
    "TrackingDepthSpatial",
    "BABucket",
    "Hash",
    "MortonSort",
    "HashSizeLog2",
    "PoolFactor",
    "GridH",
    "GridW",
    "MinCoverage",
    "FrameLimit",
    "WarmupFrames",
    "IterationTimingCsv",
    "FrameTimingCsv",
    "RunWallTimeSeconds",
)


def size_label(size: int | None) -> str:
    return "DefaultSize" if size is None else f"Size{size}"


def build_run_stem(
    scene: str,
    mode: str,
    size: int | None,
    run_idx: int,
) -> str:
    return f"{scene}_{mode}_{size_label(size)}_Run{run_idx}"


def build_timing_paths(
    timing_dir: Path,
    scene: str,
    mode: str,
    size: int | None,
    run_idx: int,
) -> tuple[Path, Path]:
    stem = build_run_stem(scene, mode, size, run_idx)
    return (
        timing_dir / "iterations" / f"{stem}_iterations.csv",
        timing_dir / "frames" / f"{stem}_frames.csv",
    )


def build_fallback_stats_path(
    timing_dir: Path,
    scene: str,
    mode: str,
    size: int | None,
    run_idx: int,
) -> Path:
    stem = build_run_stem(scene, mode, size, run_idx)
    return timing_dir / "fallbacks" / f"{stem}_fallbacks.csv"


def build_exp_name(mode: str, size: int | None, run_idx: int) -> str:
    return f"depth_spatial_timing_{mode}_{size_label(size)}_Run{run_idx}"


def override_config(
    base_cfg_path: Path,
    mode: str,
    table_size: int | None,
    run_idx: int,
    frame_count: int | None,
    warmup_frames: int,
    iteration_timing_csv: Path,
    fallback_stats_csv: Path,
    disable_eval: bool,
    pool_factor: int,
    grid_h: int,
    grid_w: int,
    min_coverage: float,
) -> Path:
    cfg = common.read_yaml(base_cfg_path)
    mode_cfg = MODE_DEFINITIONS[mode]

    cfg.setdefault("grid", {})
    cfg["grid"]["hash"] = mode_cfg["hash"]
    cfg["grid"]["morton_sort"] = mode_cfg["morton_sort"]
    if table_size is not None:
        cfg["grid"]["hash_size"] = table_size

    cfg.setdefault("tracking_depth_spatial", {})
    cfg["tracking_depth_spatial"].update({
        "enabled": mode_cfg["tracking_depth_spatial"],
        "pool_factor": pool_factor,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "min_coverage": min_coverage,
        "fallback_stats_csv": common.format_config_path(fallback_stats_csv),
    })

    cfg.setdefault("ba_bucket", {})
    cfg["ba_bucket"]["enabled"] = mode_cfg["ba_bucket"]
    # Replica's inherited base config pins this local override to False, so it
    # must be overridden explicitly for the global Morton mode to reach BA.
    cfg["ba_bucket"]["morton_inside_bucket"] = mode_cfg["morton_sort"]

    # Disable every competing/legacy tracking sampler explicitly.
    cfg.setdefault("tracking_bucket", {})
    cfg["tracking_bucket"]["enabled"] = False
    cfg.setdefault("tracking_naive_pruning", {})
    cfg["tracking_naive_pruning"]["enabled"] = False
    cfg["tracking_naive_pruning"]["collect_stats"] = False
    cfg.setdefault("training", {})
    cfg["training"]["uniform_samples_until_depth"] = False

    cfg.setdefault("mapping", {})
    cfg["mapping"]["first_mesh"] = False
    cfg.setdefault("mesh", {})
    cfg["mesh"]["vis"] = 1_000_000_000
    cfg["mesh"]["visualisation"] = False
    cfg.setdefault("profiling", {})
    cfg["profiling"]["enabled"] = False
    cfg["sync_for_ncu"] = False

    # One iteration-breakdown run contains all component and full-iteration
    # measurements. Frame totals are derived afterward from the same samples.
    cfg.setdefault("timing", {})
    cfg["timing"]["mode"] = "iteration_breakdown"
    cfg["timing"]["warmup_frames"] = warmup_frames
    cfg["timing"]["max_frames"] = frame_count
    cfg["timing"]["disable_eval"] = bool(disable_eval)
    cfg["timing"]["iteration_output_csv"] = common.format_config_path(
        iteration_timing_csv
    )

    cfg.setdefault("data", {})
    cfg["data"]["exp_name"] = build_exp_name(mode, table_size, run_idx)
    base_output = Path(
        cfg["data"].get("output", f"output/Replica/{base_cfg_path.stem}")
    )
    cfg["data"]["output"] = str(
        base_output / "depth_spatial_three_modes" / f"run_{run_idx}"
    )

    temp_path = base_cfg_path.parent / (
        f"temp_{base_cfg_path.stem}_{cfg['data']['exp_name']}.yaml"
    )
    common.write_yaml(temp_path, cfg)
    return temp_path


def derive_frame_totals(
    iteration_csv: Path,
    frame_csv: Path,
) -> int:
    totals = defaultdict(lambda: {"tracking": 0.0, "bundle_adjustment": 0.0})
    with open(iteration_csv, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            totals[int(row["frame_id"])][row["iteration_type"]] += float(
                row["full_iteration_cuda_ms"]
            )

    if not totals:
        raise RuntimeError(f"cannot derive frame totals from empty CSV: {iteration_csv}")

    common.ensure_dir(frame_csv.parent)
    with open(frame_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FRAME_COLUMNS)
        for frame_id in sorted(totals):
            tracking_ms = totals[frame_id]["tracking"]
            ba_ms = totals[frame_id]["bundle_adjustment"]
            writer.writerow([
                frame_id,
                f"{tracking_ms + ba_ms:.6f}",
                f"{tracking_ms:.6f}",
                f"{ba_ms:.6f}",
            ])
    return len(totals)


def validate_frame_csv(frame_csv: Path, expected_rows: int) -> None:
    with open(frame_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FRAME_COLUMNS:
            raise RuntimeError(f"frame CSV has an unexpected header: {frame_csv}")
        rows = list(reader)

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"frame CSV row count mismatch: expected {expected_rows}, "
            f"found {len(rows)} in {frame_csv}"
        )
    for line_number, row in enumerate(rows, start=2):
        try:
            total = float(row["total_iter_profile_ms"])
            tracking = float(row["tr_iter_profile_total_ms"])
            ba = float(row["ba_iter_profile_total_ms"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"non-numeric frame timing at line {line_number}: {frame_csv}"
            ) from error
        if any(not math.isfinite(value) or value < 0 for value in (total, tracking, ba)):
            raise RuntimeError(
                f"invalid frame timing at line {line_number}: {frame_csv}"
            )
        if not math.isclose(total, tracking + ba, rel_tol=1e-6, abs_tol=2e-6):
            raise RuntimeError(
                f"inconsistent frame total at line {line_number}: {frame_csv}"
            )


def tracking_fallback_stats_enabled(config_path: Path) -> bool:
    """Resolve the lightweight stats flag through the YAML inheritance chain."""
    seen = set()
    current_path = config_path
    while current_path is not None:
        resolved = current_path.resolve()
        if resolved in seen:
            raise RuntimeError(f"configuration inheritance cycle at {current_path}")
        seen.add(resolved)
        cfg = common.read_yaml(current_path)
        spatial_cfg = cfg.get("tracking_depth_spatial", {})
        if "collect_fallback_stats" in spatial_cfg:
            return bool(spatial_cfg["collect_fallback_stats"])
        inherited = cfg.get("inherit_from")
        current_path = Path(inherited) if inherited is not None else None
    return False


def validate_fallback_stats_csv(stats_csv: Path, expected_scene: str) -> dict:
    if not stats_csv.is_file():
        raise RuntimeError(f"fallback summary CSV was not created: {stats_csv}")
    with open(stats_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FALLBACK_COLUMNS:
            raise RuntimeError(
                f"fallback summary CSV has an unexpected header: {stats_csv}"
            )
        rows = list(reader)
    if len(rows) != 1 or rows[0]["scene"] != expected_scene:
        raise RuntimeError(
            f"fallback summary CSV must contain one row for {expected_scene}: "
            f"{stats_csv}"
        )

    row = rows[0]
    try:
        spatial_frames = int(row["spatial_frames"])
        fallback_frames = int(row["fallback_frames"])
        full_uniform_frames = int(row["full_uniform_fallback_frames"])
        fallback_percent = float(row["fallback_percent"])
        full_uniform_percent = float(row["full_uniform_fallback_percent"])
        uniform_samples = int(row["uniform_samples"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid fallback summary values: {stats_csv}") from error

    if not (
        0 <= full_uniform_frames <= fallback_frames <= spatial_frames
        and uniform_samples > 0
    ):
        raise RuntimeError(f"inconsistent fallback summary counts: {stats_csv}")
    expected_fallback_percent = (
        100.0 * fallback_frames / spatial_frames if spatial_frames else 0.0
    )
    expected_full_percent = (
        100.0 * full_uniform_frames / spatial_frames if spatial_frames else 0.0
    )
    if not math.isclose(
        fallback_percent, expected_fallback_percent, rel_tol=0.0, abs_tol=1e-5
    ) or not math.isclose(
        full_uniform_percent, expected_full_percent, rel_tol=0.0, abs_tol=1e-5
    ):
        raise RuntimeError(f"inconsistent fallback summary percentages: {stats_csv}")
    return row


def prepare_fallback_log_file(log_file: Path) -> None:
    common.ensure_dir(log_file.parent)
    if not log_file.exists():
        with open(log_file, "w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(FALLBACK_LOG_COLUMNS)
        return
    with open(log_file, "r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), [])
    if header != list(FALLBACK_LOG_COLUMNS):
        raise RuntimeError(
            f"Existing fallback log '{log_file}' has an incompatible header."
        )


def append_fallback_log_row(
    log_file: Path,
    run_idx: int,
    mode: str,
    scene: str,
    size: int | None,
    stats: dict,
    stats_csv: Path,
) -> None:
    with open(log_file, "a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([
            run_idx,
            mode,
            scene,
            "Default" if size is None else size,
            stats["spatial_frames"],
            stats["fallback_frames"],
            stats["fallback_percent"],
            stats["full_uniform_fallback_frames"],
            stats["full_uniform_fallback_percent"],
            stats["uniform_samples"],
            stats_csv,
        ])


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
            f"Existing log file '{log_file}' has an incompatible header. "
            "Use --log-file with a fresh CSV path."
        )


def append_log_row(
    log_file: Path,
    run_idx: int,
    mode: str,
    scene: str,
    size: int | None,
    args: argparse.Namespace,
    iteration_csv: Path,
    frame_csv: Path,
    duration: float,
) -> None:
    mode_cfg = MODE_DEFINITIONS[mode]
    with open(log_file, "a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([
            run_idx,
            mode,
            scene,
            mode_cfg["tracking_depth_spatial"],
            mode_cfg["ba_bucket"],
            mode_cfg["hash"],
            mode_cfg["morton_sort"],
            "Default" if size is None else size,
            args.pool_factor,
            args.grid_h,
            args.grid_w,
            args.min_coverage,
            "All" if args.frame_count is None else args.frame_count,
            args.warmup_frames,
            iteration_csv,
            frame_csv,
            f"{duration:.4f}",
        ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all Replica scenes for baseline, depth-spatial+BA, and "
            "depth-spatial+BA+Morton modes, collecting iteration and frame timings."
        )
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--modes",
        "--mode",
        nargs="+",
        default=list(MODE_DEFINITIONS),
        choices=tuple(MODE_DEFINITIONS),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=[19],
        type=common.parse_size_arg,
        metavar="N|default",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional Replica subset; defaults to all eight scenes.",
    )
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=Path("configs/Replica"),
    )
    parser.add_argument(
        "--frame-count",
        type=common.parse_frame_count_arg,
        default=None,
        metavar="N|all",
        help="Maximum frames per scene; use at least 6 to include BA.",
    )
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--pool-factor", type=int, default=4)
    parser.add_argument("--grid-h", type=int, default=8)
    parser.add_argument("--grid-w", type=int, default=8)
    parser.add_argument("--min-coverage", type=float, default=0.75)
    parser.add_argument("--timing-dir", type=Path, default=DEFAULT_TIMING_DIR)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument(
        "--fallback-log-file",
        type=Path,
        default=DEFAULT_FALLBACK_LOG_FILE,
    )
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument("--target-script", type=Path, default=Path("coslam.py"))
    parser.add_argument("--enable-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp-configs", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if common.yaml is None:
        raise RuntimeError("PyYAML is required")
    if args.runs <= 0:
        raise ValueError("--runs must be positive")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative")
    if args.pool_factor < 1:
        raise ValueError("--pool-factor must be at least 1")
    if args.grid_h < 1 or args.grid_w < 1:
        raise ValueError("--grid-h and --grid-w must be positive")
    if not 0.0 <= args.min_coverage <= 1.0:
        raise ValueError("--min-coverage must be between 0 and 1")
    if not args.target_script.is_file():
        raise FileNotFoundError(f"target script not found: {args.target_script}")
    if not args.configs_root.is_dir():
        raise FileNotFoundError(f"configs root not found: {args.configs_root}")


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        scenes = common.discover_scenes(args.configs_root, args.scenes)
        modes = list(dict.fromkeys(args.modes))
        sizes = list(dict.fromkeys(args.sizes))
        fallback_stats_by_scene = {
            scene: tracking_fallback_stats_enabled(
                args.configs_root / f"{scene}.yaml"
            )
            for scene in scenes
        }
        if not args.dry_run:
            prepare_log_file(args.log_file)
            if any(fallback_stats_by_scene.values()):
                prepare_fallback_log_file(args.fallback_log_file)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)

    total_runs = len(scenes) * len(modes) * len(sizes) * args.runs
    print(f"Total runs scheduled: {total_runs}")
    print(f"Scenes: {', '.join(scenes)}")
    print(f"Modes: {', '.join(modes)}")
    print(
        "Hash sizes: "
        + ", ".join("Default" if size is None else str(size) for size in sizes)
    )
    print(
        f"Sampler: pool_factor={args.pool_factor}, "
        f"grid={args.grid_h}x{args.grid_w}, "
        f"min_coverage={args.min_coverage}"
    )
    print("-" * 72)

    successes = 0
    failures = 0
    for run_idx in range(1, args.runs + 1):
        print(f"\n=== RUN ITERATION {run_idx}/{args.runs} ===")
        combinations = []
        for scene_index, scene in enumerate(scenes):
            # Rotate mode order across scenes/runs to reduce thermal bias while
            # keeping all three variants adjacent for each scene and size.
            shift = (scene_index + run_idx - 1) % len(modes)
            ordered_modes = modes[shift:] + modes[:shift]
            for size in sizes:
                combinations.extend(
                    (scene, mode, size) for mode in ordered_modes
                )

        for scene, mode, size in combinations:
            base_cfg = args.configs_root / f"{scene}.yaml"
            iteration_csv, frame_csv = build_timing_paths(
                args.timing_dir,
                scene,
                mode,
                size,
                run_idx,
            )
            fallback_csv = build_fallback_stats_path(
                args.timing_dir,
                scene,
                mode,
                size,
                run_idx,
            )
            common.ensure_dir(iteration_csv.parent)
            common.ensure_dir(frame_csv.parent)
            staged_iteration_csv = common.build_staging_timing_path(iteration_csv)
            staged_frame_csv = common.build_staging_timing_path(frame_csv)
            staged_fallback_csv = common.build_staging_timing_path(fallback_csv)
            config_timing_path = (
                iteration_csv if args.dry_run else staged_iteration_csv
            )
            config_fallback_path = (
                fallback_csv if args.dry_run else staged_fallback_csv
            )
            temp_cfg = override_config(
                base_cfg,
                mode,
                size,
                run_idx,
                args.frame_count,
                args.warmup_frames,
                config_timing_path,
                config_fallback_path,
                not args.enable_eval,
                args.pool_factor,
                args.grid_h,
                args.grid_w,
                args.min_coverage,
            )

            size_text = "Default" if size is None else str(size)
            print(
                f"--> [Run {run_idx}] Scene={scene} | Mode={mode} | "
                f"Size={size_text}"
            )
            print(f"    Iterations: {iteration_csv}")
            print(f"    Frames:     {frame_csv}")
            collect_fallback_stats = bool(
                MODE_DEFINITIONS[mode]["tracking_depth_spatial"]
                and fallback_stats_by_scene[scene]
            )
            if collect_fallback_stats:
                print(f"    Fallbacks:  {fallback_csv}")

            if args.dry_run:
                print(f"    DRY RUN: wrote {temp_cfg}")
                if not args.keep_temp_configs:
                    temp_cfg.unlink(missing_ok=True)
                continue

            start_time = time.time()
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
                timing_counts = common.validate_timing_csv(staged_iteration_csv)
                frame_count = derive_frame_totals(
                    staged_iteration_csv,
                    staged_frame_csv,
                )
                validate_frame_csv(staged_frame_csv, frame_count)
                fallback_stats = None
                if collect_fallback_stats:
                    fallback_stats = validate_fallback_stats_csv(
                        staged_fallback_csv,
                        scene,
                    )
                staged_iteration_csv.replace(iteration_csv)
                staged_frame_csv.replace(frame_csv)
                if fallback_stats is not None:
                    common.ensure_dir(fallback_csv.parent)
                    staged_fallback_csv.replace(fallback_csv)
                    append_fallback_log_row(
                        args.fallback_log_file,
                        run_idx,
                        mode,
                        scene,
                        size,
                        fallback_stats,
                        fallback_csv,
                    )
                duration = time.time() - start_time
                append_log_row(
                    args.log_file,
                    run_idx,
                    mode,
                    scene,
                    size,
                    args,
                    iteration_csv,
                    frame_csv,
                    duration,
                )
                successes += 1
                print(
                    f"    SUCCESS: {duration:.2f}s | frames={frame_count} | "
                    f"tracking iterations={timing_counts['tracking']} | "
                    "BA iterations="
                    f"{timing_counts['bundle_adjustment']}"
                )
                if fallback_stats is not None:
                    print(
                        "    FALLBACKS: "
                        f"{fallback_stats['fallback_frames']}/"
                        f"{fallback_stats['spatial_frames']} "
                        f"({fallback_stats['fallback_percent']}%); full-"
                        f"{fallback_stats['uniform_samples']}="
                        f"{fallback_stats['full_uniform_fallback_frames']} "
                        f"({fallback_stats['full_uniform_fallback_percent']}%)"
                    )
                if not args.keep_temp_configs:
                    temp_cfg.unlink(missing_ok=True)
            except subprocess.CalledProcessError as error:
                failures += 1
                print(f"    FAILED: Co-SLAM exited with {error.returncode}")
                print(f"    Temp config retained: {temp_cfg}")
            except RuntimeError as error:
                failures += 1
                print(f"    FAILED: timing validation error: {error}")
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
