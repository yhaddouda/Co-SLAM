#!/usr/bin/env python3
"""Collect focused per-iteration CUDA timings for Replica setups A and D.

Each timing CSV contains one row per tracking or bundle-adjustment iteration
with only its forward, backward, and full-iteration CUDA-event durations.
"""

import argparse
import csv
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


SETUP_DEFINITIONS = {
    "A": {"hash": "CoherentPrime", "morton_sort": False},
    "D": {"hash": "Morton", "morton_sort": True},
}

DEFAULT_REPLICA_SCENES = [
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
]

DEFAULT_TIMING_DIR = Path("results/iteration_breakdown_coslam_RTX/Replica")
DEFAULT_LOG_FILE = Path("output/Replica_iteration_timing_runs_coslam_RTX.csv")
LOG_COLUMNS = (
    "Run",
    "Scene",
    "Setup",
    "Hash",
    "MortonSort",
    "HashSizeLog2",
    "FrameLimit",
    "WarmupFrames",
    "TimingCsv",
    "RunWallTimeSeconds",
)
TIMING_COLUMNS = (
    "frame_id",
    "iteration_type",
    "iteration_id",
    "forward_cuda_ms",
    "backward_cuda_ms",
    "full_iteration_cuda_ms",
)


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_size_arg(value: str) -> int | None:
    if value.lower() in {"null", "none", "default"}:
        return None
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError(
            "--sizes values must be positive integers or 'default'."
        )
    return size


def parse_frame_count_arg(value: str) -> int | None:
    if value.lower() in {"all", "null", "none", "default"}:
        return None
    frame_count = int(value)
    if frame_count <= 0:
        raise argparse.ArgumentTypeError(
            "--frame-count must be a positive integer or 'all'."
        )
    return frame_count


def normalize_scene_arg(value: str) -> str:
    scene_name = Path(value).name
    return Path(scene_name).stem if scene_name.endswith(".yaml") else scene_name


def size_label(size: int | None) -> str:
    return "DefaultSize" if size is None else f"Size{size}"


def build_exp_name(setup: str, size: int | None, run_idx: int) -> str:
    return f"iteration_breakdown_{setup}_{size_label(size)}_Run{run_idx}"


def build_timing_csv_path(
    timing_dir: Path,
    scene: str,
    setup: str,
    size: int | None,
    run_idx: int,
) -> Path:
    return timing_dir / (
        f"{scene}_iteration_breakdown_{setup}_{size_label(size)}"
        f"_Run{run_idx}.csv"
    )


def build_staging_timing_path(timing_csv: Path) -> Path:
    return timing_csv.with_name(
        f".{timing_csv.name}.{uuid.uuid4().hex}.partial"
    )


def format_config_path(path: Path) -> str:
    path_text = str(path)
    if not path.is_absolute() and not path_text.startswith("."):
        return f"./{path_text}"
    return path_text


def override_config(
    base_cfg_path: Path,
    scene: str,
    setup_key: str,
    table_size: int | None,
    run_idx: int,
    frame_count: int | None,
    warmup_frames: int,
    timing_csv: Path,
    disable_eval: bool,
) -> Path:
    cfg = read_yaml(base_cfg_path)
    setup = SETUP_DEFINITIONS[setup_key]

    cfg.setdefault("grid", {})
    cfg["grid"]["hash"] = setup["hash"]
    cfg["grid"]["morton_sort"] = setup["morton_sort"]
    if table_size is not None:
        cfg["grid"]["hash_size"] = table_size

    # The paper's A-vs-D experiment isolates hash-grid hashing and Morton
    # ordering. Current Replica defaults enable separate depth-pruning paths
    # whose local Morton flags override grid.morton_sort, so force plain
    # Co-SLAM here for a faithful A/D comparison.
    cfg.setdefault("tracking_bucket", {})
    cfg["tracking_bucket"]["enabled"] = False
    cfg.setdefault("tracking_naive_pruning", {})
    cfg["tracking_naive_pruning"]["enabled"] = False
    cfg["tracking_naive_pruning"]["collect_stats"] = False
    cfg.setdefault("ba_bucket", {})
    cfg["ba_bucket"]["enabled"] = False
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

    cfg.setdefault("timing", {})
    cfg["timing"]["mode"] = "iteration_breakdown"
    cfg["timing"]["warmup_frames"] = warmup_frames
    cfg["timing"]["max_frames"] = frame_count
    cfg["timing"]["disable_eval"] = bool(disable_eval)
    cfg["timing"]["iteration_output_csv"] = format_config_path(timing_csv)

    cfg.setdefault("data", {})
    cfg["data"]["exp_name"] = build_exp_name(
        setup_key,
        table_size,
        run_idx,
    )
    base_output = Path(
        cfg["data"].get("output", f"output/Replica/{scene}")
    )
    cfg["data"]["output"] = str(
        base_output / "iteration_breakdown" / f"run_{run_idx}"
    )

    temp_name = f"temp_{scene}_{cfg['data']['exp_name']}.yaml"
    temp_path = base_cfg_path.parent / temp_name
    write_yaml(temp_path, cfg)
    return temp_path


def discover_scenes(
    configs_root: Path,
    requested_scenes: list[str] | None,
) -> list[str]:
    if requested_scenes:
        scenes = [normalize_scene_arg(scene) for scene in requested_scenes]
    else:
        scenes = list(DEFAULT_REPLICA_SCENES)

    scenes = list(dict.fromkeys(scenes))
    missing = [
        str(configs_root / f"{scene}.yaml")
        for scene in scenes
        if not (configs_root / f"{scene}.yaml").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Replica scene config(s): " + ", ".join(missing)
        )
    return scenes


def prepare_log_file(log_file: Path) -> None:
    ensure_dir(log_file.parent)
    if not log_file.exists():
        with open(log_file, "w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(LOG_COLUMNS)
        return

    with open(log_file, "r", encoding="utf-8", newline="") as handle:
        current_header = next(csv.reader(handle), [])
    if current_header != list(LOG_COLUMNS):
        raise RuntimeError(
            f"Existing log file '{log_file}' has an incompatible header. "
            "Use --log-file with a fresh CSV path."
        )


def validate_timing_csv(timing_csv: Path) -> dict[str, int]:
    if not timing_csv.is_file():
        raise RuntimeError(f"timing CSV was not created: {timing_csv}")

    counts = {"tracking": 0, "bundle_adjustment": 0}
    expected_iteration_ids = {}
    with open(timing_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TIMING_COLUMNS:
            raise RuntimeError(
                f"timing CSV has an unexpected header: {timing_csv}"
            )

        for line_number, row in enumerate(reader, start=2):
            if None in row or any(
                row.get(column) in (None, "") for column in TIMING_COLUMNS
            ):
                raise RuntimeError(
                    f"timing CSV has an incomplete row at line "
                    f"{line_number}: {timing_csv}"
                )

            try:
                frame_id = int(row["frame_id"])
                iteration_id = int(row["iteration_id"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"timing CSV has a non-integer frame/iteration ID at "
                    f"line {line_number}: {timing_csv}"
                ) from error

            iteration_type = row["iteration_type"]
            if iteration_type not in counts:
                raise RuntimeError(
                    f"timing CSV has an unknown iteration type at line "
                    f"{line_number}: {timing_csv}"
                )
            if frame_id < 0 or iteration_id < 0:
                raise RuntimeError(
                    f"timing CSV has a negative frame/iteration ID at line "
                    f"{line_number}: {timing_csv}"
                )

            sequence_key = (frame_id, iteration_type)
            expected_id = expected_iteration_ids.get(sequence_key, 0)
            if iteration_id != expected_id:
                raise RuntimeError(
                    f"timing CSV iteration IDs are not contiguous at line "
                    f"{line_number}: expected {expected_id}, got "
                    f"{iteration_id} in {timing_csv}"
                )
            expected_iteration_ids[sequence_key] = expected_id + 1

            try:
                forward_ms = float(row["forward_cuda_ms"])
                backward_ms = float(row["backward_cuda_ms"])
                full_ms = float(row["full_iteration_cuda_ms"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"timing CSV has a non-numeric duration at line "
                    f"{line_number}: {timing_csv}"
                ) from error

            timings = (forward_ms, backward_ms, full_ms)
            if any(not math.isfinite(value) or value < 0 for value in timings):
                raise RuntimeError(
                    f"timing CSV has an invalid duration at line "
                    f"{line_number}: {timing_csv}"
                )
            if full_ms < forward_ms or full_ms < backward_ms:
                raise RuntimeError(
                    f"full iteration is shorter than a component at line "
                    f"{line_number}: {timing_csv}"
                )

            counts[iteration_type] += 1

    missing_types = [
        iteration_type
        for iteration_type, count in counts.items()
        if count == 0
    ]
    if missing_types:
        raise RuntimeError(
            "timing CSV is missing "
            + ", ".join(missing_types)
            + " rows; the frame limit and warmup must include at least one "
            + f"tracking and one BA invocation: {timing_csv}"
        )
    return counts


def append_log_row(
    log_file: Path,
    run_idx: int,
    scene: str,
    setup_key: str,
    size: int | None,
    frame_count: int | None,
    warmup_frames: int,
    timing_csv: Path,
    duration: float,
) -> None:
    setup = SETUP_DEFINITIONS[setup_key]
    with open(log_file, "a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow([
            run_idx,
            scene,
            setup_key,
            setup["hash"],
            setup["morton_sort"],
            "Default" if size is None else size,
            "All" if frame_count is None else frame_count,
            warmup_frames,
            timing_csv,
            f"{duration:.4f}",
        ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all Replica scenes and collect per-iteration forward, "
            "backward, and full-iteration CUDA timings for setups A and D."
        )
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repetitions of every scene/setup/size combination.",
    )
    parser.add_argument(
        "--setups",
        nargs="+",
        default=["A", "D"],
        choices=tuple(SETUP_DEFINITIONS),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=[19],
        type=parse_size_arg,
        metavar="N|default",
        help="Hash-table log2 sizes. Defaults to 19.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional Replica scene subset; defaults to all eight scenes.",
    )
    parser.add_argument(
        "--configs-root",
        default=Path("configs/Replica"),
        type=Path,
    )
    parser.add_argument(
        "--frame-count",
        type=parse_frame_count_arg,
        default=None,
        metavar="N|all",
        help=(
            "Maximum frames per scene; defaults to all available frames. "
            "Use at least 6 to include a bundle-adjustment invocation."
        ),
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Discard timings whose frame ID is below this value.",
    )
    parser.add_argument(
        "--timing-dir",
        type=Path,
        default=DEFAULT_TIMING_DIR,
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
    )
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument(
        "--target-script",
        type=Path,
        default=Path("coslam.py"),
    )
    parser.add_argument(
        "--enable-eval",
        action="store_true",
        help="Enable mesh and pose evaluation after timing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and report configs without launching Co-SLAM.",
    )
    parser.add_argument(
        "--keep-temp-configs",
        action="store_true",
        help="Keep generated YAML files after dry or successful runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if yaml is None:
        print(
            "ERROR: PyYAML is required. Install it with: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.runs <= 0:
        print("ERROR: --runs must be positive.", file=sys.stderr)
        sys.exit(2)
    if args.warmup_frames < 0:
        print(
            "ERROR: --warmup-frames must be non-negative.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not args.target_script.is_file():
        print(
            f"ERROR: target script not found: {args.target_script}",
            file=sys.stderr,
        )
        sys.exit(2)
    if not args.configs_root.is_dir():
        print(
            f"ERROR: configs root not found: {args.configs_root}",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        scenes = discover_scenes(args.configs_root, args.scenes)
    except FileNotFoundError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)

    setups = list(dict.fromkeys(args.setups))
    sizes = list(dict.fromkeys(args.sizes))
    frame_count_label = (
        "All" if args.frame_count is None else str(args.frame_count)
    )
    total_runs = len(scenes) * len(setups) * len(sizes) * args.runs

    print(f"Total runs scheduled: {total_runs}")
    print(f"Scenes: {', '.join(scenes)}")
    print(f"Setups: {', '.join(setups)}")
    print(
        "Hash sizes: "
        + ", ".join(
            "Default" if size is None else str(size) for size in sizes
        )
    )
    print(f"Frame count limit: {frame_count_label}")
    print(f"Warmup frames: {args.warmup_frames}")
    print("-" * 60)

    if not args.dry_run:
        try:
            prepare_log_file(args.log_file)
        except RuntimeError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            sys.exit(2)

    success_count = 0
    failure_count = 0
    for run_idx in range(1, args.runs + 1):
        print(f"\n=== RUN ITERATION {run_idx}/{args.runs} ===")
        combinations = []
        for scene_index, scene in enumerate(scenes):
            # Balance thermal/order effects: A and D stay adjacent for a
            # given scene/size, and which setup runs first alternates across
            # both scenes and repetitions.
            if (scene_index + run_idx - 1) % 2 == 0:
                ordered_setups = setups
            else:
                ordered_setups = list(reversed(setups))
            for size in sizes:
                combinations.extend(
                    (scene, setup_key, size)
                    for setup_key in ordered_setups
                )

        for scene, setup_key, size in combinations:
            base_cfg = args.configs_root / f"{scene}.yaml"
            timing_csv = build_timing_csv_path(
                args.timing_dir,
                scene,
                setup_key,
                size,
                run_idx,
            )
            ensure_dir(timing_csv.parent)
            timing_output_path = (
                timing_csv
                if args.dry_run
                else build_staging_timing_path(timing_csv)
            )
            temp_cfg = override_config(
                base_cfg,
                scene,
                setup_key,
                size,
                run_idx,
                args.frame_count,
                args.warmup_frames,
                timing_output_path,
                not args.enable_eval,
            )

            size_text = "Default" if size is None else str(size)
            print(
                f"--> [Run {run_idx}] Scene={scene} | "
                f"Setup={setup_key} | Size={size_text} | "
                f"Timing={timing_csv}"
            )

            if args.dry_run:
                print(f"    DRY RUN: wrote temp config {temp_cfg}")
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
                timing_counts = validate_timing_csv(timing_output_path)
                timing_output_path.replace(timing_csv)
                duration = time.time() - start_time
                append_log_row(
                    args.log_file,
                    run_idx,
                    scene,
                    setup_key,
                    size,
                    args.frame_count,
                    args.warmup_frames,
                    timing_csv,
                    duration,
                )
                success_count += 1
                print(
                    f"    SUCCESS: finished in {duration:.2f}s | "
                    f"tracking iterations={timing_counts['tracking']} | "
                    "bundle-adjustment iterations="
                    f"{timing_counts['bundle_adjustment']}"
                )
                if not args.keep_temp_configs:
                    temp_cfg.unlink(missing_ok=True)
            except subprocess.CalledProcessError as error:
                failure_count += 1
                print(
                    "    FAILED: execution error "
                    f"(exit code {error.returncode})"
                )
                print(
                    f"    Temp config left for inspection: {temp_cfg}"
                )
                if timing_output_path.exists():
                    print(
                        "    Partial timing data left for inspection: "
                        f"{timing_output_path}"
                    )
            except RuntimeError as error:
                failure_count += 1
                print(f"    FAILED: timing validation error: {error}")
                print(
                    f"    Temp config left for inspection: {temp_cfg}"
                )
                if timing_output_path.exists():
                    print(
                        "    Partial timing data left for inspection: "
                        f"{timing_output_path}"
                    )
            except KeyboardInterrupt:
                print("\n    ABORTED: user interrupted.")
                temp_cfg.unlink(missing_ok=True)
                sys.exit(130)

    if args.dry_run:
        print("\nDry run complete.")
    else:
        print(
            f"\nAll scheduled runs attempted: {success_count} succeeded, "
            f"{failure_count} failed. Successful runs are logged in "
            f"{args.log_file}"
        )
        if failure_count:
            sys.exit(1)


if __name__ == "__main__":
    main()
