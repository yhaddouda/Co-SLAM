#!/usr/bin/env python3
"""Run frame-total profiling with three fixed pruning/hash configurations."""

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

import auto_run_frame_total_modes as common


MODE_DEFINITIONS = {
    "baseline": {
        "tracking_naive_pruning": False,
        "ba_bucket": False,
        "hash": "CoherentPrime",
        "morton_sort": False,
    },
    "pruning": {
        "tracking_naive_pruning": True,
        "ba_bucket": True,
        "hash": "CoherentPrime",
        "morton_sort": False,
    },
    "pruningD": {
        "tracking_naive_pruning": True,
        "ba_bucket": True,
        "hash": "Morton",
        "morton_sort": True,
    },
}

DEFAULT_LOG_FILE = Path("output/frame_total_three_modes_execution_times.csv")
LOG_HEADER = (
    "Run,Dataset,Mode,Scene,TableSize,FrameCount,"
    "TimingCsv,Time_Seconds\n"
)


def size_label(size: int | None) -> str:
    return "DefaultSize" if size is None else f"Size{size}"


def build_exp_name(mode: str, size: int | None, run_idx: int) -> str:
    return f"{mode}_{size_label(size)}_Run{run_idx}"


def needs_timing_suffix(runs: int, sizes: list[int | None]) -> bool:
    return runs > 1 or len(sizes) > 1


def build_timing_csv_path(
    timing_dir: Path,
    dataset: str,
    scene: str,
    mode: str,
    size: int | None,
    run_idx: int,
    add_sweep_suffix: bool,
) -> Path:
    suffix = ""
    if add_sweep_suffix:
        suffix = f"_{size_label(size)}_Run{run_idx}"
    filename = f"{scene}_frame_total_timing_complete_{mode}{suffix}.csv"
    return timing_dir / dataset / filename


def override_config(
    base_cfg_path: Path,
    dataset: str,
    scene: str,
    mode: str,
    table_size: int | None,
    run_idx: int,
    frame_count: int | None,
    timing_csv: Path,
    timing_write_header: bool,
    disable_eval: bool,
) -> Path:
    cfg = common.read_yaml(base_cfg_path)
    mode_params = MODE_DEFINITIONS[mode]

    cfg.setdefault("grid", {})
    cfg["grid"]["hash"] = mode_params["hash"]
    cfg["grid"]["morton_sort"] = mode_params["morton_sort"]
    if table_size is not None:
        cfg["grid"]["hash_size"] = table_size

    cfg.setdefault("tracking_naive_pruning", {})
    cfg["tracking_naive_pruning"]["enabled"] = mode_params[
        "tracking_naive_pruning"
    ]
    cfg["tracking_naive_pruning"]["collect_stats"] = False

    cfg.setdefault("ba_bucket", {})
    cfg["ba_bucket"]["enabled"] = mode_params["ba_bucket"]

    # Disable the legacy pruning flag so pruning is controlled only by the
    # tracking_naive_pruning and ba_bucket presets above.
    cfg.setdefault("training", {})
    cfg["training"]["uniform_samples_until_depth"] = False

    cfg.setdefault("mapping", {})
    cfg["mapping"]["first_mesh"] = False

    cfg.setdefault("mesh", {})
    cfg["mesh"]["vis"] = 1_000_000_000
    cfg["mesh"]["visualisation"] = False

    cfg.setdefault("timing", {})
    cfg["timing"]["mode"] = "frame_total"
    cfg["timing"]["warmup_frames"] = 0
    cfg["timing"]["max_frames"] = frame_count
    cfg["timing"]["disable_eval"] = bool(disable_eval)
    cfg["timing"]["frame_total_output_csv"] = common.format_config_path(
        timing_csv
    )
    cfg["timing"]["frame_total_write_header"] = bool(timing_write_header)

    cfg.setdefault("profiling", {})
    cfg["profiling"]["enabled"] = False

    cfg.setdefault("data", {})
    cfg["data"]["exp_name"] = build_exp_name(mode, table_size, run_idx)
    default_output = (
        Path("output")
        / common.DATASET_DEFINITIONS[dataset]["output_dir"]
        / scene
    )
    base_output = Path(cfg["data"].get("output", default_output))
    cfg["data"]["output"] = str(base_output / f"run_{run_idx}")

    tmp_cfg_name = f"temp_{scene}_{cfg['data']['exp_name']}.yaml"
    tmp_cfg_path = base_cfg_path.parent / tmp_cfg_name
    common.write_yaml(tmp_cfg_path, cfg)
    return tmp_cfg_path


def prepare_log_file(log_file: Path) -> None:
    common.ensure_dir(log_file.parent)
    if not log_file.exists():
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(LOG_HEADER)
        return

    with open(log_file, "r", encoding="utf-8") as f:
        current_header = f.readline()
    if current_header and current_header != LOG_HEADER:
        raise RuntimeError(
            f"Existing log file '{log_file}' has header "
            f"'{current_header.strip()}'. Expected '{LOG_HEADER.strip()}'. "
            "Use --log-file with a fresh CSV path."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner for fixed baseline, pruning, and pruningD "
            "frame-total timing configurations."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(common.DATASET_DEFINITIONS),
        help="Datasets to run: Replica, Synthetic, and/or Tum.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help=(
            "Optional scene subset, e.g. office0 or "
            "Replica:office0 Tum:fr1_desk."
        ),
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--sizes", nargs="+", default=["null"], type=str)
    parser.add_argument(
        "--mode",
        nargs="+",
        default=list(MODE_DEFINITIONS),
        choices=list(MODE_DEFINITIONS),
    )
    parser.add_argument("--configs-root", default="configs", type=Path)
    parser.add_argument(
        "--frame-count",
        type=common.parse_frame_count_arg,
        default=None,
        metavar="N|all",
        help="Maximum frames per scene; defaults to all available frames.",
    )
    parser.add_argument(
        "--timing-dir",
        type=Path,
        default=Path("results/frame_total"),
    )
    parser.add_argument("--timing-write-header", action="store_true")
    parser.add_argument("--append-timing-csv", action="store_true")
    parser.add_argument("--enable-eval", action="store_true")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, type=Path)
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument("--target-script", default="coslam.py", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp-configs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        selected_datasets = list(
            dict.fromkeys(
                common.canonical_dataset_name(dataset)
                for dataset in args.datasets
            )
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if not args.target_script.exists():
        print(f"ERROR: target script not found: {args.target_script}")
        sys.exit(1)
    if not args.configs_root.exists():
        print(f"ERROR: configs root not found: {args.configs_root}")
        sys.exit(1)
    if args.runs <= 0:
        print("ERROR: --runs must be positive.")
        sys.exit(1)

    for dataset in selected_datasets:
        dataset_cfg_dir = common.config_dir(args.configs_root, dataset)
        if not dataset_cfg_dir.exists():
            print(f"ERROR: config directory not found: {dataset_cfg_dir}")
            sys.exit(1)

    try:
        scenes_by_dataset = common.select_scenes(
            args.configs_root,
            selected_datasets,
            args.scenes,
        )
        parsed_sizes = list(
            dict.fromkeys(common.parse_size_arg(size) for size in args.sizes)
        )
    except (ValueError, TypeError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    selected_modes = list(dict.fromkeys(args.mode))
    add_timing_suffix = needs_timing_suffix(args.runs, parsed_sizes)
    scene_entries = [
        (dataset, scene)
        for dataset, scenes in scenes_by_dataset.items()
        for scene in scenes
    ]
    total_runs = (
        len(scene_entries)
        * len(selected_modes)
        * len(parsed_sizes)
        * args.runs
    )
    frame_count_log = "All" if args.frame_count is None else str(args.frame_count)

    print(f"Total runs scheduled: {total_runs}")
    for dataset, scenes in scenes_by_dataset.items():
        print(f"{dataset} scenes: {', '.join(scenes)}")
    print(f"Modes: {', '.join(selected_modes)}")
    print(f"Frame count limit: {frame_count_log}")
    print("-" * 60)

    if not args.dry_run:
        try:
            prepare_log_file(args.log_file)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    for run_idx in range(1, args.runs + 1):
        print(f"\n=== RUN ITERATION {run_idx}/{args.runs} ===")
        combinations = itertools.product(
            scene_entries,
            selected_modes,
            parsed_sizes,
        )
        for (dataset, scene), mode, size in combinations:
            base_cfg = (
                common.config_dir(args.configs_root, dataset)
                / f"{scene}.yaml"
            )
            timing_csv = build_timing_csv_path(
                args.timing_dir,
                dataset,
                scene,
                mode,
                size,
                run_idx,
                add_timing_suffix,
            )
            common.ensure_dir(timing_csv.parent)
            tmp_cfg = override_config(
                base_cfg,
                dataset,
                scene,
                mode,
                size,
                run_idx,
                args.frame_count,
                timing_csv,
                args.timing_write_header,
                not args.enable_eval,
            )

            size_log = "Default" if size is None else str(size)
            print(
                f"--> [Run {run_idx}] Dataset={dataset} | Mode={mode} | "
                f"Scene={scene} | Size={size_log} | Timing={timing_csv}"
            )

            if args.dry_run:
                print(f"    DRY RUN: wrote temp config {tmp_cfg}")
                if not args.keep_temp_configs and tmp_cfg.exists():
                    tmp_cfg.unlink()
                continue

            if timing_csv.exists() and not args.append_timing_csv:
                timing_csv.unlink()

            start_time = time.time()
            try:
                subprocess.run(
                    [
                        args.python_exec,
                        str(args.target_script),
                        "--config",
                        str(tmp_cfg),
                    ],
                    check=True,
                )
                duration = time.time() - start_time
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"{run_idx},{dataset},{mode},{scene},{size_log},"
                        f"{frame_count_log},{timing_csv},{duration:.4f}\n"
                    )
                print(f"    SUCCESS: finished in {duration:.2f}s")
                if tmp_cfg.exists():
                    tmp_cfg.unlink()
            except subprocess.CalledProcessError as exc:
                print(
                    "    FAILED: execution error "
                    f"(exit code {exc.returncode})"
                )
                print(f"    Temp config left for inspection: {tmp_cfg}")
            except KeyboardInterrupt:
                print("\n    ABORTED: user interrupted.")
                if tmp_cfg.exists():
                    tmp_cfg.unlink()
                sys.exit(1)

    if args.dry_run:
        message = "Dry run complete."
        if args.keep_temp_configs:
            message += " Temp configs were left in place for inspection."
        else:
            message += " Temp configs were cleaned up."
        print(f"\n{message}")
    else:
        print(f"\nAll runs completed. Execution times logged to {args.log_file}")


if __name__ == "__main__":
    main()
