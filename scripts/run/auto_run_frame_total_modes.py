#!/usr/bin/env python3
import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

import yaml


SETUP_DEFINITIONS = {
    "A": {"hash": "CoherentPrime", "morton_sort": False},
    "B": {"hash": "Morton", "morton_sort": False},
    "C": {"hash": "CoherentPrime", "morton_sort": True},
    "D": {"hash": "Morton", "morton_sort": True},
}

MODE_DEFINITIONS = {
    "baseline": {
        "tracking_naive_pruning": False,
        "ba_bucket": False,
    },
    "pruning": {
        "tracking_naive_pruning": True,
        "ba_bucket": True,
    },
}

DATASET_DEFINITIONS = {
    "Replica": {
        "config_dir": "Replica",
        "base_config": "replica",
        "output_dir": "Replica",
    },
    "Synthetic": {
        "config_dir": "Synthetic",
        "base_config": "synthetic",
        "output_dir": "Synthetic",
    },
    "Tum": {
        "config_dir": "Tum",
        "base_config": "tum",
        "output_dir": "TUM",
    },
}

DEFAULT_LOG_FILE = Path("output/frame_total_modes_execution_times.csv")
LOG_HEADER = (
    "Run,Dataset,Mode,Scene,Setup,TableSize,FrameCount,"
    "TimingCsv,Time_Seconds\n"
)


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_size_arg(value):
    if str(value).lower() in {"null", "none", "default"}:
        return None
    return int(value)


def parse_frame_count_arg(value):
    if str(value).lower() in {"all", "null", "none", "default"}:
        return None
    frame_count = int(value)
    if frame_count <= 0:
        raise argparse.ArgumentTypeError(
            "--frame-count must be a positive integer or 'all'."
        )
    return frame_count


def canonical_dataset_name(value: str) -> str:
    normalized = value.strip().lower()
    for dataset_name in DATASET_DEFINITIONS:
        if dataset_name.lower() == normalized:
            return dataset_name
    choices = ", ".join(DATASET_DEFINITIONS)
    raise ValueError(f"Unknown dataset '{value}'. Expected one of: {choices}.")


def normalize_scene_name(value: str) -> str:
    scene_name = Path(value).name
    return Path(scene_name).stem if scene_name.endswith(".yaml") else scene_name


def config_dir(configs_root: Path, dataset: str) -> Path:
    return configs_root / DATASET_DEFINITIONS[dataset]["config_dir"]


def discover_dataset_scenes(configs_root: Path, dataset: str) -> list[str]:
    dataset_cfg_dir = config_dir(configs_root, dataset)
    base_config = DATASET_DEFINITIONS[dataset]["base_config"]
    return [
        config_path.stem
        for config_path in sorted(dataset_cfg_dir.glob("*.yaml"))
        if not config_path.name.startswith("temp_")
        and config_path.stem != base_config
    ]


def parse_qualified_scene(value: str) -> tuple[str | None, str]:
    for separator in (":", "/"):
        if separator not in value:
            continue
        dataset_part, scene_part = value.split(separator, 1)
        try:
            dataset = canonical_dataset_name(dataset_part)
        except ValueError:
            continue
        return dataset, normalize_scene_name(scene_part)

    path = Path(value)
    if path.parent.name:
        try:
            dataset = canonical_dataset_name(path.parent.name)
            return dataset, normalize_scene_name(path.name)
        except ValueError:
            pass

    return None, normalize_scene_name(value)


def select_scenes(
    configs_root: Path,
    datasets: list[str],
    requested_scenes: list[str] | None,
) -> dict[str, list[str]]:
    available = {
        dataset: discover_dataset_scenes(configs_root, dataset)
        for dataset in datasets
    }
    if not requested_scenes:
        return available

    selected = {dataset: [] for dataset in datasets}
    for scene_arg in requested_scenes:
        qualified_dataset, scene = parse_qualified_scene(scene_arg)
        if qualified_dataset is not None:
            if qualified_dataset not in datasets:
                raise ValueError(
                    f"Scene '{scene_arg}' belongs to {qualified_dataset}, "
                    "which is not listed in --datasets."
                )
            if scene not in available[qualified_dataset]:
                raise ValueError(
                    f"Unknown {qualified_dataset} scene '{scene}'. Available scenes: "
                    f"{', '.join(available[qualified_dataset])}."
                )
            matches = [qualified_dataset]
        else:
            matches = [
                dataset
                for dataset in datasets
                if scene in available[dataset]
            ]
            if not matches:
                raise ValueError(
                    f"Scene '{scene}' was not found in the selected datasets."
                )
            if len(matches) > 1:
                choices = ", ".join(f"{dataset}:{scene}" for dataset in matches)
                raise ValueError(
                    f"Scene '{scene}' is ambiguous. Use one of: {choices}."
                )

        dataset = matches[0]
        if scene not in selected[dataset]:
            selected[dataset].append(scene)

    return {
        dataset: scenes
        for dataset, scenes in selected.items()
        if scenes
    }


def size_label(size: int | None) -> str:
    return "DefaultSize" if size is None else f"Size{size}"


def build_exp_name(mode: str, setup: str, size: int | None, run_idx: int) -> str:
    return f"{mode}_{setup}_{size_label(size)}_Run{run_idx}"


def needs_timing_suffix(
    runs: int,
    setups: list[str],
    sizes: list[int | None],
) -> bool:
    return runs > 1 or len(setups) > 1 or len(sizes) > 1


def build_timing_csv_path(
    timing_dir: Path,
    dataset: str,
    scene: str,
    mode: str,
    setup: str,
    size: int | None,
    run_idx: int,
    add_sweep_suffix: bool,
) -> Path:
    suffix = ""
    if add_sweep_suffix:
        suffix = f"_{setup}_{size_label(size)}_Run{run_idx}"
    filename = f"{scene}_frame_total_timing_complete_{mode}{suffix}.csv"
    return timing_dir / dataset / filename


def format_config_path(path: Path) -> str:
    path_str = str(path)
    if not path.is_absolute() and not path_str.startswith("."):
        return f"./{path_str}"
    return path_str


def override_config(
    base_cfg_path: Path,
    dataset: str,
    scene: str,
    mode: str,
    setup_key: str,
    table_size: int | None,
    run_idx: int,
    frame_count: int | None,
    timing_csv: Path,
    timing_write_header: bool,
    disable_eval: bool,
) -> Path:
    cfg = read_yaml(base_cfg_path)

    cfg.setdefault("grid", {})
    setup_params = SETUP_DEFINITIONS[setup_key]
    cfg["grid"]["hash"] = setup_params["hash"]
    cfg["grid"]["morton_sort"] = setup_params["morton_sort"]
    if table_size is not None:
        cfg["grid"]["hash_size"] = table_size

    cfg.setdefault("tracking_naive_pruning", {})
    cfg["tracking_naive_pruning"]["enabled"] = MODE_DEFINITIONS[mode][
        "tracking_naive_pruning"
    ]
    cfg["tracking_naive_pruning"]["collect_stats"] = False

    cfg.setdefault("ba_bucket", {})
    cfg["ba_bucket"]["enabled"] = MODE_DEFINITIONS[mode]["ba_bucket"]

    # This legacy flag is still honored by coslam.py. Keep it disabled so the
    # baseline is fully dense and pruning is controlled by the new mode flags.
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
    cfg["timing"]["frame_total_output_csv"] = format_config_path(timing_csv)
    cfg["timing"]["frame_total_write_header"] = bool(timing_write_header)

    cfg.setdefault("profiling", {})
    cfg["profiling"]["enabled"] = False

    cfg.setdefault("data", {})
    cfg["data"]["exp_name"] = build_exp_name(
        mode,
        setup_key,
        table_size,
        run_idx,
    )
    default_output = (
        Path("output")
        / DATASET_DEFINITIONS[dataset]["output_dir"]
        / scene
    )
    base_output = Path(cfg["data"].get("output", default_output))
    cfg["data"]["output"] = str(base_output / f"run_{run_idx}")

    tmp_cfg_name = f"temp_{scene}_{cfg['data']['exp_name']}.yaml"
    tmp_cfg_path = base_cfg_path.parent / tmp_cfg_name
    write_yaml(tmp_cfg_path, cfg)
    return tmp_cfg_path


def prepare_log_file(log_file: Path) -> None:
    ensure_dir(log_file.parent)
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch runner for frame-total baseline/pruning timing across "
            "Replica, Synthetic, and TUM scenes."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_DEFINITIONS),
        help="Datasets to run: Replica, Synthetic, and/or Tum.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help=(
            "Optional scene subset. Accepts bare names such as office0 or "
            "qualified names such as Tum:fr1_desk."
        ),
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--setups",
        nargs="+",
        default=["A"],
        choices=sorted(SETUP_DEFINITIONS),
    )
    parser.add_argument("--sizes", nargs="+", default=["null"], type=str)
    parser.add_argument(
        "--mode",
        nargs="+",
        default=["baseline", "pruning"],
        choices=sorted(MODE_DEFINITIONS),
    )
    parser.add_argument("--configs-root", default="configs", type=Path)
    parser.add_argument(
        "--frame-count",
        type=parse_frame_count_arg,
        default=None,
        metavar="N|all",
        help=(
            "Optional maximum number of frames per scene. "
            "Defaults to all available frames."
        ),
    )
    parser.add_argument(
        "--timing-dir",
        type=Path,
        default=Path("results/frame_total_A_D_MAXN"),
    )
    parser.add_argument("--timing-write-header", action="store_true")
    parser.add_argument("--append-timing-csv", action="store_true")
    parser.add_argument("--enable-eval", action="store_true")
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, type=Path)
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument("--target-script", default="coslam.py", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-temp-configs", action="store_true")

    args = parser.parse_args()

    try:
        selected_datasets = list(
            dict.fromkeys(
                canonical_dataset_name(dataset)
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
        dataset_cfg_dir = config_dir(args.configs_root, dataset)
        if not dataset_cfg_dir.exists():
            print(f"ERROR: config directory not found: {dataset_cfg_dir}")
            sys.exit(1)

    try:
        scenes_by_dataset = select_scenes(
            args.configs_root,
            selected_datasets,
            args.scenes,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    selected_modes = list(dict.fromkeys(args.mode))
    selected_setups = list(dict.fromkeys(args.setups))
    parsed_sizes = list(
        dict.fromkeys(parse_size_arg(size) for size in args.sizes)
    )
    add_timing_suffix = needs_timing_suffix(
        args.runs,
        selected_setups,
        parsed_sizes,
    )
    scene_count = sum(len(scenes) for scenes in scenes_by_dataset.values())
    total_runs = (
        scene_count
        * len(selected_modes)
        * len(selected_setups)
        * len(parsed_sizes)
        * args.runs
    )

    print(f"Total runs scheduled: {total_runs}")
    for dataset, scenes in scenes_by_dataset.items():
        print(f"{dataset} scenes: {', '.join(scenes)}")
    print(f"Modes: {', '.join(selected_modes)}")
    print(f"Setups: {', '.join(selected_setups)}")
    frame_count_log = "All" if args.frame_count is None else str(args.frame_count)
    print(f"Frame count limit: {frame_count_log}")
    if add_timing_suffix:
        print(
            "Timing CSV names include setup/size/run suffixes because "
            "a sweep was requested."
        )
    print("-" * 60)

    if not args.dry_run:
        try:
            prepare_log_file(args.log_file)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)

    scene_entries = [
        (dataset, scene)
        for dataset, scenes in scenes_by_dataset.items()
        for scene in scenes
    ]

    for run_idx in range(1, args.runs + 1):
        print(f"\n=== RUN ITERATION {run_idx}/{args.runs} ===")
        combinations = itertools.product(
            scene_entries,
            selected_modes,
            selected_setups,
            parsed_sizes,
        )

        for (dataset, scene), mode, setup, size in combinations:
            dataset_cfg_dir = config_dir(args.configs_root, dataset)
            base_cfg = dataset_cfg_dir / f"{scene}.yaml"
            timing_csv = build_timing_csv_path(
                args.timing_dir,
                dataset,
                scene,
                mode,
                setup,
                size,
                run_idx,
                add_timing_suffix,
            )
            ensure_dir(timing_csv.parent)

            tmp_cfg = override_config(
                base_cfg_path=base_cfg,
                dataset=dataset,
                scene=scene,
                mode=mode,
                setup_key=setup,
                table_size=size,
                run_idx=run_idx,
                frame_count=args.frame_count,
                timing_csv=timing_csv,
                timing_write_header=args.timing_write_header,
                disable_eval=not args.enable_eval,
            )

            size_log = "Default" if size is None else str(size)
            print(
                f"--> [Run {run_idx}] Dataset={dataset} | Mode={mode} | "
                f"Scene={scene} | Setup={setup} | Size={size_log} | "
                f"Timing={timing_csv}"
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
                cmd = [
                    args.python_exec,
                    str(args.target_script),
                    "--config",
                    str(tmp_cfg),
                ]
                subprocess.run(cmd, check=True)

                duration = time.time() - start_time
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write(
                        f"{run_idx},{dataset},{mode},{scene},{setup},"
                        f"{size_log},{frame_count_log},{timing_csv},"
                        f"{duration:.4f}\n"
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
        if args.keep_temp_configs:
            print(
                "\nDry run complete. Temp configs were left in place "
                "for inspection."
            )
        else:
            print("\nDry run complete. Temp configs were cleaned up.")
    else:
        print(
            "\nAll runs completed. Execution times logged to "
            f"{args.log_file}"
        )


if __name__ == "__main__":
    main()
