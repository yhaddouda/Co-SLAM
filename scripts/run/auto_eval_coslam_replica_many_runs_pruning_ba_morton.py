#!/usr/bin/env python3
import argparse
import csv
import itertools
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

# --- Constants ---
DEFAULT_DATASETS = ["Replica"]
DEFAULT_SIZES = list(range(13, 25))  # 13 to 24
DEFAULT_CONFIGS = ["A", "B", "C", "D", "E"]
DATASET_DEFINITIONS = {
    "Replica": {
        "scenes": [
            "office0",
            "office1",
            "office2",
            "office3",
            "office4",
            "room0",
            "room1",
            "room2",
        ],
        "configs_root": Path("configs/Replica"),
        "dataset_root": Path("../datasets/vv_data/Replica"),
        "output_root": Path("output/Replica"),
        "eval_dataset_type": "Replica",
        "scene_mapping": {},
    },
    "Synthetic": {
        "scenes": ["br", "ck", "gr", "gwr", "ma", "tg", "wr"],
        "configs_root": Path("configs/Synthetic"),
        "dataset_root": Path("../datasets/vv_data/neural_rgbd_data"),
        "output_root": Path("output/Synthetic"),
        "eval_dataset_type": "RGBD",
        "scene_mapping": {
            "br": "breakfast_room",
            "ck": "complete_kitchen",
            "gr": "green_room",
            "gwr": "grey_white_room",
            "ma": "morning_apartment",
            "tg": "thin_geometry",
            "wr": "whiteroom",
        },
    },
}
CONFIG_DEFINITIONS = {
    # A: original tracking and original BA sampling.
    "A": {
        "tracking_naive_pruning_enabled": False,
        "ba_bucket_enabled": False,
        "hash": "CoherentPrime",
        "morton_sort": False,
    },
    # B: naive tracking pruning only.
    "B": {
        "tracking_naive_pruning_enabled": True,
        "ba_bucket_enabled": False,
        "hash": "CoherentPrime",
        "morton_sort": False,
    },
    # C: bucketed BA only.
    "C": {
        "tracking_naive_pruning_enabled": False,
        "ba_bucket_enabled": True,
        "hash": "CoherentPrime",
        "morton_sort": False,
    },
    # D: naive tracking pruning + bucketed BA.
    "D": {
        "tracking_naive_pruning_enabled": True,
        "ba_bucket_enabled": True,
        "hash": "CoherentPrime",
        "morton_sort": False,
    },
    # E: config D with Morton hashing and Morton sorting.
    "E": {
        "tracking_naive_pruning_enabled": True,
        "ba_bucket_enabled": True,
        "hash": "Morton",
        "morton_sort": True,
    },
}


def normalize_dataset_name(value: str) -> str:
    for dataset_name in DATASET_DEFINITIONS:
        if value.lower() == dataset_name.lower():
            return dataset_name
    choices = ", ".join(DATASET_DEFINITIONS)
    raise argparse.ArgumentTypeError(f"Unknown dataset '{value}'. Choose from: {choices}")


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_metrics(output_str: str) -> dict:
    """Parses standard output from eval_recon_headless.py."""
    metrics = {"Accuracy": "N/A", "Completion": "N/A", "Completion_Ratio": "N/A", "Depth_L1": "N/A"}

    patterns = {
        "Accuracy": r"accuracy:\s+([0-9.]+)",
        "Completion": r"completion:\s+([0-9.]+)",
        "Completion_Ratio": r"completion ratio:\s+([0-9.]+)",
        "Depth_L1": r"Depth L1:\s+([0-9.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output_str)
        if match:
            metrics[key] = match.group(1)

    return metrics


def build_exp_name(
    config_name: str,
    hash_size: int,
    run_idx: Optional[int] = None,
) -> str:
    base_name = f"{config_name}_T{hash_size}"
    if run_idx is None:
        return base_name
    return f"{base_name}_Run{run_idx}"


def is_raw_mesh(path: Path) -> bool:
    return re.fullmatch(r"mesh_track\d+\.ply", path.name) is not None


def apply_config_mode(cfg: dict, config_name: str) -> None:
    mode_cfg = CONFIG_DEFINITIONS[config_name]
    cfg.setdefault("grid", {})
    cfg.setdefault("tracking_bucket", {})
    cfg.setdefault("tracking_naive_pruning", {})
    cfg.setdefault("ba_bucket", {})
    cfg.setdefault("training", {})

    cfg["grid"]["hash"] = mode_cfg["hash"]
    cfg["grid"]["morton_sort"] = mode_cfg["morton_sort"]
    cfg["tracking_bucket"]["enabled"] = False
    cfg["tracking_bucket"]["morton_inside_bucket"] = False
    cfg["training"]["uniform_samples_until_depth"] = False
    cfg["tracking_naive_pruning"]["enabled"] = mode_cfg["tracking_naive_pruning_enabled"]
    cfg["ba_bucket"]["enabled"] = mode_cfg["ba_bucket_enabled"]
    cfg["ba_bucket"]["morton_inside_bucket"] = False


def get_dataset_context(args, dataset_name: str) -> dict:
    definition = DATASET_DEFINITIONS[dataset_name]
    prefix = dataset_name.lower()
    configs_root = getattr(args, f"{prefix}_configs_root")
    dataset_root = getattr(args, f"{prefix}_dataset_root")

    if args.configs_root is not None:
        configs_root = args.configs_root
    if args.dataset_root is not None:
        dataset_root = args.dataset_root

    return {
        **definition,
        "configs_root": configs_root,
        "dataset_root": dataset_root,
    }


def resolve_scene_jobs(
    dataset_names: List[str],
    requested_scenes: Optional[List[str]],
) -> List[Tuple[str, str]]:
    if requested_scenes is None:
        return [
            (dataset_name, scene)
            for dataset_name in dataset_names
            for scene in DATASET_DEFINITIONS[dataset_name]["scenes"]
        ]

    jobs = []
    seen = set()
    for scene in requested_scenes:
        matching_datasets = [
            dataset_name
            for dataset_name in dataset_names
            if scene in DATASET_DEFINITIONS[dataset_name]["scenes"]
        ]
        if not matching_datasets:
            owners = [
                dataset_name
                for dataset_name, definition in DATASET_DEFINITIONS.items()
                if scene in definition["scenes"]
            ]
            if owners:
                raise SystemExit(
                    f"Scene '{scene}' belongs to {owners[0]}, but that dataset was not selected."
                )
            known_scenes = sorted(
                scene_name
                for definition in DATASET_DEFINITIONS.values()
                for scene_name in definition["scenes"]
            )
            raise SystemExit(
                f"Unknown scene '{scene}'. Known scene names: {', '.join(known_scenes)}"
            )

        job = (matching_datasets[0], scene)
        if job not in seen:
            jobs.append(job)
            seen.add(job)

    return jobs


def default_output_csv(dataset_names: List[str]) -> Path:
    if dataset_names == ["Replica"]:
        return Path("output/replica_tracking_ba_ablation_results.csv")
    slug = "_".join(dataset_name.lower() for dataset_name in dataset_names)
    return Path("output") / f"{slug}_tracking_ba_ablation_results.csv"


def build_csv_columns(
    use_run_column: bool,
    include_dataset_column: bool,
) -> List[str]:
    columns = []
    if use_run_column:
        columns.append("Run")
    if include_dataset_column:
        columns.append("Dataset")
    columns.extend(
        [
            "Scene",
            "Config",
            "HashSize",
            "Accuracy",
            "Completion",
            "CompletionRatio",
            "DepthL1",
        ]
    )
    return columns


def run_sequence(args, dataset_name, scene, config_name, hash_size, run_idx=None):
    dataset = get_dataset_context(args, dataset_name)

    # 1. Prepare Configuration
    base_cfg = dataset["configs_root"] / f"{scene}.yaml"
    if not base_cfg.exists():
        print(f"  [Error] Config not found: {base_cfg}")
        return None

    cfg = read_yaml(base_cfg)

    # Include config mode, and optionally run index, so repeated runs do not overwrite each other.
    exp_name = build_exp_name(config_name, hash_size, run_idx)

    # Ensure keys exist
    cfg.setdefault("grid", {})
    cfg.setdefault("data", {})
    cfg.setdefault("timing", {})

    # Set flags
    cfg["grid"]["hash_size"] = hash_size
    cfg["data"]["exp_name"] = exp_name
    cfg["timing"]["disable_eval"] = False
    apply_config_mode(cfg, config_name)
    base_output = cfg["data"].get("output", str(dataset["output_root"] / scene))
    cfg["data"]["output"] = str(Path(base_output) / args.output_prefix)
    if run_idx is not None:
        cfg["data"]["output"] = str(Path(cfg["data"]["output"]) / f"run_{run_idx}")

    # Save Temp Config
    tmp_config_path = (
        Path("temp_configs")
        / args.output_prefix
        / dataset_name.lower()
        / f"{scene}_{exp_name}.yaml"
    )
    ensure_dir(tmp_config_path.parent)
    write_yaml(tmp_config_path, cfg)

    # 2. Run CoSLAM
    if run_idx is None:
        print(
            f"  [1/3] Running CoSLAM "
            f"(Dataset={dataset_name}, Scene={scene}, Config={config_name}, Hash={hash_size})..."
        )
    else:
        print(
            f"  [1/3] Running CoSLAM "
            f"(Run={run_idx}, Dataset={dataset_name}, Scene={scene}, "
            f"Config={config_name}, Hash={hash_size})..."
        )
    log_path = (
        Path("output")
        / "logs"
        / args.output_prefix
        / dataset_name.lower()
        / f"{scene}_{exp_name}.log"
    )
    ensure_dir(log_path.parent)

    try:
        with open(log_path, "w") as f:
            subprocess.run(
                [args.python_exec, "coslam.py", "--config", str(tmp_config_path)],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except subprocess.CalledProcessError as exc:
        print(f"  [Error] SLAM crashed with exit code {exc.returncode}. See {log_path}.")
        return None

    # 3. Locate Mesh (Deterministic Path)
    # Path logic: <cfg[data][output]>/<exp_name>/mesh_track*.ply
    output_dir = Path(cfg["data"]["output"]) / exp_name
    meshes = sorted(
        (path for path in output_dir.glob("mesh_track*.ply") if is_raw_mesh(path)),
        key=os.path.getmtime,
    )

    if not meshes:
        print(f"  [Error] No mesh found in expected dir: {output_dir}")
        return None

    input_mesh = meshes[-1]  # Take the last one generated
    print(f"  > Found mesh: {input_mesh.name}")

    # 4. Cull Mesh
    print("  [2/3] Culling Mesh...")
    full_scene_name = dataset["scene_mapping"].get(scene, scene)
    virt_cam_path = dataset["dataset_root"] / full_scene_name / "virtual_cameras"

    # Cull output name is automatically: {input_stem}_cull_virt_cams.ply
    culled_mesh = input_mesh.parent / f"{input_mesh.stem}_cull_virt_cams.ply"

    cmd_cull = [
        args.python_exec,
        "neural_slam_eval/cull_mesh.py",
        "--config",
        str(tmp_config_path),
        "--input_mesh",
        str(input_mesh),
        "--remove_occlusion",
        "--virtual_cameras",
        "--virt_cam_path",
        str(virt_cam_path),
        "--gt_pose",
    ]

    try:
        subprocess.run(cmd_cull, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("  [Error] Culling failed.")
        return None

    # 5. Evaluate
    print("  [3/3] Evaluating...")
    gt_mesh = dataset["dataset_root"] / full_scene_name / "gt_mesh_cull_virt_cams.ply"

    cmd_eval = [
        args.python_exec,
        "neural_slam_eval/eval_recon_headless.py",
        "--rec_mesh",
        str(culled_mesh),
        "--gt_mesh",
        str(gt_mesh),
        "--dataset_type",
        dataset["eval_dataset_type"],
        "-2d",
        "-3d",
    ]

    try:
        result = subprocess.run(cmd_eval, check=True, capture_output=True, text=True)
        return parse_metrics(result.stdout)
    except subprocess.CalledProcessError:
        print("  [Error] Evaluation failed.")
        return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run tracking-pruning, BA-bucket, and Morton-grid ablations "
            "on Replica and Synthetic."
        ),
        epilog=(
            "Examples:\n"
            "  Run all Replica and Synthetic scenes:\n"
            "    --datasets Replica Synthetic\n"
            "  Run only selected scenes:\n"
            "    --datasets Replica Synthetic --scenes office0 room1 br tg"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=normalize_dataset_name,
        default=DEFAULT_DATASETS,
        choices=sorted(DATASET_DEFINITIONS),
        help="Dataset(s) to process. Without --scenes, every scene in each dataset is processed.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Optional scene-name filter. Replica uses office0...room2; Synthetic uses br/ck/gr/gwr/ma/tg/wr.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        choices=sorted(CONFIG_DEFINITIONS),
        help="Ablation config mode(s): A, B, C, D, E.",
    )
    parser.add_argument(
        "--sampling",
        nargs="+",
        default=None,
        choices=sorted(CONFIG_DEFINITIONS),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to repeat the entire experiment grid for averaging.",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES, help="Hash sizes")
    parser.add_argument(
        "--configs-root",
        type=Path,
        default=None,
        help="Legacy config-root override; valid only when one dataset is selected.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Legacy evaluation-dataset-root override; valid only when one dataset is selected.",
    )
    parser.add_argument(
        "--replica-configs-root",
        type=Path,
        default=DATASET_DEFINITIONS["Replica"]["configs_root"],
    )
    parser.add_argument(
        "--synthetic-configs-root",
        type=Path,
        default=DATASET_DEFINITIONS["Synthetic"]["configs_root"],
    )
    parser.add_argument(
        "--replica-dataset-root",
        type=Path,
        default=DATASET_DEFINITIONS["Replica"]["dataset_root"],
    )
    parser.add_argument(
        "--synthetic-dataset-root",
        type=Path,
        default=DATASET_DEFINITIONS["Synthetic"]["dataset_root"],
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="CSV destination. The default is derived from the selected dataset names.",
    )
    parser.add_argument(
        "--output-prefix",
        default="tracking_ba_eval",
        help="Subdirectory prefix used to isolate CoSLAM outputs, logs, and temp configs.",
    )
    parser.add_argument("--python-exec", default=sys.executable)
    args = parser.parse_args()

    args.datasets = list(dict.fromkeys(args.datasets))
    if (args.configs_root is not None or args.dataset_root is not None) and len(args.datasets) != 1:
        raise SystemExit(
            "--configs-root and --dataset-root can only be used with one selected dataset. "
            "Use the dataset-specific root options for multi-dataset runs."
        )

    if args.configs is not None and args.sampling is not None:
        raise SystemExit("Use either --configs or --sampling, not both.")
    args.configs = args.configs if args.configs is not None else args.sampling
    if args.configs is None:
        args.configs = DEFAULT_CONFIGS

    scene_jobs = resolve_scene_jobs(args.datasets, args.scenes)
    if not scene_jobs:
        raise SystemExit("No scenes were selected.")

    if args.output_csv is None:
        args.output_csv = default_output_csv(args.datasets)

    # Init CSV
    use_run_column = args.runs > 1
    include_dataset_column = len(args.datasets) > 1
    csv_columns = build_csv_columns(use_run_column, include_dataset_column)
    csv_header = ",".join(csv_columns)
    if not args.output_csv.exists():
        ensure_dir(args.output_csv.parent)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerow(csv_columns)
    else:
        with open(args.output_csv, "r", encoding="utf-8") as f:
            existing_header = f.readline().strip()
        if existing_header != csv_header:
            raise SystemExit(
                f"Existing CSV has incompatible header: {args.output_csv}\n"
                f"Expected: {csv_header}\n"
                "Use a new --output-csv path or update the existing CSV header before appending."
            )

    # Main Loop
    total_runs = len(scene_jobs) * len(args.configs) * len(args.sizes) * args.runs
    selected_summary = ", ".join(
        f"{dataset_name}:{scene}" for dataset_name, scene in scene_jobs
    )
    print(f"Selected scenes: {selected_summary}")
    print(f"Total Runs Scheduled: {total_runs} ({args.runs} iterations of the grid)")

    for run_idx in range(1, args.runs + 1):
        if use_run_column:
            print(f"\n=== STARTING RUN ITERATION {run_idx}/{args.runs} ===")

        for (dataset_name, scene), config_name, size in itertools.product(
            scene_jobs, args.configs, args.sizes
        ):
            effective_run_idx = run_idx if use_run_column else None
            metrics = run_sequence(
                args,
                dataset_name,
                scene,
                config_name,
                size,
                run_idx=effective_run_idx,
            )

            if metrics:
                row = []
                if use_run_column:
                    row.append(run_idx)
                if include_dataset_column:
                    row.append(dataset_name)
                row.extend(
                    [
                        scene,
                        config_name,
                        size,
                        metrics["Accuracy"],
                        metrics["Completion"],
                        metrics["Completion_Ratio"],
                        metrics["Depth_L1"],
                    ]
                )
                print(f"  > Done: {','.join(map(str, row))}")
                with open(args.output_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f, lineterminator="\n").writerow(row)
            else:
                prefix = f"Run {run_idx} " if use_run_column else ""
                print(
                    f"  > Failed: {prefix}{dataset_name} {scene} "
                    f"{config_name} T{size}"
                )


if __name__ == "__main__":
    main()
