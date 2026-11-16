#!/usr/bin/env python3
import argparse
import copy
import sys
import os
import subprocess
import yaml
import itertools
import re
from pathlib import Path

# --- Constants ---
DEFAULT_SCENES = ["office0"]  # Add more scenes here or pass via command line
DEFAULT_SIZES = list(range(13, 25))  # 13 to 24
DEFAULT_SAMPLING = ["baseline", "bucket", "bucketD"]
DATASET_DEFINITIONS = {
    "replica": {
        "configs_root": "configs/Replica",
        "dataset_root": "../datasets/vv_data/Replica",
        "scenes": DEFAULT_SCENES,
        "output_dir": "Replica",
        "eval_type": "Replica",
        "scene_mapping": {},
    },
    "synthetic": {
        "configs_root": "configs/Synthetic",
        "dataset_root": "../datasets/vv_data/neural_rgbd_data",
        "scenes": ["br", "ck", "gr", "gwr", "ma", "tg", "wr"],
        "output_dir": "Synthetic",
        "eval_type": "RGBD",
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
    "tum": {
        "configs_root": "configs/Tum",
        "dataset_root": None,  # Use data.datadir from each scene config.
        "scenes": ["fr1_desk", "fr2_xyz", "fr3_office"],
        "output_dir": "TUM",
        "scene_mapping": {
            "fr1_desk": "rgbd_dataset_freiburg1_desk",
            "fr2_xyz": "rgbd_dataset_freiburg2_xyz",
            "fr3_office": "rgbd_dataset_freiburg3_long_office_household",
        },
    },
    "scannet": {
        "configs_root": "configs/ScanNet",
        "dataset_root": "../datasets/data/scans",
        "scenes": [
            "scene0000", "scene0059", "scene0106",
            "scene0169", "scene0181", "scene0207",
        ],
        "output_dir": "scannet",
        "scene_mapping": {
            scene: f"{scene}_00" for scene in (
                "scene0000", "scene0059", "scene0106",
                "scene0169", "scene0181", "scene0207",
            )
        },
    },
}
ATE_DATASETS = {"tum", "scannet"}
SAMPLING_DEFINITIONS = {
    "baseline": {
        "hash": "CoherentPrime",
        "morton_sort": False,
        "tracking_bucket_enabled": False,
        "morton_inside_bucket": False,
    },
    "bucket": {
        "hash": "CoherentPrime",
        "morton_sort": False,
        "tracking_bucket_enabled": True,
        "morton_inside_bucket": False,
    },
    "bucketD": {
        "hash": "Morton",
        "morton_sort": True,
        "tracking_bucket_enabled": True,
        "morton_inside_bucket": True,
    },
}


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


def build_exp_name(sampling_name: str, hash_size: int, run_idx: int | None = None) -> str:
    base_name = f"{sampling_name}_T{hash_size}"
    if run_idx is None:
        return base_name
    return f"{base_name}_Run{run_idx}"


def read_ate_rmse(path: Path) -> str:
    """Read the final trajectory evaluation appended by pose_evaluation.

    NumPy versions may write scalar values as np.float64(value).
    """
    last_line = next(
        line for line in reversed(path.read_text().splitlines()) if line.strip()
    )
    match = re.search(
        r"['\"]absolute_translational_error\.rmse['\"]\s*:\s*"
        r"(?:[\w.]+\()?([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)",
        last_line,
    )
    if match is None:
        raise ValueError(f"No final ATE RMSE in {path}")
    return match.group(1)


def is_raw_mesh(path: Path) -> bool:
    return re.fullmatch(r"mesh_track\d+\.ply", path.name) is not None


def apply_sampling_mode(cfg: dict, sampling: str) -> None:
    sampling_cfg = SAMPLING_DEFINITIONS[sampling]
    cfg.setdefault("grid", {})
    cfg.setdefault("training", {})
    cfg.setdefault("tracking_bucket", {})
    cfg.setdefault("tracking_naive_pruning", {})
    cfg.setdefault("tracking_depth_spatial", {})
    cfg.setdefault("ba_bucket", {})

    cfg["grid"]["hash"] = sampling_cfg["hash"]
    cfg["grid"]["morton_sort"] = sampling_cfg["morton_sort"]
    cfg["training"]["uniform_samples_until_depth"] = False
    cfg["tracking_naive_pruning"]["enabled"] = False
    cfg["tracking_depth_spatial"]["enabled"] = False
    cfg["ba_bucket"]["enabled"] = False
    cfg["tracking_bucket"]["enabled"] = sampling_cfg["tracking_bucket_enabled"]
    cfg["tracking_bucket"]["morton_inside_bucket"] = sampling_cfg["morton_inside_bucket"]


def run_sequence(args, scene, sampling, hash_size, run_idx=None):
    dataset = getattr(args, "dataset", "replica")
    dataset_cfg = DATASET_DEFINITIONS[dataset]
    full_scene_name = dataset_cfg["scene_mapping"].get(scene, scene)
    # 1. Prepare Configuration
    base_cfg = args.configs_root / f"{scene}.yaml"
    if not base_cfg.exists():
        print(f"  [Error] Config not found: {base_cfg}")
        return None

    cfg = read_yaml(base_cfg)
    sampling_name = sampling

    # Include sampling mode, and optionally run index, so repeated runs do not overwrite each other.
    exp_name = build_exp_name(sampling_name, hash_size, run_idx)

    # Ensure keys exist
    cfg.setdefault("grid", {})
    cfg.setdefault("data", {})
    cfg.setdefault("timing", {})

    # Set flags
    cfg["grid"]["hash_size"] = hash_size
    cfg["data"]["exp_name"] = exp_name
    cfg["timing"]["disable_eval"] = False
    apply_sampling_mode(cfg, sampling)
    if dataset in ATE_DATASETS and args.dataset_root is not None:
        cfg["data"]["datadir"] = str(args.dataset_root / full_scene_name)
    base_output = cfg["data"].get("output", f"output/{dataset_cfg['output_dir']}/{scene}")
    cfg["data"]["output"] = str(Path(base_output) / args.output_prefix)
    if run_idx is not None:
        cfg["data"]["output"] = str(Path(cfg["data"]["output"]) / f"run_{run_idx}")

    # Save Temp Config
    tmp_config_path = Path("temp_configs") / args.output_prefix / f"{scene}_{exp_name}.yaml"
    ensure_dir(tmp_config_path.parent)
    write_yaml(tmp_config_path, cfg)

    # 2. Run CoSLAM
    if run_idx is None:
        print(f"  [1/3] Running CoSLAM (Scene={scene}, Sampling={sampling_name}, Hash={hash_size})...")
    else:
        print(f"  [1/3] Running CoSLAM (Run={run_idx}, Scene={scene}, Sampling={sampling_name}, Hash={hash_size})...")
    log_path = Path("output") / "logs" / args.output_prefix / f"{scene}_{exp_name}.log"
    ensure_dir(log_path.parent)

    try:
        with open(log_path, "w") as f:
            subprocess.run(
                [args.python_exec, "coslam.py", "--config", str(tmp_config_path)],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except subprocess.CalledProcessError:
        print("  [Error] SLAM crashed. See logs.")
        return None

    output_dir = Path(cfg["data"]["output"]) / exp_name
    if dataset in ATE_DATASETS:
        # TUM and ScanNet runs use trajectory metrics rather than mesh metrics.
        # CoSLAM already evaluates both absolute and relative-pose trajectories.
        try:
            return {
                "ATE_RMSE_m": read_ate_rmse(output_dir / "output.txt"),
                "ATE_Relative_RMSE_m": read_ate_rmse(output_dir / "output_relative.txt"),
            }
        except (OSError, ValueError, StopIteration) as exc:
            print(f"  [Error] Trajectory evaluation failed: {exc}")
            return None

    # 3. Locate Mesh (Deterministic Path)
    # Path logic: <cfg[data][output]>/<exp_name>/mesh_track*.ply
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
    virt_cam_path = args.dataset_root / full_scene_name / "virtual_cameras"

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
    gt_mesh = args.dataset_root / full_scene_name / "gt_mesh_cull_virt_cams.ply"

    cmd_eval = [
        args.python_exec,
        "neural_slam_eval/eval_recon_headless.py",
        "--rec_mesh",
        str(culled_mesh),
        "--gt_mesh",
        str(gt_mesh),
        "--dataset_type",
        dataset_cfg["eval_type"],
        "-2d",
        "-3d",
    ]

    try:
        result = subprocess.run(cmd_eval, check=True, capture_output=True, text=True)
        return parse_metrics(result.stdout)
    except subprocess.CalledProcessError:
        print("  [Error] Evaluation failed.")
        return None


def main(
    *,
    dataset_support=False,
    default_output_csv="output/replica_bucket_sampling_results.csv",
    default_output_prefix="bucket_eval",
):
    parser = argparse.ArgumentParser(
        description="Run a repeated experiment grid and record dataset quality metrics."
    )
    if dataset_support:
        parser.add_argument(
            "--dataset", type=str.lower, choices=sorted(DATASET_DEFINITIONS),
            nargs="+", default=["replica"],
            help="Dataset(s) to run in order (default: replica).",
        )
    parser.add_argument(
        "--scenes", nargs="+", default=None,
        help="Scene config names; defaults: Replica office0; all TUM/Synthetic/ScanNet scenes",
    )
    parser.add_argument(
        "--sampling",
        nargs="+",
        default=DEFAULT_SAMPLING,
        choices=sorted(SAMPLING_DEFINITIONS),
        help="Experiment mode(s): " + ", ".join(DEFAULT_SAMPLING),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to repeat the entire experiment grid for averaging.",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES, help="Hash sizes")
    parser.add_argument(
        "--configs-root", type=Path,
        help="Scene config directory; defaults depend on the dataset",
    )
    parser.add_argument(
        "--dataset-root", type=Path,
        help="Mesh evaluation data root for Replica/Synthetic; sequence data root for TUM/ScanNet",
    )
    parser.add_argument(
        "--output-csv", type=Path,
        help="Results CSV; use {dataset} in the path for multiple datasets (default: " + default_output_csv + ")",
    )
    parser.add_argument(
        "--output-prefix",
        default=default_output_prefix,
        help="Subdirectory prefix used to isolate CoSLAM outputs, logs, and temp configs.",
    )
    parser.add_argument("--python-exec", default=sys.executable)
    args = parser.parse_args()
    datasets = list(dict.fromkeys(getattr(args, "dataset", ["replica"])))
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if len(datasets) > 1:
        csv_template = str(args.output_csv) if args.output_csv is not None else default_output_csv
        if "{dataset}" not in csv_template:
            parser.error("For multiple datasets, --output-csv must contain {dataset} to keep results separate")
    for dataset in datasets:
        dataset_args = copy.copy(args)
        dataset_args.dataset = dataset
        resolve_dataset_defaults(dataset_args, default_output_csv)
        print(f"\n=== DATASET: {dataset} ===")
        run_dataset(dataset_args)


def resolve_dataset_defaults(args, default_output_csv):
    """Resolve defaults independently so one dataset cannot change the next."""
    dataset_cfg = DATASET_DEFINITIONS[args.dataset]
    if args.scenes is None:
        args.scenes = dataset_cfg["scenes"]
    if args.configs_root is None:
        args.configs_root = Path(dataset_cfg["configs_root"])
    if args.dataset_root is None and dataset_cfg["dataset_root"] is not None:
        args.dataset_root = Path(dataset_cfg["dataset_root"])
    if args.output_csv is None:
        args.output_csv = Path(default_output_csv.format(dataset=args.dataset))
    else:
        args.output_csv = Path(str(args.output_csv).format(dataset=args.dataset))


def run_dataset(args):
    # Init CSV
    use_run_column = args.runs > 1
    metric_keys = (
        ["ATE_RMSE_m", "ATE_Relative_RMSE_m"] if args.dataset in ATE_DATASETS
        else ["Accuracy", "Completion", "Completion_Ratio", "Depth_L1"]
    )
    metric_header = (
        "ATE_RMSE_m,ATE_Relative_RMSE_m" if args.dataset in ATE_DATASETS
        else "Accuracy,Completion,CompletionRatio,DepthL1"
    )
    csv_header = ("Run," if use_run_column else "") + "Scene,Sampling,HashSize," + metric_header
    if not args.output_csv.exists():
        ensure_dir(args.output_csv.parent)
        with open(args.output_csv, "w") as f:
            f.write(csv_header + "\n")
    else:
        with open(args.output_csv, "r") as f:
            existing_header = f.readline().strip()
        if existing_header != csv_header:
            raise SystemExit(
                f"Existing CSV has incompatible header: {args.output_csv}\n"
                f"Expected: {csv_header}\n"
                "Use a new --output-csv path or update the existing CSV header before appending."
            )

    # Main Loop
    total_runs = len(args.scenes) * len(args.sampling) * len(args.sizes) * args.runs
    print(f"Total Runs Scheduled: {total_runs} ({args.runs} iterations of the grid)")

    for run_idx in range(1, args.runs + 1):
        if use_run_column:
            print(f"\n=== STARTING RUN ITERATION {run_idx}/{args.runs} ===")

        for scene, sampling, size in itertools.product(args.scenes, args.sampling, args.sizes):
            effective_run_idx = run_idx if use_run_column else None
            metrics = run_sequence(args, scene, sampling, size, run_idx=effective_run_idx)
            sampling_name = sampling

            if metrics:
                row = (f"{run_idx}," if use_run_column else "") + f"{scene},{sampling_name},{size},"
                row += ",".join(metrics[key] for key in metric_keys)
                print(f"  > Done: {row}")
                with open(args.output_csv, "a") as f:
                    f.write(row + "\n")
            else:
                prefix = f"Run {run_idx} " if use_run_column else ""
                print(f"  > Failed: {prefix}{scene} {sampling_name} T{size}")


if __name__ == "__main__":
    main(
        dataset_support=True,
        default_output_csv="output/{dataset}_bucket_sampling_results.csv",
    )
