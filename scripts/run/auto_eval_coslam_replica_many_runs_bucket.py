#!/usr/bin/env python3
import argparse
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


def is_raw_mesh(path: Path) -> bool:
    return re.fullmatch(r"mesh_track\d+\.ply", path.name) is not None


def apply_sampling_mode(cfg: dict, sampling: str) -> None:
    sampling_cfg = SAMPLING_DEFINITIONS[sampling]
    cfg.setdefault("grid", {})
    cfg.setdefault("training", {})
    cfg.setdefault("tracking_bucket", {})
    cfg.setdefault("tracking_naive_pruning", {})

    cfg["grid"]["hash"] = sampling_cfg["hash"]
    cfg["grid"]["morton_sort"] = sampling_cfg["morton_sort"]
    cfg["training"]["uniform_samples_until_depth"] = False
    cfg["tracking_naive_pruning"]["enabled"] = False
    cfg["tracking_bucket"]["enabled"] = sampling_cfg["tracking_bucket_enabled"]
    cfg["tracking_bucket"]["morton_inside_bucket"] = sampling_cfg["morton_inside_bucket"]


def run_sequence(args, scene, sampling, hash_size, run_idx=None):
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
    base_output = cfg["data"].get("output", f"output/Replica/{scene}")
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
    virt_cam_path = args.dataset_root / scene / "virtual_cameras"

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
    gt_mesh = args.dataset_root / scene / "gt_mesh_cull_virt_cams.ply"

    cmd_eval = [
        args.python_exec,
        "neural_slam_eval/eval_recon_headless.py",
        "--rec_mesh",
        str(culled_mesh),
        "--gt_mesh",
        str(gt_mesh),
        "--dataset_type",
        "Replica",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=DEFAULT_SCENES, help="List of scenes")
    parser.add_argument(
        "--sampling",
        nargs="+",
        default=DEFAULT_SAMPLING,
        choices=sorted(SAMPLING_DEFINITIONS),
        help="Sampling bucket mode(s): baseline, bucket, bucketD.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to repeat the entire experiment grid for averaging.",
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES, help="Hash sizes")
    parser.add_argument("--configs-root", type=Path, default="configs/Replica")
    parser.add_argument("--dataset-root", type=Path, default="../datasets/vv_data/Replica")
    parser.add_argument("--output-csv", type=Path, default="output/replica_bucket_sampling_results.csv")
    parser.add_argument(
        "--output-prefix",
        default="bucket_eval",
        help="Subdirectory prefix used to isolate CoSLAM outputs, logs, and temp configs.",
    )
    parser.add_argument("--python-exec", default=sys.executable)
    args = parser.parse_args()

    # Init CSV
    use_run_column = args.runs > 1
    csv_header = (
        "Run,Scene,Sampling,HashSize,Accuracy,Completion,CompletionRatio,DepthL1"
        if use_run_column
        else "Scene,Sampling,HashSize,Accuracy,Completion,CompletionRatio,DepthL1"
    )
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
                if use_run_column:
                    row = (
                        f"{run_idx},{scene},{sampling_name},{size},{metrics['Accuracy']},"
                        f"{metrics['Completion']},{metrics['Completion_Ratio']},{metrics['Depth_L1']}"
                    )
                else:
                    row = (
                        f"{scene},{sampling_name},{size},{metrics['Accuracy']},"
                        f"{metrics['Completion']},{metrics['Completion_Ratio']},{metrics['Depth_L1']}"
                    )
                print(f"  > Done: {row}")
                with open(args.output_csv, "a") as f:
                    f.write(row + "\n")
            else:
                prefix = f"Run {run_idx} " if use_run_column else ""
                print(f"  > Failed: {prefix}{scene} {sampling_name} T{size}")


if __name__ == "__main__":
    main()
