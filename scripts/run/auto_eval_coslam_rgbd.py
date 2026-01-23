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
# The list of config files provided in the prompt
DEFAULT_SCENES = ["br", "ck", "gr", "gwr", "ma", "tg", "wr"]
DEFAULT_SIZES = list(range(13, 25))  # Sweep hash sizes 13 to 24

# Mapping from the short config name to the actual folder name in neural_rgbd_data
# based on standard conventions for this dataset (e.g., wr -> whiteroom)
SCENE_MAPPING = {
    "br": "breakfast_room",
    "ck": "complete_kitchen",
    "gr": "green_room",
    "gwr": "grey_white_room",
    "ma": "morning_apartment",
    "tg": "thin_geometry",
    "wr": "whiteroom"
}

def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def parse_metrics(output_str: str) -> dict:
    """Parses standard output from eval_recon_headless.py."""
    metrics = {"Accuracy": "N/A", "Completion": "N/A", "Completion_Ratio": "N/A", "Depth_L1": "N/A"}
    
    patterns = {
        "Accuracy": r"accuracy:\s+([0-9.]+)",
        "Completion": r"completion:\s+([0-9.]+)",
        "Completion_Ratio": r"completion ratio:\s+([0-9.]+)",
        "Depth_L1": r"Depth L1:\s+([0-9.]+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output_str)
        if match:
            metrics[key] = match.group(1)
            
    return metrics

def run_sequence(args, scene, hash_size):
    # 1. Prepare Configuration
    # Looks for configs/Synthetic/wr.yaml, etc.
    base_cfg = args.configs_root / f"{scene}.yaml"
    if not base_cfg.exists():
        print(f"  [Error] Config not found: {base_cfg}")
        return None

    cfg = read_yaml(base_cfg)
    
    # We force the experiment name to T{hash_size} to ensure we know where the output goes
    # This overrides the default timestamp/random naming.
    exp_name = f"T{hash_size}"
    
    # Ensure keys exist
    cfg.setdefault("grid", {})
    cfg.setdefault("data", {})
    
    # Set Flags
    cfg["grid"]["hash_size"] = hash_size
    cfg["data"]["exp_name"] = exp_name
    
    # Save Temp Config
    tmp_config_path = Path("temp_configs") / f"{scene}_{exp_name}.yaml"
    tmp_config_path.parent.mkdir(exist_ok=True)
    write_yaml(tmp_config_path, cfg)
    
    # 2. Run CoSLAM
    print(f"  [1/3] Running CoSLAM (Scene={scene}, Hash={hash_size})...")
    log_path = Path("output") / "logs" / f"{scene}_{exp_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(log_path, "w") as f:
            subprocess.run(
                [args.python_exec, "coslam.py", "--config", str(tmp_config_path)],
                stdout=f, stderr=subprocess.STDOUT, check=True
            )
    except subprocess.CalledProcessError:
        print("  [Error] SLAM crashed. See logs.")
        return None

    # 3. Locate Mesh
    # Path logic: output/Synthetic/{scene}/{exp_name}/mesh_track*.ply
    output_dir = Path("output") / "Synthetic" / scene / exp_name
    
    # Use glob to find the mesh regardless of the specific iteration number (e.g. 1999 vs 1675)
    meshes = sorted(output_dir.glob("mesh_track*.ply"), key=os.path.getmtime)
    
    if not meshes:
        print(f"  [Error] No mesh found in expected dir: {output_dir}")
        return None
        
    input_mesh = meshes[-1] # Take the last one generated
    print(f"  > Found mesh: {input_mesh.name}")

    # 4. Cull Mesh
    print("  [2/3] Culling Mesh...")
    
    # Resolve the full folder name (e.g. 'wr' -> 'whiteroom')
    full_scene_name = SCENE_MAPPING.get(scene, scene)
    virt_cam_path = args.dataset_root / full_scene_name / "virtual_cameras"
    
    # Cull output name is automatically: {input_stem}_cull_virt_cams.ply
    culled_mesh = input_mesh.parent / f"{input_mesh.stem}_cull_virt_cams.ply"
    
    cmd_cull = [
        args.python_exec, "neural_slam_eval/cull_mesh.py",
        "--config", str(tmp_config_path),
        "--input_mesh", str(input_mesh),
        "--remove_occlusion",
        "--virtual_cameras",
        "--virt_cam_path", str(virt_cam_path),
        "--gt_pose"
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
        args.python_exec, "neural_slam_eval/eval_recon_headless.py",
        "--rec_mesh", str(culled_mesh),
        "--gt_mesh", str(gt_mesh),
        "--dataset_type", "RGBD", # Changed from Replica to RGBD
        "-2d", "-3d"
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
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES, help="Hash sizes")
    # Updated default paths for RGBD Synthetic workflow
    parser.add_argument("--configs-root", type=Path, default="configs/Synthetic")
    parser.add_argument("--dataset-root", type=Path, default="../datasets/vv_data/neural_rgbd_data")
    parser.add_argument("--output-csv", type=Path, default="output/rgbd_results.csv")
    parser.add_argument("--python-exec", default=sys.executable)
    args = parser.parse_args()

    # Init CSV
    if not args.output_csv.exists():
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w") as f:
            f.write("Scene,HashSize,Accuracy,Completion,CompletionRatio,DepthL1\n")

    # Main Loop
    for scene, size in itertools.product(args.scenes, args.sizes):
        metrics = run_sequence(args, scene, size)
        
        if metrics:
            row = f"{scene},{size},{metrics['Accuracy']},{metrics['Completion']},{metrics['Completion_Ratio']},{metrics['Depth_L1']}"
            print(f"  > Done: {row}")
            with open(args.output_csv, "a") as f:
                f.write(row + "\n")
        else:
            print(f"  > Failed: {scene} T{size}")

if __name__ == "__main__":
    main()