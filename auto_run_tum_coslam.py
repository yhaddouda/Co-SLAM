#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess
import time
import yaml
import itertools
from pathlib import Path

# --- Configuration Definitions ---
# These remain the same to test your specific Hash/Morton setups
SETUP_DEFINITIONS = {
    "A": {"hash": "CoherentPrime", "morton_sort": False},
    "B": {"hash": "Morton",        "morton_sort": False},
    "C": {"hash": "CoherentPrime", "morton_sort": True},
    "D": {"hash": "Morton",        "morton_sort": True},
}

# Scenes specific to TUM (used if auto-scan fails or for default reference)
DEFAULT_TUM_SCENES = ["fr1_desk", "fr2_xyz", "fr3_office"]

def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def build_exp_name(setup: str, size: str) -> str:
    size_str = "DefaultSize" if size is None else f"Size{size}"
    return f"{setup}_{size_str}"

def override_config(base_cfg_path: Path, scene: str, setup_key: str, 
                    table_size: int | None) -> Path:
    cfg = read_yaml(base_cfg_path)

    # --- grid Injection ---
    # We ensure 'grid' exists to inject your EC-SLAM parameters into Co-SLAM
    cfg.setdefault("grid", {})
    
    # 1. Apply Setup
    setup_params = SETUP_DEFINITIONS[setup_key]
    cfg["grid"]["hash"] = setup_params["hash"] 
    cfg["grid"]["morton_sort"] = setup_params["morton_sort"]

    # 2. Apply Hash Table Size
    if table_size is not None:
        cfg["grid"]["hash_size"] = table_size

    # 3. Construct Experiment Name and Output Path for Co-SLAM
    # Co-SLAM usually defines output dir in the config or arguments. 
    # We will attempt to update it inside the YAML if the key exists, 
    # otherwise Co-SLAM likely handles it via command line args or defaults.
    # We inject it into a generic 'data' or 'experiment' block if present.
    exp_dir_name = build_exp_name(setup_key, table_size)
    
    # Check if 'data' block exists (common in Co-SLAM/NeuralSLAM configs)
    if "data" not in cfg:
        cfg["data"] = {}
        
    cfg["data"]["exp_name"] = exp_dir_name
    
    # Update output path if present to keep things organized
    base_output = cfg["data"].get("output", f"output/TUM/{scene}")
    if base_output.endswith("/"):
        base_output = base_output[:-1]
    cfg["data"]["output"] = base_output

    # 4. Write Temporary Config
    # Save in the SAME folder as base config to preserve relative paths
    tmp_cfg_name = f"temp_{scene}_{exp_dir_name}.yaml"
    tmp_cfg_path = base_cfg_path.parent / tmp_cfg_name
    
    write_yaml(tmp_cfg_path, cfg)
    
    return tmp_cfg_path

def parse_size_arg(value):
    if str(value).lower() in ["null", "none", "default"]:
        return None
    return int(value)

def main():
    parser = argparse.ArgumentParser(description="Batch Runner for Co-SLAM on TUM Dataset")

    # --- Sweep Parameters ---
    parser.add_argument("--setups", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"],
                        help="List of setups to run.")
    parser.add_argument("--sizes", nargs="+", default=["null"], type=str,
                        help="List of hashmap log2 sizes (e.g., 19). Use 'null' for default.")

    # --- Scene Selection ---
    # Default changed to look in configs/Tum
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="Explicit list of scene names (e.g. fr1_desk).")
    parser.add_argument("--configs-root", default="configs/Tum", type=Path,
                        help="Folder containing base scene YAML files.")
    
    # --- Execution & Logging ---
    parser.add_argument("--log-file", default="output/tum_execution_times.csv", type=Path,
                        help="Path to the global CSV log file.")
    parser.add_argument("--python-exec", default=sys.executable,
                        help="Python executable to use.")

    args = parser.parse_args()

    # --- Check for coslam.py ---
    target_script = "coslam.py"
    if not Path(target_script).exists():
        # Fallback check if user renamed it
        if Path("coslam.py").exists():
             target_script = "coslam.py"
        else:
            print(f"ERROR: '{target_script}' not found in current directory.")
            sys.exit(1)
        
    if not args.configs_root.exists():
        print(f"ERROR: Configs root not found: {args.configs_root.resolve()}")
        print("Please ensure you have created the 'configs/TUM' directory.")
        sys.exit(1)

    # 1. Discover Scenes
    if args.scenes:
        scenes = args.scenes
    else:
        # Scan directory for yaml files
        scenes = []
        # Exclude temp files or other non-scene files
        for yml in sorted(args.configs_root.glob("*.yaml")):
            if not yml.name.startswith("temp_") and "tum" not in yml.stem.lower():
                scenes.append(yml.stem)
        
        # If folder scan yields nothing, fallback to the hardcoded known TUM list
        if not scenes:
            print(f"WARNING: No YAML files found in {args.configs_root}. Using default TUM list.")
            scenes = DEFAULT_TUM_SCENES

    print(f"Total Runs Scheduled: {len(scenes) * len(args.setups) * len(args.sizes)}")
    print(f"  > Scenes: {scenes}")
    print(f"  > Target Script: {target_script}")
    print("-" * 60)

    # 2. Prepare Log File
    ensure_dir(args.log_file.parent)
    if not args.log_file.exists():
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write("Scene,Setup,TableSize,Time_Seconds\n")

    # 3. Build Combination Grid
    parsed_sizes = [parse_size_arg(s) for s in args.sizes]
    combinations = list(itertools.product(scenes, args.setups, parsed_sizes))

    # 4. Main Execution Loop
    for scene, setup, size in combinations:
        base_cfg = args.configs_root / f"{scene}.yaml"
        
        if not base_cfg.exists():
            print(f"SKIPPING: Config file not found: {base_cfg}")
            continue

        # Create temporary config
        tmp_cfg = override_config(base_cfg, scene, setup, size)

        print(f"\n--> STARTING: Scene={scene} | Setup={setup} | Size={size}")
        
        start_time = time.time()
        try:
            # --- ADAPTED COMMAND FOR CO-SLAM ---
            # Uses --config flag instead of positional argument
            cmd = [args.python_exec, target_script, "--config", str(tmp_cfg)]
            
            subprocess.run(cmd, check=True)  
            
            duration = time.time() - start_time
            
            size_log = "Default" if size is None else str(size)
            log_line = f"{scene},{setup},{size_log},{duration:.4f}"
            
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
            
            print(f"    SUCCESS: Finished in {duration:.2f}s")
            
            # Clean up temp file
            if tmp_cfg.exists(): os.remove(tmp_cfg)

        except subprocess.CalledProcessError as e:
            print(f"    FAILED: Execution error (Exit Code: {e.returncode})")
        except KeyboardInterrupt:
            print("\n    ABORTED: User interrupted.")
            if tmp_cfg.exists(): os.remove(tmp_cfg)
            sys.exit(1)
        except Exception as e:
            print(f"    ERROR: Unexpected script error: {e}")

    print(f"\nAll runs completed. Results saved to {args.log_file}")

if __name__ == "__main__":
    main()