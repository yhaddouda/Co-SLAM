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
SETUP_DEFINITIONS = {
    "A": {"hash": "CoherentPrime", "morton_sort": False},
    "B": {"hash": "Morton",        "morton_sort": False},
    "C": {"hash": "CoherentPrime", "morton_sort": True},
    "D": {"hash": "Morton",        "morton_sort": True},
}

MODE_UNIFORM_SAMPLES_UNTIL_DEPTH = {
    "baseline": False,
    "cropped": True,
}

DEFAULT_Replica_SCENES = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]
DEFAULT_LOG_FILE = Path("output/Replica_execution_times_sampling.csv")
LOG_HEADER = "Run,Mode,Scene,Setup,TableSize,Time_Seconds\n"

def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def build_exp_name(mode: str, setup: str, size: int | None, run_idx: int) -> str:
    size_str = "DefaultSize" if size is None else f"Size{size}"
    # Added Run index to the experiment name to ensure unique folders in Co-SLAM
    return f"{mode}_{setup}_{size_str}_Run{run_idx}"

def override_config(base_cfg_path: Path, scene: str, setup_key: str, 
                    table_size: int | None, run_idx: int, mode: str) -> Path:
    cfg = read_yaml(base_cfg_path)

    cfg.setdefault("grid", {})
    setup_params = SETUP_DEFINITIONS[setup_key]
    cfg["grid"]["hash"] = setup_params["hash"] 
    cfg["grid"]["morton_sort"] = setup_params["morton_sort"]

    cfg.setdefault("training", {})
    cfg["training"]["uniform_samples_until_depth"] = MODE_UNIFORM_SAMPLES_UNTIL_DEPTH[mode]

    if table_size is not None:
        cfg["grid"]["hash_size"] = table_size

    exp_dir_name = build_exp_name(mode, setup_key, table_size, run_idx)
    
    if "data" not in cfg:
        cfg["data"] = {}
        
    cfg["data"]["exp_name"] = exp_dir_name
    
    # Update output path to include a 'run_X' subfolder for organization
    base_output = cfg["data"].get("output", f"output/Replica/{scene}")
    if base_output.endswith("/"):
        base_output = base_output[:-1]
    
    # Path will look like: output/Replica/office0/run_1/
    cfg["data"]["output"] = str(Path(base_output) / f"run_{run_idx}")

    # Save temporary config
    tmp_cfg_name = f"temp_{scene}_{exp_dir_name}.yaml"
    tmp_cfg_path = base_cfg_path.parent / tmp_cfg_name
    write_yaml(tmp_cfg_path, cfg)
    
    return tmp_cfg_path

def parse_size_arg(value):
    if str(value).lower() in ["null", "none", "default"]:
        return None
    return int(value)

def normalize_scene_arg(value: str) -> str:
    scene_name = Path(value).name
    return Path(scene_name).stem if scene_name.endswith(".yaml") else scene_name

def main():
    parser = argparse.ArgumentParser(description="Batch Runner for Co-SLAM with Multiple Runs")

    # --- New Multi-Run Parameter ---
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of times to repeat the entire experiment suite for averaging.")

    # --- Sweep Parameters ---
    parser.add_argument("--setups", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    parser.add_argument("--sizes", nargs="+", default=["null"], type=str)
    parser.add_argument(
        "--mode",
        nargs="+",
        choices=sorted(MODE_UNIFORM_SAMPLES_UNTIL_DEPTH),
        default=["baseline"],
        help=(
            "One or more Replica sampling modes. "
            "'baseline' sets training.uniform_samples_until_depth=False, "
            "'cropped' sets it to True."
        ),
    )

    # --- Scene Selection ---
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--configs-root", default="configs/Replica", type=Path)
    
    # --- Execution & Logging ---
    parser.add_argument("--log-file", default=DEFAULT_LOG_FILE, type=Path)
    parser.add_argument("--python-exec", default=sys.executable)

    args = parser.parse_args()
    selected_modes = list(dict.fromkeys(args.mode))

    target_script = "coslam.py"
    if not Path(target_script).exists():
        print(f"ERROR: '{target_script}' not found.")
        sys.exit(1)
        
    if not args.configs_root.exists():
        print(f"ERROR: Configs root not found: {args.configs_root}")
        sys.exit(1)

    # 1. Discover Scenes
    if args.scenes:
        scenes = [normalize_scene_arg(scene) for scene in args.scenes]
    else:
        scenes = [yml.stem for yml in sorted(args.configs_root.glob("*.yaml")) 
                  if not yml.name.startswith("temp_")]
        if not scenes:
            scenes = DEFAULT_Replica_SCENES

    total_iterations = len(scenes) * len(selected_modes) * len(args.setups) * len(args.sizes) * args.runs
    print(f"Total Runs Scheduled: {total_iterations} ({args.runs} iterations of the grid)")
    mode_summary = ", ".join(
        f"{mode}(uniform_samples_until_depth={MODE_UNIFORM_SAMPLES_UNTIL_DEPTH[mode]})"
        for mode in selected_modes
    )
    print(f"Selected modes: {mode_summary}")
    print("-" * 60)

    # 2. Prepare Log File (Added 'Run' column)
    ensure_dir(args.log_file.parent)
    if not args.log_file.exists():
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write(LOG_HEADER)
    else:
        with open(args.log_file, "r", encoding="utf-8") as f:
            current_header = f.readline()
        if current_header and current_header != LOG_HEADER:
            print(
                f"ERROR: Existing log file '{args.log_file}' has header "
                f"'{current_header.strip()}'. Expected '{LOG_HEADER.strip()}'. "
                "Use --log-file with a fresh CSV path."
            )
            sys.exit(1)

    # 3. Build Combination Grid
    parsed_sizes = [parse_size_arg(s) for s in args.sizes]
    
    # 4. Main Execution Loop
    # Outer loop handles the number of runs
    for run_idx in range(1, args.runs + 1):
        print(f"\n=== STARTING RUN ITERATION {run_idx}/{args.runs} ===")
        
        combinations = list(itertools.product(scenes, selected_modes, args.setups, parsed_sizes))

        for scene, mode, setup, size in combinations:
            base_cfg = args.configs_root / f"{scene}.yaml"
            if not base_cfg.exists():
                print(f"    SKIP: Config not found: {base_cfg}")
                continue

            # Pass run_idx to ensure unique output directories and experiment names
            tmp_cfg = override_config(base_cfg, scene, setup, size, run_idx, mode)

            print(f"--> [Run {run_idx}] Mode={mode} | Scene={scene} | Setup={setup} | Size={size}")
            
            start_time = time.time()
            try:
                cmd = [args.python_exec, target_script, "--config", str(tmp_cfg)]
                subprocess.run(cmd, check=True)  
                
                duration = time.time() - start_time
                size_log = "Default" if size is None else str(size)
                
                # Log with run index
                log_line = f"{run_idx},{mode},{scene},{setup},{size_log},{duration:.4f}"
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")
                
                print(f"    SUCCESS: Finished in {duration:.2f}s")
                if tmp_cfg.exists(): os.remove(tmp_cfg)

            except subprocess.CalledProcessError as e:
                print(f"    FAILED: Execution error (Exit Code: {e.returncode})")
            except KeyboardInterrupt:
                print("\n    ABORTED: User interrupted.")
                if tmp_cfg.exists(): os.remove(tmp_cfg)
                sys.exit(1)

    print(f"\nAll {args.runs} runs completed. Data logged to {args.log_file}")

if __name__ == "__main__":
    main()
