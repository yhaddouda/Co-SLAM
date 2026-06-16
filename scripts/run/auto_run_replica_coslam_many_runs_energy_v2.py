#!/usr/bin/env python3
import argparse
import sys
import os
import subprocess
import time
import yaml
import itertools
import glob
from pathlib import Path

# --- Configuration Definitions ---
SETUP_DEFINITIONS = {
    "A": {"hash": "CoherentPrime", "morton_sort": False},
    "B": {"hash": "Morton",        "morton_sort": False},
    "C": {"hash": "CoherentPrime", "morton_sort": True},
    "D": {"hash": "Morton",        "morton_sort": True},
}

DEFAULT_Replica_SCENES = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]

# ---------------- Energy (INA3221) helpers (Jetson AGX Orin) ----------------

def _read_int(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return int(f.read().strip())

def _read_str(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def find_ina3221_hwmon(addr: str = "0040") -> str | None:
    """
    Find hwmon directory for INA3221 at a given I2C address (e.g., '0040').
    We use a wildcard for the bus prefix because it may vary (0-0040, 1-0040, ...).
    """
    candidates = glob.glob(f"/sys/bus/i2c/drivers/ina3221/*-{addr}/hwmon/hwmon*")
    if not candidates:
        return None
    return sorted(candidates)[0]

def read_ina_channel(hwmon_dir: str, ch: int):
    """Return (label, v_mv, i_ma, p_w) for channel ch (1..3)."""
    label_path = os.path.join(hwmon_dir, f"in{ch}_label")
    v_path = os.path.join(hwmon_dir, f"in{ch}_input")      # mV
    i_path = os.path.join(hwmon_dir, f"curr{ch}_input")    # mA

    label = _read_str(label_path) if os.path.exists(label_path) else f"CH{ch}"
    v_mv = _read_int(v_path)
    i_ma = _read_int(i_path)
    p_w = (v_mv * i_ma) / 1_000_000.0
    return label, v_mv, i_ma, p_w

def read_total_power_w(hwmon_0040: str) -> tuple[float, dict]:
    """
    Total power = sum of channels 1..3 on INA3221 @ 0x40.
    On AGX Orin, these are typically: VDD_GPU_SOC, VDD_CPU_CV, VIN_SYS_5V0.
    """
    rails = {}
    p_total = 0.0
    for ch in (1, 2, 3):
        label, v_mv, i_ma, p_w = read_ina_channel(hwmon_0040, ch)
        rails[label] = {"v_mv": v_mv, "i_ma": i_ma, "p_w": p_w}
        p_total += p_w
    return p_total, rails

def measure_idle_power(hwmon_0040: str, hz: float, seconds: float) -> float:
    """Measure average idle power over a short window."""
    if seconds <= 0:
        return 0.0
    dt = 1.0 / hz
    t_end = time.monotonic() + seconds
    ps = []
    while time.monotonic() < t_end:
        p, _ = read_total_power_w(hwmon_0040)
        ps.append(p)
        time.sleep(dt)
    return sum(ps) / len(ps) if ps else 0.0

def run_with_energy(cmd: list[str], hz: float, hwmon_0040: str, idle_power_w: float = 0.0,
                    stdout=None, stderr=None):
    """
    Run a command while sampling power and integrating energy (trapezoid).
    Returns: (returncode, runtime_s, energy_j, avg_power_w, idle_power_w, net_energy_j)
    """
    dt = 1.0 / hz
    proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)

    t0 = time.monotonic()
    prev_t = None
    prev_p = None
    energy_j = 0.0

    while True:
        now = time.monotonic()
        p, _rails = read_total_power_w(hwmon_0040)

        if prev_t is not None:
            dte = now - prev_t
            energy_j += 0.5 * (prev_p + p) * dte

        prev_t = now
        prev_p = p

        ret = proc.poll()
        if ret is not None:
            break

        time.sleep(dt)

    t1 = time.monotonic()
    runtime_s = t1 - t0
    avg_power_w = (energy_j / runtime_s) if runtime_s > 0 else float("nan")
    net_energy_j = energy_j - idle_power_w * runtime_s

    return proc.returncode, runtime_s, energy_j, avg_power_w, idle_power_w, net_energy_j

# ---------------- Original YAML helpers ----------------

def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def build_exp_name(setup: str, size: str, run_idx: int) -> str:
    size_str = "DefaultSize" if size is None else f"Size{size}"
    # Added Run index to the experiment name to ensure unique folders in Co-SLAM
    return f"{setup}_{size_str}_Run{run_idx}"

def override_config(base_cfg_path: Path, scene: str, setup_key: str,
                    table_size: int | None, run_idx: int) -> Path:
    cfg = read_yaml(base_cfg_path)

    cfg.setdefault("grid", {})
    setup_params = SETUP_DEFINITIONS[setup_key]
    cfg["grid"]["hash"] = setup_params["hash"]
    cfg["grid"]["morton_sort"] = setup_params["morton_sort"]

    if table_size is not None:
        cfg["grid"]["hash_size"] = table_size

    exp_dir_name = build_exp_name(setup_key, table_size, run_idx)

    if "data" not in cfg:
        cfg["data"] = {}

    cfg["data"]["exp_name"] = exp_dir_name

    base_output = cfg["data"].get("output", f"output/Replica/{scene}")
    if base_output.endswith("/"):
        base_output = base_output[:-1]
    cfg["data"]["output"] = str(Path(base_output) / f"run_{run_idx}")

    tmp_cfg_name = f"temp_{scene}_{exp_dir_name}.yaml"
    tmp_cfg_path = base_cfg_path.parent / tmp_cfg_name
    write_yaml(tmp_cfg_path, cfg)

    return tmp_cfg_path

def parse_size_arg(value):
    if str(value).lower() in ["null", "none", "default"]:
        return None
    return int(value)

def main():
    parser = argparse.ArgumentParser(
        description="Batch Runner for Co-SLAM with Multiple Runs (Execution time + optional Energy logging on Jetson)"
    )

    # --- Multi-Run Parameter ---
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of times to repeat the entire experiment suite for averaging.")

    # --- Sweep Parameters ---
    parser.add_argument("--setups", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    parser.add_argument("--sizes", nargs="+", default=["null"], type=str)

    # --- Scene Selection ---
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--configs-root", default="configs/Replica", type=Path)

    # --- Execution & Logging ---
    parser.add_argument("--log-file", default="output/Replica_execution_times_energy.csv", type=Path)
    parser.add_argument("--python-exec", default=sys.executable)

    # --- Energy logging options ---
    parser.add_argument("--log-energy", action="store_true", default=False,
                        help="Enable energy logging via INA3221 sysfs (AGX Orin).")
    parser.add_argument("--energy-hz", type=float, default=20.0,
                        help="Power sampling rate (Hz) when --log-energy is enabled.")
    parser.add_argument("--idle-seconds", type=float, default=0.0,
                        help="If >0, measure idle power for this many seconds before each run and log NetEnergy_J.")

    # --- Output handling (fixes 'doesn't go back to line' issues) ---
    parser.add_argument("--redirect-output", action="store_true", default=False,
                        help="Redirect CoSLAM stdout/stderr to per-run log files (prevents mixed carriage-return output).")
    parser.add_argument("--run-log-dir", type=Path, default=Path("output/coslam_run_logs"),
                        help="Directory to store per-run stdout logs when --redirect-output is used.")

    args = parser.parse_args()

    target_script = "coslam.py"
    if not Path(target_script).exists():
        print(f"ERROR: '{target_script}' not found.")
        sys.exit(1)

    if not args.configs_root.exists():
        print(f"ERROR: Configs root not found: {args.configs_root}")
        sys.exit(1)

    hwmon_0040 = None
    if args.log_energy:
        hwmon_0040 = find_ina3221_hwmon("0040")
        if hwmon_0040 is None:
            print("WARNING: --log-energy requested but INA3221 0x40 hwmon was not found.")
            print("         Energy columns will be NaN. Check sysfs path/permissions or run with sudo.")
        else:
            try:
                labels = []
                for ch in (1, 2, 3):
                    lp = os.path.join(hwmon_0040, f"in{ch}_label")
                    labels.append(_read_str(lp) if os.path.exists(lp) else f"CH{ch}")
                print(f"Energy logging enabled using: {hwmon_0040} (channels: {', '.join(labels)})", flush=True)
            except Exception:
                print(f"Energy logging enabled using: {hwmon_0040}", flush=True)

    # 1. Discover Scenes
    if args.scenes:
        scenes = args.scenes
    else:
        scenes = [yml.stem for yml in sorted(args.configs_root.glob("*.yaml"))
                  if not yml.name.startswith("temp_")]
        if not scenes:
            scenes = DEFAULT_Replica_SCENES

    total_iterations = len(scenes) * len(args.setups) * len(args.sizes) * args.runs
    print(f"Total Runs Scheduled: {total_iterations} ({args.runs} iterations of the grid)", flush=True)
    print("-" * 60, flush=True)

    # 2. Prepare Log File
    ensure_dir(args.log_file.parent)
    if not args.log_file.exists():
        with open(args.log_file, "w", encoding="utf-8") as f:
            f.write("Run,Scene,Setup,TableSize,Time_Seconds,Energy_J,AvgPower_W,IdlePower_W,NetEnergy_J,StdoutLog\n")

    # Prepare log dir
    if args.redirect_output:
        ensure_dir(args.run_log_dir)

    # 3. Build Combination Grid
    parsed_sizes = [parse_size_arg(s) for s in args.sizes]

    # 4. Main Execution Loop
    for run_idx in range(1, args.runs + 1):
        print(f"\n=== STARTING RUN ITERATION {run_idx}/{args.runs} ===", flush=True)

        combinations = list(itertools.product(scenes, args.setups, parsed_sizes))

        for scene, setup, size in combinations:
            base_cfg = args.configs_root / f"{scene}.yaml"
            if not base_cfg.exists():
                continue

            tmp_cfg = override_config(base_cfg, scene, setup, size, run_idx)

            print(f"--> [Run {run_idx}] Scene={scene} | Setup={setup} | Size={size}", flush=True)

            size_log = "Default" if size is None else str(size)

            cmd = [args.python_exec, target_script, "--config", str(tmp_cfg)]

            stdout_log_path = ""
            log_fh = None

            try:
                # Optional redirection of child output (best fix for line/carriage-return issues)
                if args.redirect_output:
                    stdout_log_path = str(args.run_log_dir / f"run{run_idx}_{scene}_{setup}_{size_log}.log")
                    log_fh = open(stdout_log_path, "w", encoding="utf-8")
                    child_stdout = log_fh
                    child_stderr = subprocess.STDOUT
                else:
                    child_stdout = None
                    child_stderr = None

                # Execute and optionally measure energy
                if args.log_energy and hwmon_0040 is not None:
                    idle_power_w = measure_idle_power(hwmon_0040, args.energy_hz, args.idle_seconds) if args.idle_seconds > 0 else 0.0
                    ret, runtime_s, energy_j, avg_power_w, idle_pw, net_energy_j = run_with_energy(
                        cmd, args.energy_hz, hwmon_0040, idle_power_w=idle_power_w,
                        stdout=child_stdout, stderr=child_stderr
                    )
                else:
                    start_time = time.time()
                    ret = subprocess.call(cmd, stdout=child_stdout, stderr=child_stderr)
                    runtime_s = time.time() - start_time
                    energy_j = avg_power_w = idle_pw = net_energy_j = float("nan")

                # Ensure our summary starts on a new line (if child used carriage returns)
                sys.stdout.write("\n")
                sys.stdout.flush()

                # Logging
                log_line = f"{run_idx},{scene},{setup},{size_log},{runtime_s:.4f},{energy_j:.6f},{avg_power_w:.6f},{idle_pw:.6f},{net_energy_j:.6f},{stdout_log_path}"
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")

                if ret == 0:
                    msg = f"    SUCCESS: Finished in {runtime_s:.2f}s"
                    if args.log_energy and hwmon_0040 is not None:
                        msg += f" | Energy={energy_j:.2f}J | AvgP={avg_power_w:.2f}W"
                        if args.idle_seconds > 0:
                            msg += f" | NetEnergy={net_energy_j:.2f}J (Idle={idle_pw:.2f}W)"
                    if stdout_log_path:
                        msg += f" | stdout: {stdout_log_path}"
                    print(msg, flush=True)

                    if tmp_cfg.exists():
                        os.remove(tmp_cfg)
                else:
                    msg = f"    FAILED: Exit Code {ret}"
                    if stdout_log_path:
                        msg += f" | stdout: {stdout_log_path}"
                    print(msg, flush=True)
                    # keep tmp_cfg for debugging

            except KeyboardInterrupt:
                print("\n    ABORTED: User interrupted.", flush=True)
                if tmp_cfg.exists():
                    os.remove(tmp_cfg)
                sys.exit(1)
            except Exception as e:
                print(f"    FAILED: {e}", flush=True)
                # keep tmp_cfg for debugging
            finally:
                if log_fh is not None:
                    log_fh.close()

    print(f"\nAll {args.runs} runs completed. Data logged to {args.log_file}", flush=True)

if __name__ == "__main__":
    main()
