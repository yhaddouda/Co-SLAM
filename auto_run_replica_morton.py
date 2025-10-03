#!/usr/bin/env python3
"""
auto_run_replica_morton.py
--------------------------
Batch-run CoSLAM experiments on Replica/office0 by sweeping:
- grid.hash_size (13..20 by default)
- grid.morton_sort (enabled by default; can be disabled)
- grid.morton_R (default set: 24, 32, 48, 64, 96, 128)

Unlike the patch/random script, this one DOES NOT modify sampling_tracking.*
It only overrides the grid parameters and data.exp_name for clean comparisons.

For each run:
- Write a temporary YAML that overrides:
    grid.hash_size
    grid.morton_sort
    grid.morton_R
    data.exp_name -> e.g., "13_fp16_R24"
- Launch `python coslam.py --config <temp_config.yaml>`
- Time the run, optionally parse peak RAM via `tegrastats`
- Append one line per run to a single global log (CSV-ish key=value line)

Usage example:
    python3 auto_run_replica_morton.py \
      --scene office0 \
      --scene-cfg configs/Replica/office0.yaml \
      --tag fp16 \
      --hash-min 13 --hash-max 20 \
      --morton-sort \
      --morton-Rs 24 32 48 64 96 128 \
      --global-log output/Replica/office0/office0_morton_runs.log
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import yaml  # PyYAML
except Exception as e:
    print("ERROR: PyYAML is required. Install it with:  pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def find_tegrastats() -> str | None:
    candidates = ["/usr/bin/tegrastats", "/bin/tegrastats", "/usr/sbin/tegrastats", "tegrastats"]
    for c in candidates:
        if shutil.which(c):
            return shutil.which(c)
    return None


def extract_peak_ram_from_file(logfile: Path) -> int | None:
    if not logfile.exists():
        return None
    peak = None
    pat = re.compile(r"RAM\s+(\d+)[/ ]")  # captures the 'used' MB
    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                val = int(m.group(1))
                if peak is None or val > peak:
                    peak = val
    return peak


def build_exp_name(hash_size: int, tag: str, morton_R: int, morton_sort: bool) -> str:
    if morton_sort:
        return f"{hash_size}_{tag}_R{morton_R}"
    else:
        return f"{hash_size}_{tag}_noMorton_R{morton_R}"


def override_config(base_cfg_path: Path, hash_size: int, morton_sort: bool, morton_R: int,
                    tag: str, scene: str) -> Path:
    cfg = read_yaml(base_cfg_path)

    cfg.setdefault("data", {})
    cfg.setdefault("grid", {})

    cfg["grid"]["hash_size"] = int(hash_size)
    cfg["grid"]["morton_sort"] = bool(morton_sort)
    cfg["grid"]["morton_R"] = int(morton_R)

    exp = build_exp_name(hash_size, tag, morton_R, morton_sort)
    cfg["data"]["exp_name"] = exp

    tmp_dir = base_cfg_path.parent
    ensure_parent(tmp_dir)
    tmp_cfg = tmp_dir / f"{scene}_morton_autorun.yaml"
    write_yaml(tmp_cfg, cfg)
    return tmp_cfg


def run_one(cfg_path: Path, global_log: Path, scene: str, hash_size: int, morton_R: int,
            morton_sort: bool, tag: str, python_exec: str) -> None:
    ensure_parent(global_log)

    tegra = find_tegrastats()
    tegra_proc = None
    tegra_tmp = None
    if tegra:
        tegra_tmp = Path(tempfile.mkstemp(prefix="tegrastats_", suffix=".log")[1])
        tegra_cmd = [tegra, "--interval", "1000"]
        tegra_proc = subprocess.Popen(tegra_cmd, stdout=open(tegra_tmp, "w"), stderr=subprocess.DEVNULL)
    else:
        tegra_tmp = None

    start = time.time()
    try:
        cmd = [python_exec, "coslam.py", "--config", str(cfg_path)]
        print(">>> Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        if tegra_proc is not None:
            tegra_proc.terminate()
            try:
                tegra_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                tegra_proc.kill()

    end = time.time()
    runtime = int(round(end - start))

    peak_ram = extract_peak_ram_from_file(tegra_tmp) if tegra_tmp else None
    if tegra_tmp:
        try:
            Path(tegra_tmp).unlink(missing_ok=True)
        except Exception:
            pass

    exp_name = build_exp_name(hash_size, tag, morton_R, morton_sort)

    line = {
        "scene": scene,
        "hash_size": hash_size,
        "tag": tag,
        "morton_sort": morton_sort,
        "morton_R": morton_R,
        "exp_name": exp_name,
        "time_s": runtime,
        "peak_RAM_MB": peak_ram,
    }

    parts = [f"{k}={v}" for k, v in line.items() if v is not None]
    log_line = ", ".join(parts)

    with open(global_log, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

    print(">>> Logged:", log_line)


def main():
    parser = argparse.ArgumentParser(description="Batch-run Replica experiments sweeping hash_size and morton_R.")
    parser.add_argument("--scene", default="office0", help="Scene name (default: office0).")
    parser.add_argument("--scene-cfg", default="configs/Replica/office0.yaml", help="Path to the base scene YAML.")
    parser.add_argument("--tag", default="fp16", help="Short tag to include in exp_name, e.g., fp16 or fp32.")
    parser.add_argument("--hash-min", type=int, default=13, help="Minimum hash size (inclusive).")
    parser.add_argument("--hash-max", type=int, default=20, help="Maximum hash size (inclusive).")
    parser.add_argument("--morton-sort", dest="morton_sort", action="store_true", default=True, help="Enable Morton sorting (default: on).")
    parser.add_argument("--no-morton-sort", dest="morton_sort", action="store_false", help="Disable Morton sorting.")
    parser.add_argument("--morton-Rs", nargs="+", type=int, default=[24, 32, 48, 64, 96, 128], help="List of Morton R values to test.")
    parser.add_argument("--python-exec", default=sys.executable, help="Python executable to run coslam.py.")
    parser.add_argument("--global-log", default="output/Replica/office0/office0_morton_runs.log", help="Single log file path to append results.")
    args = parser.parse_args()

    base_cfg_path = Path(args.scene_cfg).resolve()
    if not base_cfg_path.exists():
        print(f"ERROR: Base scene config not found at: {base_cfg_path}", file=sys.stderr)
        sys.exit(2)

    hvals = list(range(int(args.hash_min), int(args.hash_max) + 1))
    rvals = []
    for r in args.morton_Rs:
        r = int(r)
        if r <= 0:
            continue
        rvals.append(r)

    if not hvals or not rvals:
        print("Nothing to do. Provide --hash-min/--hash-max and --morton-Rs.", file=sys.stderr)
        sys.exit(0)

    global_log = Path(args.global_log)

    for h in hvals:
        for R in rvals:
            tmp_cfg = override_config(
                base_cfg_path=base_cfg_path,
                hash_size=h,
                morton_sort=args.morton_sort,
                morton_R=R,
                tag=args.tag,
                scene=args.scene,
            )
            try:
                run_one(
                    cfg_path=tmp_cfg,
                    global_log=global_log,
                    scene=args.scene,
                    hash_size=h,
                    morton_R=R,
                    morton_sort=args.morton_sort,
                    tag=args.tag,
                    python_exec=args.python_exec,
                )
            finally:
                pass


if __name__ == "__main__":
    main()
