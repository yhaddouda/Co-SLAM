#!/usr/bin/env python3
"""
auto_run_office2_three_setups.py
--------------------------------
Batch-run CoSLAM experiments on Replica/office2 by sweeping table sizes 13..20
for FOUR setups (per your spec):

  A) CoherentPrime hash fn, morton_sort = False
  B) Morton hash fn,       morton_sort = False      <-- exp_name includes "hashfn_morton"
  C) CoherentPrime hash fn, morton_sort = True
  D) Morton hash fn,        morton_sort = True      <-- exp_name includes "hashfn_morton_sort"

We keep the same dynamic as your previous script: for every run we write a *temporary*
YAML that overrides the scene config (office2 inherits from replica.yaml).

NOTE: We do NOT sweep R; it's fixed to 128 for all runs.

Fixed flags we set in each temp YAML (unless your base overrides differently elsewhere):
    grid:
      enc: 'HashGrid'
      tcnn_encoding: True
      hash_size: <13..20>
      voxel_color: 0.08
      voxel_sdf:   0.02
      oneGrid: True
      hash: <CoherentPrime | Morton>
      morton_sort: <True | False>
      morton_R: 128

exp_name naming:
  - Setup A  (prime, no sort):          "{size}_{tag}_noMorton_R128"
  - Setup B  (Morton fn, no sort):      "{size}_{tag}_hashfn_morton_R128"
  - Setup C  (prime, morton sort True): "{size}_{tag}_R128"
  - Setup D  (Morton fn, morton sort):  "{size}_{tag}_hashfn_morton_sort_R128"

We append one line per run to a single global log file.

Usage (example):
  python3 auto_run_office2_three_setups.py \
      --scene office2 \
      --scene-cfg configs/Replica/office2.yaml \
      --tag fp16 \
      --hash-min 13 --hash-max 20 \
      --python-exec python3 \
      --global-log output/Replica/office2/office2_three_setups.log
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
    if not logfile or not Path(logfile).exists():
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


def build_exp_name(setup: str, size: int, tag: str) -> str:
    """
    setup in {"A", "B", "C", "D"}:
      A: prime + no morton sort
      B: morton hash fn + no morton sort  (add 'hashfn_morton')
      C: prime + morton sort
      D: morton hash fn + morton sort     (add 'hashfn_morton_sort')
    """
    if setup == "A":
        return f"{size}_{tag}_noMorton_R128"
    elif setup == "B":
        return f"{size}_{tag}_hashfn_morton_R128"
    elif setup == "C":
        return f"{size}_{tag}_R128"
    elif setup == "D":
        return f"{size}_{tag}_hashfn_morton_sort_R128"
    else:
        raise ValueError(f"Unknown setup: {setup}")


def override_config(base_cfg_path: Path, scene: str, size: int, setup: str, tag: str) -> Path:
    """
    Return path to a temp config for the given (size, setup).
    """
    cfg = read_yaml(base_cfg_path)

    # Ensure nodes exist
    cfg.setdefault("data", {})
    cfg.setdefault("grid", {})

    # Fixed fields (as requested)
    g = cfg["grid"]
    g["enc"] = "HashGrid"
    g["tcnn_encoding"] = True
    g["hash_size"] = int(size)
    g["voxel_color"] = 0.08
    g["voxel_sdf"] = 0.02
    g["oneGrid"] = True
    g["morton_R"] = 128  # fixed; no sweep of R

    # Switch per setup
    if setup == "A":
        g["hash"] = "CoherentPrime"
        g["morton_sort"] = False
    elif setup == "B":
        g["hash"] = "Morton"
        g["morton_sort"] = False
    elif setup == "C":
        g["hash"] = "CoherentPrime"
        g["morton_sort"] = True
    elif setup == "D":
        g["hash"] = "Morton"
        g["morton_sort"] = True
    else:
        raise ValueError(f"Unknown setup: {setup}")

    # exp_name per setup rule
    exp = build_exp_name(setup, size, tag)
    cfg["data"]["exp_name"] = exp

    # Write temp YAML next to base
    tmp_dir = base_cfg_path.parent
    ensure_parent(tmp_dir)
    tmp_cfg = tmp_dir / f"{scene}_autorun_{setup}.yaml"
    write_yaml(tmp_cfg, cfg)
    return tmp_cfg


def run_one(cfg_path: Path, global_log: Path, scene: str, size: int, setup: str, tag: str, python_exec: str) -> None:
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

    exp_name = build_exp_name(setup, size, tag)

    line = {
        "scene": scene,
        "hash_size": size,
        "tag": tag,
        "setup": setup,
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
    parser = argparse.ArgumentParser(description="Batch-run Replica/office2 sweeping hash_size for 4 setups.")
    parser.add_argument("--scene", default="office2", help="Scene name (default: office2).")
    parser.add_argument("--scene-cfg", default="configs/Replica/office2.yaml", help="Path to the base scene YAML (inherits from replica.yaml).")
    parser.add_argument("--tag", default="fp16", help="Short tag to include in exp_name, e.g., fp16 or fp32.")
    parser.add_argument("--hash-min", type=int, default=13, help="Minimum hash size (inclusive).")
    parser.add_argument("--hash-max", type=int, default=20, help="Maximum hash size (inclusive).")
    parser.add_argument("--python-exec", default=sys.executable, help="Python executable to run coslam.py.")
    parser.add_argument("--global-log", default="output/Replica/office2/office2_three_setups.log", help="Single log file path to append results.")
    args = parser.parse_args()

    base_cfg_path = Path(args.scene_cfg).resolve()
    if not base_cfg_path.exists():
        print(f"ERROR: Base scene config not found at: {base_cfg_path}", file=sys.stderr)
        sys.exit(2)

    sizes = list(range(int(args.hash_min), int(args.hash_max) + 1))
    if not sizes:
        print("Nothing to do. Provide --hash-min/--hash-max.", file=sys.stderr)
        sys.exit(0)

    global_log = Path(args.global_log)

    # A, B, C, D in this fixed order for each size
    setups = ["A", "B", "C", "D"]

    for size in sizes:
        for setup in setups:
            tmp_cfg = override_config(
                base_cfg_path=base_cfg_path,
                scene=args.scene,
                size=size,
                setup=setup,
                tag=args.tag,
            )
            try:
                run_one(
                    cfg_path=tmp_cfg,
                    global_log=global_log,
                    scene=args.scene,
                    size=size,
                    setup=setup,
                    tag=args.tag,
                    python_exec=args.python_exec,
                )
            finally:
                # leave temp YAML in place for reproducibility; comment next two lines to keep
                # try:
                #     Path(tmp_cfg).unlink(missing_ok=True)
                # except Exception:
                #     pass
                pass


if __name__ == "__main__":
    main()