#!/usr/bin/env python3
"""
auto_run_replica.py
-------------------
Batch-run CoSLAM experiments on Replica/office0 with automated config overrides.

What it does (for each run):
- Writes a temporary YAML config that inherits from your base office0.yaml but overrides:
    - data.exp_name         -> e.g., "12_fp16_1h_16w" or "12_fp16_random"
    - grid.hash_size        -> 12..20
    - sampling_tracking.*   -> mode=random OR mode=patch with (ph,pw) and (sh,sw) = (ph,pw)

- Launches:  python3 coslam.py --config <temp_config.yaml>
- Measures total runtime in seconds.
- (If available) starts `tegrastats` to capture peak RAM usage.
- Appends ONE line per run to a single global log file (default: output/Replica/office0/office0_runs.log)

Usage example:
    python3 auto_run_replica.py --scene office0 --tag fp16 --hash-min 12 --hash-max 20 \
        --do-random --do-patch --dims 1 2 4 8 16 32

Notes:
- Run this from the project root (so relative config paths resolve correctly).
- Requires PyYAML (pip install pyyaml). If missing, the script will prompt you.
- Assumes your base scene config is at: ./configs/Replica/office0.yaml which inherits from ./configs/Replica/replica.yaml
- On Jetson, `tegrastats` should exist; if not, peak RAM will be omitted gracefully.
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
    # Common locations; rely on PATH if not found here.
    candidates = ["/usr/bin/tegrastats", "/bin/tegrastats", "/usr/sbin/tegrastats", "tegrastats"]
    for c in candidates:
        if shutil.which(c):
            return shutil.which(c)
    return None


def extract_peak_ram_from_file(logfile: Path) -> int | None:
    """
    Parse tegrastats output to find the maximum "RAM <used>/<total>MB" used value.
    Returns MB (int) or None if not found.
    """
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


def build_exp_name(hash_size: int, tag: str, mode: str, ph: int | None, pw: int | None) -> str:
    if mode == "random":
        return f"{hash_size}_{tag}_random"
    else:
        return f"{hash_size}_{tag}_{ph}h_{pw}w"


def override_config(base_cfg_path: Path, hash_size: int, mode: str, ph: int | None, pw: int | None,
                    tag: str, scene: str) -> Path:
    """
    Load base scene config (e.g., configs/Replica/office0.yaml), override relevant fields,
    and write to a temporary config file. Return the temp path.
    """
    cfg = read_yaml(base_cfg_path)

    # Ensure subtrees exist
    cfg.setdefault("data", {})
    cfg.setdefault("grid", {})

    # sampling_tracking block
    if mode == "random":
        cfg["sampling_tracking"] = {"mode": "random"}
        ph = pw = None  # sanitize for exp_name formatting
    else:
        # PATCH mode with sh/sw locking to ph/pw
        cfg["sampling_tracking"] = {
            "mode": "patch",
            "ph": int(ph),
            "pw": int(pw),
            "sh": int(ph),
            "sw": int(pw),
        }

    # grid.hash_size override
    cfg["grid"]["hash_size"] = int(hash_size)

    # data.exp_name
    exp = build_exp_name(hash_size, tag, mode, ph, pw)
    cfg["data"]["exp_name"] = exp

    # Write to a temp file next to the base cfg for easy relative paths
    tmp_dir = base_cfg_path.parent
    ensure_parent(tmp_dir)
    tmp_cfg = tmp_dir / f"{scene}_autorun.yaml"
    write_yaml(tmp_cfg, cfg)
    return tmp_cfg


def run_one(cfg_path: Path, global_log: Path, scene: str, mode: str, hash_size: int,
            tag: str, ph: int | None, pw: int | None, python_exec: str) -> None:
    """Run one experiment with tegrastats (if available), measure time, log to a single file."""
    ensure_parent(global_log)

    # Start tegrastats if available
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
        # Launch CoSLAM
        cmd = [python_exec, "coslam.py", "--config", str(cfg_path)]
        print(">>> Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        # Stop tegrastats
        if tegra_proc is not None:
            tegra_proc.terminate()
            try:
                tegra_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                tegra_proc.kill()

    end = time.time()
    runtime = int(round(end - start))

    # Parse peak RAM
    peak_ram = extract_peak_ram_from_file(tegra_tmp) if tegra_tmp else None
    if tegra_tmp:
        try:
            tegra_tmp.unlink(missing_ok=True)
        except Exception:
            pass

    # Build exp_name again for logging
    exp_name = build_exp_name(hash_size, tag, mode, ph, pw)

    # Log one line with all parameters
    line = {
        "scene": scene,
        "mode": mode,
        "ph": ph,
        "pw": pw,
        "sh": ph if ph is not None else None,
        "sw": pw if pw is not None else None,
        "hash_size": hash_size,
        "tag": tag,
        "exp_name": exp_name,
        "time_s": runtime,
        "peak_RAM_MB": peak_ram,
    }

    # Compact key=value CSV-ish line
    parts = []
    for k, v in line.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    log_line = ", ".join(parts)

    with open(global_log, "a", encoding="utf-8") as f:
        f.write(log_line + "\n")

    print(">>> Logged:", log_line)


def main():
    parser = argparse.ArgumentParser(description="Batch-run Replica/office0 experiments with patch sampling and varying hash sizes.")
    parser.add_argument("--scene", default="office0", help="Scene name (default: office0).")
    parser.add_argument("--scene-cfg", default="configs/Replica/office0.yaml", help="Path to the base scene YAML.")
    parser.add_argument("--tag", default="fp16", help="Short tag to include in exp_name, e.g., fp16 or fp32.")
    parser.add_argument("--hash-min", type=int, default=12, help="Minimum hash size (inclusive).")
    parser.add_argument("--hash-max", type=int, default=20, help="Maximum hash size (inclusive).")
    parser.add_argument("--dims", nargs="+", type=int, default=[1,2,4,8,16,32], help="Patch dimensions to combine for ph×pw (only used for --do-patch).")
    parser.add_argument("--do-random", action="store_true", help="Include random sampling runs.")
    parser.add_argument("--do-patch", action="store_true", help="Include patch sampling runs for all ph×pw combos.")
    parser.add_argument("--python-exec", default=sys.executable, help="Python executable to run coslam.py (default: current interpreter).")
    parser.add_argument("--global-log", default="output/Replica/office0/office0_runs.log", help="Single log file path to append results.")
    args = parser.parse_args()

    base_cfg_path = Path(args.scene_cfg).resolve()
    if not base_cfg_path.exists():
        print(f"ERROR: Base scene config not found at: {base_cfg_path}", file=sys.stderr)
        sys.exit(2)

    # Build list of runs
    runs = []

    # Hash sizes
    hvals = list(range(int(args.hash_min), int(args.hash_max) + 1))

    if args.do_random:
        for h in hvals:
            runs.append(("random", None, None, h))

    if args.do_patch:
        dims = sorted(set(int(d) for d in args.dims))
        for ph in dims:
            for pw in dims:
                for h in hvals:
                    runs.append(("patch", ph, pw, h))

    if not runs:
        print("Nothing to do. Add --do-random and/or --do-patch.", file=sys.stderr)
        sys.exit(0)

    # Iterate
    global_log = Path(args.global_log)
    for (mode, ph, pw, hsize) in runs:
        # Create a temp override config for this run
        tmp_cfg = override_config(
            base_cfg_path=base_cfg_path,
            hash_size=hsize,
            mode=mode,
            ph=ph,
            pw=pw,
            tag=args.tag,
            scene=args.scene,
        )
        try:
            run_one(
                cfg_path=tmp_cfg,
                global_log=global_log,
                scene=args.scene,
                mode=mode,
                hash_size=hsize,
                tag=args.tag,
                ph=ph,
                pw=pw,
                python_exec=args.python_exec,
            )
        finally:
            # Keep the last tmp config around for debugging; if you prefer, uncomment the deletion:
            # try: tmp_cfg.unlink(missing_ok=True)
            # except Exception: pass
            pass


if __name__ == "__main__":
    main()
