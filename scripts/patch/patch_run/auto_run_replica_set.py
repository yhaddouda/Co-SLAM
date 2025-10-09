#!/usr/bin/env python3
"""
auto_run_replica.py
-------------------
Batch-run CoSLAM experiments on Replica/office0 with automated config overrides.

This version supports explicit patch pairs via --pairs, e.g.:
    --do-patch --pairs 16x1 16x2 16x4 32x1 32x2 32x4

What it does (for each run):
- Writes a temporary YAML config that inherits from your base office0.yaml but overrides:
    - data.exp_name         -> e.g., "12_fp16_16h_1w"
    - grid.hash_size        -> 12..20
    - sampling_tracking.*   -> mode=patch with (ph,pw) and (sh,sw) = (ph,pw)
      (or mode=random if --do-random is requested)

- Launches:  python3 coslam.py --config <temp_config.yaml>
- Measures total runtime in seconds.
- (If available) starts `tegrastats` to capture peak RAM usage.
- Appends ONE line per run to a single global log file (default: output/Replica/office0/office0_runs.log)
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
    """Parse tegrastats output to find the maximum 'RAM <used>/<total>MB' used value."""
    if not logfile.exists():
        return None
    peak = None
    pat = re.compile(r"RAM\s+(\d+)[/ ]")
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
    """Load base scene config, override fields, and write a temp config. Return temp path."""
    cfg = read_yaml(base_cfg_path)
    cfg.setdefault("data", {})
    cfg.setdefault("grid", {})

    if mode == "random":
        cfg["sampling_tracking"] = {"mode": "random"}
        ph = pw = None
    else:
        cfg["sampling_tracking"] = {
            "mode": "patch",
            "ph": int(ph),
            "pw": int(pw),
            "sh": int(ph),
            "sw": int(pw),
        }

    cfg["grid"]["hash_size"] = int(hash_size)

    exp = build_exp_name(hash_size, tag, mode, ph, pw)
    cfg["data"]["exp_name"] = exp

    tmp_cfg = base_cfg_path.parent / f"{scene}_autorun.yaml"
    ensure_parent(tmp_cfg)
    write_yaml(tmp_cfg, cfg)
    return tmp_cfg


def run_one(cfg_path: Path, global_log: Path, scene: str, mode: str, hash_size: int,
            tag: str, ph: int | None, pw: int | None, python_exec: str) -> None:
    """Run one experiment with tegrastats (if available), measure time, log to a single file."""
    ensure_parent(global_log)

    tegra = find_tegrastats()
    tegra_proc = None
    tegra_tmp = None
    if tegra:
        tegra_tmp = Path(tempfile.mkstemp(prefix="tegrastats_", suffix=".log")[1])
        tegra_cmd = [tegra, "--interval", "1000"]
        tegra_proc = subprocess.Popen(tegra_cmd, stdout=open(tegra_tmp, "w"), stderr=subprocess.DEVNULL)

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

    runtime = int(round(time.time() - start))
    peak_ram = extract_peak_ram_from_file(tegra_tmp) if tegra_tmp else None
    if tegra_tmp:
        try:
            tegra_tmp.unlink(missing_ok=True)
        except Exception:
            pass

    exp_name = build_exp_name(hash_size, tag, mode, ph, pw)
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
    parts = [f"{k}={v}" for k, v in line.items() if v is not None]
    with open(global_log, "a", encoding="utf-8") as f:
        f.write(", ".join(parts) + "\n")
    print(">>> Logged:", ", ".join(parts))


def parse_pairs(pair_tokens: list[str]) -> list[tuple[int, int]]:
    """Parse strings like '16x1' into (16,1). Deduplicated and sorted by (ph,pw)."""
    pairs = set()
    for tok in pair_tokens:
        tok = tok.lower().replace(" ", "")
        if "x" not in tok:
            raise ValueError(f"Bad pair '{tok}'. Use format HxW, e.g., 16x1")
        h, w = tok.split("x", 1)
        ph = int(h)
        pw = int(w)
        if ph <= 0 or pw <= 0:
            raise ValueError(f"Non-positive pair '{tok}'")
        pairs.add((ph, pw))
    return sorted(pairs)


def main():
    p = argparse.ArgumentParser(description="Batch-run Replica/office0 experiments with selected patch pairs and hash sizes.")
    p.add_argument("--scene", default="office0", help="Scene name (default: office0).")
    p.add_argument("--scene-cfg", default="configs/Replica/office0.yaml", help="Path to the base scene YAML.")
    p.add_argument("--tag", default="fp16", help="Short tag to include in exp_name, e.g., fp16 or fp32.")
    p.add_argument("--hash-min", type=int, default=12, help="Minimum hash size (inclusive).")
    p.add_argument("--hash-max", type=int, default=20, help="Maximum hash size (inclusive).")
    p.add_argument("--do-random", action="store_true", help="Include random sampling runs.")
    p.add_argument("--do-patch", action="store_true", help="Include patch sampling runs.")
    p.add_argument("--pairs", nargs="+", default=None, help="Explicit patch pairs HxW (e.g., 16x1 16x2 16x4 32x1 32x2 32x4). If set, --dims is ignored.")
    p.add_argument("--dims", nargs="+", type=int, default=[1,2,4,8,16,32], help="(Legacy) dims to combine for ph×pw when --pairs is not set.")
    p.add_argument("--python-exec", default=sys.executable, help="Python executable to run coslam.py (default: current interpreter).")
    p.add_argument("--global-log", default="output/Replica/office0/office0_runs.log", help="Single log file path to append results.")
    args = p.parse_args()

    base_cfg_path = Path(args.scene_cfg).resolve()
    if not base_cfg_path.exists():
        print(f"ERROR: Base scene config not found at: {base_cfg_path}", file=sys.stderr)
        sys.exit(2)

    hvals = list(range(int(args.hash_min), int(args.hash_max) + 1))
    runs = []

    if args.do_random:
        for h in hvals:
            runs.append(("random", None, None, h))

    if args.do_patch:
        if args.pairs:
            patch_pairs = parse_pairs(args.pairs)
        else:
            dims = sorted(set(int(d) for d in args.dims))
            patch_pairs = [(ph, pw) for ph in dims for pw in dims]
        for ph, pw in patch_pairs:
            for h in hvals:
                runs.append(("patch", ph, pw, h))

    if not runs:
        print("Nothing to do. Add --do-patch with --pairs, or --do-random.", file=sys.stderr)
        sys.exit(0)

    global_log = Path(args.global_log)
    for (mode, ph, pw, hsize) in runs:
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
            pass


if __name__ == "__main__":
    main()
