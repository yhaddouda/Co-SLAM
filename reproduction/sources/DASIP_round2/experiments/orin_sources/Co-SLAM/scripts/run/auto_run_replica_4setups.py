#!/usr/bin/env python3
"""
auto_run_replica_multi_scenes.py
--------------------------------
Batch-run CoSLAM experiments across *multiple Replica scenes* by sweeping table
sizes 16..23 for FOUR setups:

  A) CoherentPrime hash fn, morton_sort = False
  B) Morton hash fn,       morton_sort = False      <-- exp_name includes "hashfn_morton"
  C) CoherentPrime hash fn, morton_sort = True
  D) Morton hash fn,        morton_sort = True      <-- exp_name includes "hashfn_morton_sort"

Key points:
- We now run over *all scenes in configs/Replica* (or a custom root),
  automatically discovering scene .yaml files.
- We EXCLUDE **only** the scene: office2
- We do NOT sweep R; it's fixed to 128 for all runs.
- Hash sizes default to 16..23 inclusive.

For each run we create a *temporary* YAML that overrides the scene config.
(Each scene config is expected to inherit from replica.yaml as in your setup.)

Fixed flags in the temp YAML:
    grid:
      enc: 'HashGrid'
      tcnn_encoding: True
      hash_size: <16..23>
      voxel_color: 0.08
      voxel_sdf:   0.02
      oneGrid: True
      hash: <CoherentPrime | Morton>
      morton_sort: <True | False>
      morton_R: 128   # fixed

exp_name naming:
  - Setup A  (prime, no sort):          "{size}_{tag}_noMorton_R128"
  - Setup B  (Morton fn, no sort):      "{size}_{tag}_hashfn_morton_R128"
  - Setup C  (prime, morton sort True): "{size}_{tag}_R128"
  - Setup D  (Morton fn, morton sort):  "{size}_{tag}_hashfn_morton_sort_R128"

We append one line per run to a single global log file.

Usage (examples):

  # Auto-discover all scenes (excluding office2 only)
  python3 auto_run_replica_multi_scenes.py \
      --configs-root configs/Replica \
      --tag fp16 \
      --global-log output/Replica/all_scenes/replica_multi_scenes.log

  # Run a custom subset of scenes
  python3 auto_run_replica_multi_scenes.py \
      --configs-root configs/Replica \
      --scenes office0 office1 room0 room1 room2 \
      --tag fp16

  # Single-scene fallback still works
  python3 auto_run_replica_multi_scenes.py \
      --scene office5 \
      --scene-cfg configs/Replica/office5.yaml \
      --tag fp16
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


# Exclude only office2 by default
DEFAULT_EXCLUDES = {"office2"}


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


def discover_scenes(configs_root: Path, excludes: set[str]) -> list[str]:
    """Return list of scene names by scanning configs_root for *.yaml (excluding replica.yaml and listed excludes)."""
    scenes = []
    if not configs_root.exists():
        return scenes
    for yml in sorted(configs_root.glob("*.yaml")):
        scene = yml.stem
        if scene == "replica":
            continue
        if scene in excludes:
            continue
        scenes.append(scene)
    return scenes


def run_for_scene(scene: str, base_cfg_path: Path, sizes: list[int], setups: list[str],
                  tag: str, python_exec: str, global_log: Path) -> None:
    for size in sizes:
        for setup in setups:
            tmp_cfg = override_config(
                base_cfg_path=base_cfg_path,
                scene=scene,
                size=size,
                setup=setup,
                tag=tag,
            )
            try:
                run_one(
                    cfg_path=tmp_cfg,
                    global_log=global_log,
                    scene=scene,
                    size=size,
                    setup=setup,
                    tag=tag,
                    python_exec=python_exec,
                )
            finally:
                # keep temp YAMLs for reproducibility
                pass


def main():
    parser = argparse.ArgumentParser(description="Batch-run Replica multi-scenes sweeping hash_size for 4 setups.")
    # Single-scene fallback args
    parser.add_argument("--scene", default=None, help="If provided, run only this scene (e.g., office2).")
    parser.add_argument("--scene-cfg", default=None, help="Path to the base scene YAML for single-scene mode.")
    # Multi-scene discovery args
    parser.add_argument("--configs-root", default="configs/Replica", help="Folder containing scene YAMLs.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional explicit list of scenes to run (overrides discovery).")
    parser.add_argument("--exclude", nargs="*", default=[], help="Extra scene names to exclude, in addition to defaults.")
    # Hash range and misc
    parser.add_argument("--tag", default="fp16", help="Short tag to include in exp_name, e.g., fp16 or fp32.")
    parser.add_argument("--hash-min", type=int, default=16, help="Minimum hash size (inclusive).")
    parser.add_argument("--hash-max", type=int, default=23, help="Maximum hash size (inclusive).")
    parser.add_argument("--python-exec", default=sys.executable, help="Python executable to run coslam.py.")
    parser.add_argument("--global-log", default="output/Replica/all_scenes/replica_multi_scenes.log", help="Single log file path to append results.")
    args = parser.parse_args()

    # Build size list (defaults to 16..23 per request)
    sizes = list(range(int(args.hash_min), int(args.hash_max) + 1))
    if not sizes:
        print("Nothing to do. Provide --hash-min/--hash-max.", file=sys.stderr)
        sys.exit(0)

    global_log = Path(args.global_log)
    ensure_parent(global_log)

    setups = ["A", "B", "C", "D"]

    # Determine mode
    if args.scene:
        # Single-scene mode
        if not args.scene_cfg:
            cfg_path = Path(args.configs_root) / f"{args.scene}.yaml"
        else:
            cfg_path = Path(args.scene_cfg)
        cfg_path = cfg_path.resolve()
        if not cfg_path.exists():
            print(f"ERROR: Base scene config not found at: {cfg_path}", file=sys.stderr)
            sys.exit(2)
        run_for_scene(args.scene, cfg_path, sizes, setups, args.tag, args.python_exec, global_log)
        return

    # Multi-scene mode (discover or use explicit list)
    configs_root = Path(args.configs_root).resolve()

    if args.scenes:
        scenes = args.scenes
    else:
        excludes = set(map(str, DEFAULT_EXCLUDES)) | set(map(str, args.exclude))
        scenes = discover_scenes(configs_root, excludes)

    if not scenes:
        print("No scenes selected. Check configs root and excludes.", file=sys.stderr)
        sys.exit(0)

    print(f">>> Scenes to run ({len(scenes)}): {', '.join(scenes)}")

    for scene in scenes:
        cfg_path = (configs_root / f"{scene}.yaml").resolve()
        if not cfg_path.exists():
            print(f"WARNING: Skipping {scene} (missing config at {cfg_path})", file=sys.stderr)
            continue
        run_for_scene(scene, cfg_path, sizes, setups, args.tag, args.python_exec, global_log)


if __name__ == "__main__":
    main()