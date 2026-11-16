#!/usr/bin/env python3
"""Build report figures and tables from the saved Replica experiments.

This reads existing measurements only; it does not launch SLAM experiments.
Run this file, then pdflatex twice in its directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/coslam-report-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FULL = ROOT / "results/ba_experiments/full"
SCENES = ("office1", "room1", "office0")
COLORS = {"baseline": "#3e5266", "baseline_fast": "#008879"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(scene: str, variant: str):
    directory = FULL / scene / f"{variant}_seed0"
    summary = json.loads((directory / "summary.json").read_text())
    with (directory / "frames.csv").open() as stream:
        frames = [
            {"frame": int(r["frame_id"]), "wall": float(r["wall_ms"]),
             "ba": bool(int(r["has_ba"])), "ba_wall": float(r["ba_wall_ms"])}
            for r in csv.DictReader(stream)
        ]
    return directory, summary, frames


def main():
    with (FULL / "comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    paired = {(r["scene"], r["variant"]): r for r in rows}
    assert all(int(r["frames"]) == 2000 and int(r["seed"]) == 0 for r in rows)
    tables = []
    manifest = {"inputs": {}, "source_hashes": {}, "derived": {}}
    inputs = [FULL / "comparison.csv", ROOT / "results/ba_experiments/gpu_equivalence_fast.json"]

    def macro(name, value):
        tables.append("\\newcommand{\\" + name + "}{" + str(value) + "}")

    def table_macro(name, lines):
        macro(name, "\n" + "\n".join(lines) + "\n")

    def num(row, key):
        return float(row[key])

    full_rows, timing_rows, whole_rows, curve_rows = [], [], [], []
    data = {}
    for scene in SCENES:
        b, f = paired[scene, "baseline"], paired[scene, "baseline_fast"]
        full_rows.append(
            f"{scene} & {num(b, 'ate_mm'):.3f} & {num(f, 'ate_mm'):.3f} & "
            f"{num(f, 'ate_change_mm'):+.3f} & {num(f, 'ate_change_pct'):+.2f}\\% \\\\"
        )
        timing_rows.append(
            f"{scene} & {num(b, 'ba_mean_ms'):.2f} & {num(f, 'ba_mean_ms'):.2f} & "
            f"{num(f, 'ba_reduction_pct'):.2f}\\% & {num(b, 'ba_frame_mean_ms'):.2f} & "
            f"{num(f, 'ba_frame_mean_ms'):.2f} & {num(f, 'ba_frame_reduction_pct'):.2f}\\% \\\\"
        )
        whole = []
        for variant in ("baseline", "baseline_fast"):
            directory, summary, frames = read_run(scene, variant)
            inputs.extend(directory / name for name in ("summary.json", "frames.csv", "config.json"))
            data[scene, variant] = summary, frames
            measured = [r for r in frames if r["frame"] >= summary["warmup_frames"]]
            ba = [r for r in measured if r["ba"] and r["frame"] >= 100]
            assert len(measured) == 1980 and sum(r["ba"] for r in measured) == 396
            first, last = statistics.mean(r["ba_wall"] for r in ba[:50]), statistics.mean(r["ba_wall"] for r in ba[-50:])
            variant_label = variant.replace("_", "\\_")
            curve_rows.append(f"{scene} & {variant_label} & {first:.2f} & {last:.2f} \\\\")
            whole.append((statistics.mean(r["wall"] for r in measured), summary["run_wall_seconds"]))
        reduction = 100 * (1 - whole[1][0] / whole[0][0])
        whole_rows.append(
            f"{scene} & {whole[0][0]:.2f} & {whole[1][0]:.2f} & {reduction:.2f}\\% & "
            f"{whole[0][1]:.2f} & {whole[1][1]:.2f} \\\\"
        )
        manifest["derived"][scene] = {"all_frame_time_reduction_pct": reduction}

    table_macro("FullResults", full_rows)
    table_macro("TimingResults", timing_rows)
    table_macro("WholeRunResults", whole_rows)
    table_macro("CurveResults", curve_rows)
    summary_rows = []
    for variant in ("baseline_fast", "bucket", "lattice_guarded", "lattice_guarded_fast"):
        selected = [paired[scene, variant] for scene in SCENES]
        changes = [num(r, "ate_change_pct") for r in selected]
        mean_ate = statistics.mean(changes)
        mean_ba = statistics.mean(num(r, "ba_reduction_pct") for r in selected)
        mean_frame = statistics.mean(num(r, "ba_frame_reduction_pct") for r in selected)
        variant_label = variant.replace("_", "\\_")
        summary_rows.append(
            f"{variant_label} & {mean_ate:+.2f}\\% & {max(changes):+.2f}\\% & "
            f"{mean_ba:+.2f}\\% & {mean_frame:+.2f}\\% \\\\"
        )
        if variant == "baseline_fast":
            macro("MeanATE", f"{mean_ate:+.2f}\\%")
            macro("MeanBASaving", f"{mean_ba:.2f}\\%")
            macro("MeanFrameSaving", f"{mean_frame:.2f}\\%")
            macro("WorstATE", f"{max(changes):+.2f}\\%")
            macro("MeanBAFactor", f"{1 / (1 - mean_ba / 100):.2f}")
    table_macro("VariantResults", summary_rows)
    micro = json.loads(inputs[1].read_text())["pose_commit_equivalence"]
    macro("MicroOld", f"{micro['original_singleton_commit_ms']:.2f}")
    macro("MicroFast", f"{micro['cached_batch_commit_ms']:.2f}")
    macro("MatrixDifference", f"{micro['maximum_matrix_difference']:.2e}")
    macro("SingletonCount", f"{sum(range(2, 400)):,}")
    (HERE / "derived.tex").write_text("% Generated from saved experiment files.\n" + "\n".join(tables) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True, constrained_layout=True)
    for ax, scene in zip(axes, SCENES):
        for variant, label in (("baseline", "Dense baseline"), ("baseline_fast", "Dense baseline_fast")):
            summary, frames = data[scene, variant]
            ba = [r for r in frames if r["ba"] and r["frame"] >= summary["warmup_frames"]]
            x, y = np.array([r["frame"] for r in ba]), np.array([r["ba_wall"] for r in ba])
            median = np.array([np.median(y[max(0, i - 10):min(len(y), i + 11)]) for i in range(len(y))])
            ax.plot(x, y, color=COLORS[variant], alpha=.20, lw=.6)
            ax.plot(x, median, color=COLORS[variant], lw=1.8, label=label)
        ax.set(title=scene, xlabel="Frame index", xlim=(20, 1995), ylim=(250, 1120))
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Complete BA invocation (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(HERE / "runtime_growth.pdf")
    fig.savefig(HERE / "runtime_growth.png", dpi=180)
    plt.close(fig)

    # Record the exact result inputs and tested fast source, so regeneration is auditable.
    for path in inputs:
        manifest["inputs"][str(path.relative_to(ROOT))] = digest(path)
    source = FULL / "office1/baseline_fast_seed0/source_hashes.json"
    hashes = json.loads(source.read_text())
    core = ("coslam.py", "optimization/ba_experiments.py", "optimization/utils.py", "model/keyframe.py", "model/scene_rep.py", "model/utils.py")
    for scene in SCENES:
        for variant in ("baseline_fast", "lattice_guarded_fast"):
            path = FULL / scene / f"{variant}_seed0/source_hashes.json"
            other = json.loads(path.read_text())
            assert all(other[name] == hashes[name] for name in core)
    for name in core:
        snapshot = source.parent / "source" / name
        assert digest(snapshot) == hashes[name]
        manifest["source_hashes"][name] = {"tested_sha256": hashes[name], "current_sha256": digest(ROOT / name)}
    manifest["note"] = "Tables/plots regenerate from saved seed-0 results; no experiments were rerun."
    (HERE / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Generated derived.tex, runtime_growth.pdf/png and evidence_manifest.json")


if __name__ == "__main__":
    main()
