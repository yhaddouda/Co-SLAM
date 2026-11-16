#!/usr/bin/env python3
"""Plot baseline versus pruning execution time across Replica scenes."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("output/Replica_frame_total_modes_execution_times.csv")
DEFAULT_OUTPUT = Path("output/Replica_frame_total_modes_execution_times.png")
MODES = ("baseline", "pruning")


def natural_scene_key(scene):
    prefix = scene.rstrip("0123456789")
    suffix = scene[len(prefix) :]
    return prefix, int(suffix) if suffix else -1


def load_times(csv_path):
    values = defaultdict(lambda: defaultdict(list))

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"Mode", "Scene", "Time_Seconds"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path} is missing required columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            mode = row["Mode"].strip().lower()
            if mode in MODES:
                values[row["Scene"].strip()][mode].append(float(row["Time_Seconds"]))

    complete_scenes = [
        scene for scene, mode_values in values.items() if all(mode_values[mode] for mode in MODES)
    ]
    if not complete_scenes:
        raise ValueError(f"No scenes with both baseline and pruning data found in {csv_path}")

    return sorted(complete_scenes, key=natural_scene_key), values


def plot(csv_path, output_path):
    scenes, values = load_times(csv_path)
    baseline = np.array([np.mean(values[scene]["baseline"]) for scene in scenes])
    pruning = np.array([np.mean(values[scene]["pruning"]) for scene in scenes])
    speedups = (baseline - pruning) / baseline * 100.0

    x = np.arange(len(scenes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.6))

    baseline_bars = ax.bar(
        x - width / 2,
        baseline,
        width,
        label="Baseline",
        color="#4C78A8",
    )
    pruning_bars = ax.bar(
        x + width / 2,
        pruning,
        width,
        label="Pruning",
        color="#F58518",
    )

    ax.set_title("Replica execution time: baseline vs. pruning", pad=14)
    ax.set_ylabel("Execution time (seconds, lower is better)")
    ax.set_xlabel("Replica scene")
    ax.set_xticks(x, scenes)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc="upper center")

    max_time = max(baseline.max(), pruning.max())
    ax.set_ylim(0, max_time * 1.16)

    for bars, times in ((baseline_bars, baseline), (pruning_bars, pruning)):
        for bar, time_s in zip(bars, times):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_time * 0.012,
                f"{time_s:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    for bar, speedup in zip(pruning_bars, speedups):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - max_time * 0.045,
            f"{speedup:.1f}% faster",
            ha="center",
            va="top",
            rotation=90,
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    aggregate_speedup = (baseline.sum() - pruning.sum()) / baseline.sum() * 100.0
    print(f"Saved {output_path}")
    print(f"Saved {pdf_path}")
    print(f"Aggregate pruning speedup across plotted scenes: {aggregate_speedup:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and pruning execution times across Replica scenes."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.input, args.output)


if __name__ == "__main__":
    main()
