#!/usr/bin/env python3
"""
Plot time (s) vs hash table size for four setups (A–D) from office2 log.

Usage:
  python plot_office2_four_setups.py office2_three_setups.log [output.png]

If output path is omitted, saves: office2_four_setups_plot.png
"""
import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path

LABELS = {
    "A": "A — CoherentPrime (no Morton sort)",
    "B": "B — Morton hash",
    "C": "C — CoherentPrime + Morton sort (R=128)",
    "D": "D — Morton hash + Morton sort (R=128)",
}
MARKERS = {"A": "o", "B": "s", "C": "^", "D": "d"}

def plot_four_setups(data_file: str, out_png: str, scene_label: str = "office2"):
    series = {"A": {"sizes": [], "times": []},
              "B": {"sizes": [], "times": []},
              "C": {"sizes": [], "times": []},
              "D": {"sizes": [], "times": []}}

    # Capture: hash_size, setup (A–D), time_s
    pattern = re.compile(r"hash_size=(\d+).*?setup=([ABCD]).*?time_s=([\d.]+)", re.IGNORECASE)

    with open(data_file, "r") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            size = int(m.group(1))
            setup = m.group(2).upper()
            time_s = float(m.group(3))
            series[setup]["sizes"].append(size)
            series[setup]["times"].append(time_s)

    # Sort each setup by size
    for s in series.values():
        order = sorted(range(len(s["sizes"])), key=lambda i: s["sizes"][i])
        s["sizes"] = [s["sizes"][i] for i in order]
        s["times"] = [s["times"][i] for i in order]

    # Plot (default matplotlib colors; markers differentiate lines)
    plt.figure()
    for key in ["A", "B", "C", "D"]:
        if series[key]["sizes"]:
            plt.plot(series[key]["sizes"],
                     series[key]["times"],
                     marker=MARKERS[key],
                     label=LABELS[key])
    plt.xlabel("Hash table size")
    plt.ylabel("Time (s)")
    plt.title(f"{scene_label} — Time vs Hash Table Size (A–D setups)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) >= 2 else "office2_three_setups.log"
    out_path = sys.argv[2] if len(sys.argv) >= 3 else "office2_four_setups_plot.png"
    plot_four_setups(in_path, out_path)
