#!/usr/bin/env python3
"""
Plot time (s) vs hash table size for Morton (R=128) vs No Morton.

Usage:
  python plot_time_morton_vs_off_R128.py time_morton_vs_off_R128.txt [output.png]

If output path is omitted, saves: time_morton_vs_off_R128_plot.png
"""
import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def plot_time_vs_size(data_file: str, out_png: str, scene_label: str = "office0"):
    morton = {'sizes': [], 'times': []}
    normal = {'sizes': [], 'times': []}

    # Capture: hash_size, morton_sort, time_s
    pattern = re.compile(r"hash_size=(\d+).*?morton_sort=(True|False).*?time_s=(\d+)", re.IGNORECASE)

    with open(data_file, "r") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            size = int(m.group(1))
            is_morton = (m.group(2) == "True")
            time_s = float(m.group(3))
            target = morton if is_morton else normal
            target['sizes'].append(size)
            target['times'].append(time_s)

    # Sort by size for clean lines
    for d in (morton, normal):
        order = sorted(range(len(d['sizes'])), key=lambda i: d['sizes'][i])
        d['sizes'] = [d['sizes'][i] for i in order]
        d['times'] = [d['times'][i] for i in order]

    # Plot (no custom colors; markers only)
    plt.figure()
    if normal['sizes']:
        plt.plot(normal['sizes'], normal['times'], marker="o", label="No Morton")
    if morton['sizes']:
        plt.plot(morton['sizes'], morton['times'], marker="s", label="Morton (R=128)")
    plt.xlabel("Hash table size")
    plt.ylabel("Time (s)")
    plt.title(f"{scene_label} — Time vs Hash Table Size (Morton R=128) vs No Morton on RTX4070")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) >= 2 else "time_morton_vs_off_R128.txt"
    out_path = sys.argv[2] if len(sys.argv) >= 3 else "time_morton_vs_off_R128_plot.png"
    plot_time_vs_size(in_path, out_path)
