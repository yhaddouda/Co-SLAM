#!/usr/bin/env python3
"""
Plot ATE RMSE vs hash table size for Morton (R=128) vs No Morton.

Usage:
  python plot_ate_rmse_morton_vs_off_R128.py ate_rmse_morton_vs_off_R128.txt [output.png]
"""
import re
import sys
import matplotlib.pyplot as plt

FLOAT_RE = r"[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?"

def plot_ate_vs_size(data_file: str, out_png: str, scene_label: str = "office0"):
    morton = {'sizes': [], 'vals': []}
    normal = {'sizes': [], 'vals': []}

    # Robust patterns: accept "ate.rmse" or "ate_rmse", spaces around '='
    pattern = re.compile(
        rf"hash_size\s*=\s*(\d+).*?morton_sort\s*=\s*(True|False).*?"
        rf"(?:ate\.rmse|ate_rmse)\s*=\s*({FLOAT_RE})",
        re.IGNORECASE
    )

    with open(data_file, "r") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            size = int(m.group(1))
            is_morton = (m.group(2) == "True")
            val = float(m.group(3))
            target = morton if is_morton else normal
            target['sizes'].append(size)
            target['vals'].append(val)

    # Sort by size (x-axis)
    for d in (morton, normal):
        order = sorted(range(len(d['sizes'])), key=lambda i: d['sizes'][i])
        d['sizes'] = [d['sizes'][i] for i in order]
        d['vals'] = [d['vals'][i] for i in order]

    # Plot
    plt.figure()
    if normal['sizes']:
        plt.plot(normal['sizes'], normal['vals'], marker="o", label="No Morton")
    if morton['sizes']:
        plt.plot(morton['sizes'], morton['vals'], marker="s", label="Morton (R=128)")
    plt.xlabel("Hash table size")
    plt.ylabel("ATE RMSE")
    plt.title(f"{scene_label} — ATE RMSE vs Hash Table Size (Morton R=128 vs No Morton) on Orin")
    if normal['sizes'] or morton['sizes']:
        plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

if __name__ == "__main__":
    data_file = "ate_rmse_morton_vs_off_R128.txt"
    out_png = "ate_rmse_morton_vs_off_R128_plot.png"
    if len(sys.argv) >= 2:
        data_file = sys.argv[1]
    if len(sys.argv) >= 3:
        out_png = sys.argv[2]
    plot_ate_vs_size(data_file, out_png)