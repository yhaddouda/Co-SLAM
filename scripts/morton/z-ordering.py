#!/usr/bin/env python3
# Morton sorting interactive demo (CPU, NumPy + Plotly)
#
# Features:
# - Generates toy 3D points with shape [B, S, 3] inside a user-specified AABB
# - Flattens and normalizes to [0,1]^3 (like CoSLAM.run_network prelude)
# - Quantizes to resolution R, builds 64-bit Morton keys (magic-bit method)
# - Stable sorts by keys and outputs:
#     * Two interactive HTML 3D scatter plots (original vs. Morton-sorted, first K points)
#     * A CSV with stage timings
#     * An .npz with arrays (pts_norm, pts_sorted, perm, invperm, keys, bounds, R)
#
# Usage example:
#   python morton_sort_interactive.py --B 2048 --S 43 --R 128 --K 1000 \
#     --bounds -3 3 -4 2.5 -2 2.5 --outdir ./out
#
# Author: (you)
# License: MIT

import argparse
import csv
import os
import time
from typing import Tuple

import numpy as np

def parse_args():
    p = argparse.ArgumentParser(description="Morton sorting demo with interactive HTML plots and timings.")
    p.add_argument("--B", type=int, default=2048, help="Number of rays (batch size).")
    p.add_argument("--S", type=int, default=43, help="Samples per ray.")
    p.add_argument("--R", type=int, default=128, help="Quantization resolution per axis for Morton.")
    p.add_argument("--K", type=int, default=1000, help="How many points to visualize in the first K plots.")
    p.add_argument("--bounds", type=float, nargs=6, default=[-3,3, -4,2.5, -2,2.5],
                   metavar=("xmin","xmax","ymin","ymax","zmin","zmax"),
                   help="Scene AABB as 6 floats.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--outdir", type=str, default="./out", help="Output directory.")
    p.add_argument("--save-npz", action="store_true", help="Save arrays to artifacts.npz")
    return p.parse_args()

# -----------------------------
# Morton helpers (u64, magic-bit mask/shift)
# -----------------------------

def split_by_3_u64(a: np.ndarray) -> np.ndarray:
    """Spread lower 21 bits of 'a' so there are two zero bits between each original bit."""
    a = a.astype(np.uint64) & np.uint64(0x1fffff)  # keep 21 bits
    a = (a | (a << np.uint64(32))) & np.uint64(0x1f00000000ffff)
    a = (a | (a << np.uint64(16))) & np.uint64(0x1f0000ff0000ff)
    a = (a | (a << np.uint64(8)))  & np.uint64(0x100f00f00f00f00f)
    a = (a | (a << np.uint64(4)))  & np.uint64(0x10c30c30c30c30c3)
    a = (a | (a << np.uint64(2)))  & np.uint64(0x1249249249249249)
    return a

def morton3d_u64(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
    """3D Morton (Z-order) code from integer indices (NumPy u64 arrays)."""
    return (split_by_3_u64(ix) |
            (split_by_3_u64(iy) << np.uint64(1)) |
            (split_by_3_u64(iz) << np.uint64(2)))

# -----------------------------
# Plot helpers
# -----------------------------

def save_interactive_html(points01: np.ndarray, title: str, out_html: str, K: int = 1000) -> None:
    """
    Save an interactive 3D scatter (first K points) to HTML.
    Uses Plotly if available; otherwise falls back to Matplotlib PNG (same path with .png).
    """
    K = min(K, points01.shape[0])
    P = points01[:K]

    try:
        import plotly.graph_objects as go  # type: ignore
        fig = go.Figure(data=[go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
            marker=dict(size=2)
        )])
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(range=[0,1], title="x (norm)"),
                yaxis=dict(range=[0,1], title="y (norm)"),
                zaxis=dict(range=[0,1], title="z (norm)"),
                aspectmode="cube",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        os.makedirs(os.path.dirname(out_html), exist_ok=True)
        fig.write_html(out_html, include_plotlyjs="cdn", auto_open=False)
        print(f"[OK] Wrote interactive HTML: {out_html}")
    except Exception as e:
        print(f"[WARN] Plotly not available ({e}). Falling back to Matplotlib PNG.")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=3)
        ax.set_title(title)
        ax.set_xlabel("x (norm)"); ax.set_ylabel("y (norm)"); ax.set_zlabel("z (norm)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        plt.tight_layout()
        png_path = os.path.splitext(out_html)[0] + ".png"
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        fig.savefig(png_path, dpi=160)
        plt.close(fig)
        print(f"[OK] Wrote PNG fallback: {png_path}")

# -----------------------------
# Metrics (optional, for intuition)
# -----------------------------

def cluster_metrics(points01: np.ndarray, K: int = 1000, R_metric: int = 128) -> Tuple[float, float, int]:
    """
    Compute clustering stats on first K points:
      - AABB diagonal length
      - Mean radius to centroid
      - Unique coarse voxels among K (with resolution R_metric)
    """
    K = min(K, points01.shape[0])
    Q = points01[:K]
    aabb_diag = float(np.linalg.norm(Q.max(axis=0) - Q.min(axis=0)))
    c = Q.mean(axis=0)
    mean_rad = float(np.linalg.norm(Q - c, axis=1).mean())
    vox = np.floor(np.clip(Q * R_metric, 0, R_metric - 1)).astype(np.int32)
    uniq_vox = int(np.unique(vox, axis=0).shape[0])
    return aabb_diag, mean_rad, uniq_vox

# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    np.random.seed(args.seed)
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Bounds
    xmin, xmax, ymin, ymax, zmin, zmax = args.bounds
    bounds = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]], dtype=np.float64)
    mins = bounds[:, 0]; maxs = bounds[:, 1]; extent = maxs - mins

    # --- Stage timings ---
    t0 = time.perf_counter()

    # Generate world-space points [B,S,3]
    B, S, R, K = args.B, args.S, args.R, args.K
    pts_world = mins + np.random.rand(B, S, 3) * extent
    N = B * S
    t1 = time.perf_counter()

    # Flatten & normalize to [0,1]^3
    pts = pts_world.reshape(N, 3)
    pts_norm = (pts - mins) / extent
    t2 = time.perf_counter()

    # Quantize to voxel indices
    ixyz = np.floor(np.clip(pts_norm * R, 0, R - 1)).astype(np.uint64)
    ix, iy, iz = ixyz[:, 0], ixyz[:, 1], ixyz[:, 2]
    t3 = time.perf_counter()

    # Morton keys (u64)
    keys = morton3d_u64(ix, iy, iz)
    t4 = time.perf_counter()

    # Stable sort by keys
    perm = np.argsort(keys, kind="mergesort")
    invperm = np.empty_like(perm)
    invperm[perm] = np.arange(N, dtype=perm.dtype)
    t5 = time.perf_counter()

    # Reorder normalized points (for visualization only)
    pts_sorted = pts_norm[perm]
    t6 = time.perf_counter()

    timings = {
        "generate": t1 - t0,
        "normalize": t2 - t1,
        "quantize": t3 - t2,
        "morton": t4 - t3,
        "sort": t5 - t4,
        "reorder": t6 - t5,
        "total": t6 - t0,
        "N": int(N),
        "B": int(B),
        "S": int(S),
        "R": int(R),
        "K": int(K),
    }

    # Metrics for the first K
    a0, r0, u0 = cluster_metrics(pts_norm, K=K, R_metric=128)
    a1, r1, u1 = cluster_metrics(pts_sorted, K=K, R_metric=128)

    # Save interactive plots
    html_orig = os.path.join(outdir, "original_firstK.html")
    html_sorted = os.path.join(outdir, f"morton_sorted_firstK_R{R}.html")
    save_interactive_html(pts_norm,  f"First {K} — Original order",         html_orig,  K=K)
    save_interactive_html(pts_sorted, f"First {K} — Morton-sorted (R={R})", html_sorted, K=K)

    # Save timings CSV
    csv_path = os.path.join(outdir, "morton_timings.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "seconds"])
        for k in ["generate","normalize","quantize","morton","sort","reorder","total"]:
            w.writerow([k, f"{timings[k]:.6f}"])
    print(f"[OK] Wrote timings: {csv_path}")

    # Print metrics
    print("=== Clustering metrics on first K ===")
    print(f"Original: AABB diag={a0:.4f}, mean radius={r0:.4f}, unique voxels={u0}")
    print(f"Morton  : AABB diag={a1:.4f}, mean radius={r1:.4f}, unique voxels={u1}")

    # Optionally save artifacts for inspection
    if args.save_npz:
        npz_path = os.path.join(outdir, "artifacts.npz")
        np.savez(npz_path,
                 pts_norm=pts_norm.astype(np.float32),
                 pts_sorted=pts_sorted.astype(np.float32),
                 perm=perm, invperm=invperm, keys=keys,
                 bounds=bounds, R=np.array([R], dtype=np.int32))
        print(f"[OK] Wrote artifacts: {npz_path}")

if __name__ == "__main__":
    main()


