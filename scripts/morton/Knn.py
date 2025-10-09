#!/usr/bin/env python3
# Knn.py
# Visualize locality effects of Morton ordering vs original ordering on 3D points.

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ------------ Utils ------------

def read_points_csv(path: str, sep: str = ",", header: bool = False) -> np.ndarray:
    """Read Nx3 CSV into float64 numpy array (no downcast)."""
    df = pd.read_csv(path, sep=sep, header=0 if header else None, dtype=np.float64)
    arr = df.values
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{path}: expected shape (N,3), got {arr.shape}")
    return arr


def ensure_same_points_set(A: np.ndarray,
                           B: np.ndarray,
                           rtol: float = 0.0,
                           atol: float = 0.0,
                           verbose: bool = True) -> bool:
    """
    FULL multiset equality check (no sampling).
    - Returns True iff A and B contain exactly the same rows (order ignored)
      within the given tolerances.
    - If verbose, prints a one-line verdict and (if unequal) the max abs diff.
    """
    if A.shape != B.shape:
        if verbose:
            print(f" Shapes differ: {A.shape} vs {B.shape}")
        return False

    # Lexicographic sort both arrays by (x, y, z) and compare elementwise
    idxA = np.lexsort((A[:, 2], A[:, 1], A[:, 0]))
    idxB = np.lexsort((B[:, 2], B[:, 1], B[:, 0]))
    As, Bs = A[idxA], B[idxB]

    same = np.allclose(As, Bs, rtol=rtol, atol=atol, equal_nan=True)
    if verbose:
        if same:
            print("Multiset equality (full check): True")
        else:
            max_abs_diff = float(np.max(np.abs(As - Bs)))
            print("⚠️ Multiset equality (full check): False")
            print("   Max abs diff:", max_abs_diff)
    return same

def consecutive_step_lengths(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X[1:] - X[:-1], axis=1)

def print_step_length_stats(d_orig: np.ndarray, d_mor: np.ndarray):
    """
    Print total, mean, and variance of consecutive step lengths for both orders,
    plus percent deltas (Morton vs Original).
    """
    L_o, L_m = d_orig.sum(), d_mor.sum()
    mu_o, mu_m = d_orig.mean(), d_mor.mean()
    var_o, var_m = d_orig.var(), d_mor.var()

    print("Consecutive step-length stats (‖X[i+1]-X[i]‖):")
    print(f"  Original -> total: {L_o:.6g} | mean: {mu_o:.6g} | var: {var_o:.6g}")
    print(f"  Morton   -> total: {L_m:.6g} | mean: {mu_m:.6g} | var: {var_m:.6g}")

    def pct(a, b):  # % change of Morton vs Original
        return 100.0 * (a - b) / b if b != 0 else float('nan')

    print("  Δ Morton vs Original:")
    print(f"    total: {pct(L_m, L_o):+.2f}% | mean: {pct(mu_m, mu_o):+.2f}% | var: {pct(var_m, var_o):+.2f}%")
    print("  (Lower mean/variance indicate fewer/shorter long jumps along the order.)")


def neighbor_continuity(X: np.ndarray, k: int = 16, sample: int | None = 20000) -> float:
    """
    Fraction of adjacent indices (i-1, i+1) that fall within the spatial k-NN of i.
    """
    N = len(X)
    if N < 3:
        return 0.0
    rng = np.random.default_rng(0)
    idx = np.arange(N) if (sample is None or sample >= N) else np.sort(rng.choice(N, size=sample, replace=False))
    nn = NearestNeighbors(n_neighbors=min(k+1, N), algorithm="auto")
    nn.fit(X)
    _, neigh = nn.kneighbors(X[idx], return_distance=True)
    neigh_sets = [set(n[1:]) for n in neigh]  # skip self (index 0)
    hits, total = 0, 0
    for j, i in enumerate(idx):
        for nxt in (i-1, i+1):
            if 0 <= nxt < N:
                total += 1
                if nxt in neigh_sets[j]:
                    hits += 1
    return hits / total if total else 0.0

# ------------ Plots ------------

def index_colored_scatter_2d(X: np.ndarray, title: str, outdir: Path, subsample: int | None):
    """
    Scatter plots in XY, XZ, YZ colored by index.
    """
    N = len(X)
    if subsample is not None and subsample < N:
        rng = np.random.default_rng(2)
        sel = np.sort(rng.choice(N, size=subsample, replace=False))
    else:
        sel = np.arange(N)

    axes_list = [(0,1,"XY"), (0,2,"XZ"), (1,2,"YZ")]
    for a,b,name in axes_list:
        plt.figure()
        sc = plt.scatter(X[sel, a], X[sel, b], s=1, c=sel)  # default colormap
        plt.xlabel(["X","Y","Z"][a])
        plt.ylabel(["X","Y","Z"][b])
        plt.title(f"{title} — index-colored {name}")
        plt.colorbar(sc, label="Index")
        plt.tight_layout()
        plt.savefig(outdir / f"{title.lower().replace(' ','_')}_index_scatter_{name.lower()}.png", dpi=200)
        plt.close()

def step_length_hist(d1: np.ndarray, d2: np.ndarray, label1: str, label2: str, outpath: Path):
    plt.figure()
    plt.hist(d1, bins=50, density=True, alpha=0.6, label=label1)
    plt.hist(d2, bins=50, density=True, alpha=0.6, label=label2)
    plt.xlabel(r"Consecutive step length $\|X_{i+1}-X_i\|$")
    plt.ylabel("Density")
    plt.title("Consecutive step-length distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def knn_mean_distance_hist(m1: np.ndarray, m2: np.ndarray, label1: str, label2: str, outpath: Path):
    plt.figure()
    plt.hist(m1, bins=100, density=True, alpha=0.6, label=label1)
    plt.hist(m2, bins=100, density=True, alpha=0.6, label=label2)
    plt.xlabel(f"Mean distance to k-NN")
    plt.ylabel("Density")
    plt.title("Mean k-NN distance per point")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def continuity_bar(c1: float, c2: float, label1: str, label2: str, outpath: Path):
    plt.figure()
    vals = [c1, c2]
    labels = [label1, label2]
    x = np.arange(len(vals))
    plt.bar(x, vals)
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel(f"Fraction of (i±1) found in k-NN")
    plt.title("k-NN continuity (higher is better)")
    for xi, v in zip(x, vals):
        plt.text(xi, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# ------------ Main ------------

def main():
    ap = argparse.ArgumentParser(description="Visualize locality improvements from Morton ordering.")
    ap.add_argument("--orig", required=True, help="Path to original-order CSV (N x 3).")
    ap.add_argument("--morton", required=True, help="Path to Morton-ordered CSV (N x 3).")
    ap.add_argument("--sep", default=",", help="CSV separator (default: ',').")
    ap.add_argument("--header", action="store_true", help="Set if CSVs have a header row.")
    ap.add_argument("--k", type=int, default=16, help="k for k-NN (default: 16).")
    ap.add_argument("--sample", type=int, default=20000,
                    help="Sample size for k-NN stats (None or <=0 to use all). Default: 20000.")
    ap.add_argument("--scatter", type=int, default=20000,
                    help="Subsample for index-colored scatter (None or <=0 to use all). Default: 20000.")
    ap.add_argument("--out", default="plots/morton_plots", help="Output directory for plots.")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading CSVs...")
    Xo = read_points_csv(args.orig, sep=args.sep, header=args.header)
    Xm = read_points_csv(args.morton, sep=args.sep, header=args.header)

    print(f"Original shape: {Xo.shape} | Morton shape: {Xm.shape}")
    _ = ensure_same_points_set(Xo, Xm, rtol=0.0, atol=0.0, verbose=True)

    # 1) Index-colored 2D projections
    print("Plotting index-colored 2D projections...")
    scatter_ss = None if (args.scatter is None or args.scatter <= 0) else args.scatter
    index_colored_scatter_2d(Xo, "Original", outdir, subsample=scatter_ss)
    index_colored_scatter_2d(Xm, "Morton",   outdir, subsample=scatter_ss)

    # 2) Consecutive step-lengths (order path smoothness)
    print("Computing consecutive step lengths...")
    d_o = consecutive_step_lengths(Xo)
    d_m = consecutive_step_lengths(Xm)
    print_step_length_stats(d_o, d_m)


    # 3) k-NN continuity (adjacent indices within k-NN)
    print("Computing k-NN continuity...")
    sample_knn = None if (args.sample is None or args.sample <= 0) else args.sample
    c_o = neighbor_continuity(Xo, k=args.k, sample=sample_knn)
    c_m = neighbor_continuity(Xm, k=args.k, sample=sample_knn)
    continuity_bar(c_o, c_m, "Original", "Morton", outdir / "knn_continuity.png")
    print(f"k-NN continuity — Original: {c_o:.4f} | Morton: {c_m:.4f} (k={args.k})")

if __name__ == "__main__":
    main()
