#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse des logs de type:
=== Running eval for 13 ===
accuracy:  1.5922
completion:  1.5875
completion ratio:  95.9940
Depth L1:  0.000766
ATE.RMSE: 0.00590
...
et produit 5 graphiques (accuracy, completion, completion ratio, Depth L1, ATE.RMSE)
avec 2 courbes: full vs half, en fonction de la taille (13, 14, ...).
"""

import argparse
import re
from pathlib import Path
import math
import matplotlib.pyplot as plt

BLOCK_RE = re.compile(
    r"^===\s*Running\s+eval\s+for\s+([^\s=]+)\s*===\s*\n(.*?)(?=^===\s*Running\s+eval|\Z)",
    re.MULTILINE | re.DOTALL,
)

METRIC_PATTERNS = {
    "accuracy": re.compile(r"^accuracy:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.MULTILINE),
    "completion": re.compile(r"^completion:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.MULTILINE),
    "completion_ratio": re.compile(r"^completion\s+ratio:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.MULTILINE),
    "depth_l1": re.compile(r"^Depth\s+L1:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.MULTILINE),
    "ate_rmse": re.compile(r"^ATE\.RMSE:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.MULTILINE),
}

METRIC_PRETTY = {
    "accuracy": "Accuracy",
    "completion": "Completion",
    "completion_ratio": "Completion Ratio",
    "depth_l1": "Depth L1",
    "ate_rmse": "ATE RMSE",
}

def parse_size_and_kind(label: str):
    """
    label: ex '13', '13_half', '16_base', '20_half'...
    retourne (taille:int, kind:str in {'full','half'})
    """
    m = re.search(r"(\d+)", label)
    if not m:
        return None, None
    size = int(m.group(1))
    kind = "half" if "_half" in label.lower() else "full"
    return size, kind

def parse_blocks(text: str):
    """
    Retourne une structure:
    data[kind]['metric'][size] = value
    où kind ∈ {'full','half'}
    """
    data = {"full": {k: {} for k in METRIC_PATTERNS}, "half": {k: {} for k in METRIC_PATTERNS}}
    for m in BLOCK_RE.finditer(text):
        label = m.group(1).strip()
        body = m.group(2)
        size, kind = parse_size_and_kind(label)
        if size is None or kind is None:
            continue
        for key, preg in METRIC_PATTERNS.items():
            mm = preg.search(body)
            if mm:
                try:
                    val = float(mm.group(1))
                except ValueError:
                    continue
                data[kind][key][size] = val
    return data

def ensure_sorted_common_sizes(full_dict, half_dict):
    """Renvoie l'ensemble trié des tailles rencontrées (union)."""
    sizes = set(full_dict.keys()) | set(half_dict.keys())
    return sorted(sizes)

def series_for_sizes(dct, sizes):
    """Retourne une liste alignée sur sizes, avec float('nan') si manquant."""
    return [dct.get(s, float("nan")) for s in sizes]

def plot_metric(sizes, y_full, y_half, metric_key, outdir: Path):
    plt.figure()
    plt.plot(sizes, y_full, marker="o", label="Full precision")
    plt.plot(sizes, y_half, marker="s", label="Half precision")
    plt.xlabel("Taille du tableau")
    plt.ylabel(METRIC_PRETTY.get(metric_key, metric_key))
    plt.title(f"{METRIC_PRETTY.get(metric_key, metric_key)} vs Taille")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    outpath = outdir / f"{metric_key}.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    return outpath

def main():
    ap = argparse.ArgumentParser(description="Plot des métriques full vs half par taille à partir d'un fichier de logs.")
    ap.add_argument("logfile", type=Path, help="Chemin du fichier de logs")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("plots"), help="Dossier de sortie pour les PNG")
    args = ap.parse_args()

    text = args.logfile.read_text(encoding="utf-8", errors="ignore")
    data = parse_blocks(text)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Pour chaque métrique, tracer une figure avec 2 courbes (full/half).
    for metric_key in METRIC_PATTERNS.keys():
        sizes = ensure_sorted_common_sizes(data["full"][metric_key], data["half"][metric_key])
        if not sizes:
            # rien à tracer pour cette métrique
            continue
        y_full = series_for_sizes(data["full"][metric_key], sizes)
        y_half = series_for_sizes(data["half"][metric_key], sizes)
        plot_metric(sizes, y_full, y_half, metric_key, args.outdir)

    print(f"Figures enregistrées dans: {args.outdir.resolve()}")

if __name__ == "__main__":
    main()
