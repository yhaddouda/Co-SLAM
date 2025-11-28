import pandas as pd
import matplotlib.pyplot as plt

log_file = "replica.log"

# -----------------------------
# 1. Parse replica.log
# -----------------------------
rows = []

with open(log_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Expect lines like:
        # scene=office0, hash_size=16, tag=fp16, setup=A, exp_name=..., time_s=1524, peak_RAM_MB=14553
        parts = [p.strip() for p in line.split(",")]
        entry = {}
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()
            entry[k] = v

        # Skip if required keys are missing
        if "scene" not in entry or "hash_size" not in entry or "setup" not in entry or "time_s" not in entry:
            continue

        try:
            entry["hash_size"] = int(entry["hash_size"])
            entry["time_s"] = float(entry["time_s"])
        except ValueError:
            continue

        rows.append(entry)

df = pd.DataFrame(rows)

# -----------------------------
# 2. Filter to T=19 and T=20
# -----------------------------
target_hashes = [19, 20]
df_sub = df[df["hash_size"].isin(target_hashes)]

# Average time over all scenes for each (hash_size, setup)
grouped = (
    df_sub
    .groupby(["hash_size", "setup"])["time_s"]
    .mean()
    .reset_index()
)

# Ensure setups are in A, B, C, D order
setup_order = ["A", "B", "C", "D"]
setup_labels = {
    "A": "A – Baseline",
    "B": "B – Morton hash (no sort)",
    "C": "C – Prior sort (baseline hash)",
    "D": "D – Morton sort complet",
}

# -----------------------------
# 3. Make one plot per T, with 4 bars (A,B,C,D)
# -----------------------------
for T in target_hashes:
    df_T = grouped[grouped["hash_size"] == T].set_index("setup")

    # Make sure we have all four setups
    missing = [s for s in setup_order if s not in df_T.index]
    if missing:
        print(f"[Warning] For T={T}, missing setups: {missing}")
        continue

    times = [df_T.loc[s, "time_s"] for s in setup_order]  # in seconds
    baseline_time = times[0]  # setup A

    # Speedups vs baseline (A)
    speedups = {}
    print(f"\n=== Hash size T={T} (2^{T}) ===")
    for s in setup_order:
        t_s = df_T.loc[s, "time_s"]
        if s == "A":
            speedups[s] = 0.0
            print(f"{setup_labels[s]}: {t_s:.2f} s (baseline)")
        else:
            gain = (baseline_time - t_s) / baseline_time * 100.0
            speedups[s] = gain
            print(f"{setup_labels[s]}: {t_s:.2f} s  ->  {gain:.2f}% faster than baseline")

    # We'll highlight D vs A in the title
    gain_D = speedups["D"]

    # -----------------------------
    # Plot: histogram-style bar plot
    # -----------------------------
    x = range(len(setup_order))

    plt.figure(figsize=(7, 5))
    bars = plt.bar(x, times)

    plt.xticks(
        x,
        [setup_labels[s] for s in setup_order],
        rotation=20,
        ha="right"
    )
    plt.ylabel("Average execution time (s, lower is better)")
    plt.title(
        f"Replica – hash size 2^{T} (T={T})\n"
        f"Morton sort complet vs baseline: {gain_D:.1f}% faster"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bars with their values in seconds
    for bar, val in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.1f} s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    out_name = f"replica_T{T}_avg_time_setups_hist.png"
    plt.savefig(out_name, dpi=300)
    # If you also want PDF for LaTeX:
    # plt.savefig(f"replica_T{T}_avg_time_setups_hist.pdf")
    plt.close()

    print(f"Saved figure: {out_name}")
