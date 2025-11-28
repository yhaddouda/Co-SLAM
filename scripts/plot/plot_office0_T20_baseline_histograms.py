import pandas as pd
import matplotlib.pyplot as plt

# Input files (office0, hash size 2^20)
baseline_file = "office0_T20_baseline_nvtx_sum.csv"
morton_file   = "office0_T20_morton_nvtx_sum.csv"

# NVTX ranges -> pretty labels
wanted_ranges = {
    ":Tracking": "Tracking",
    ":Bundle_Adjustment": "Bundle adjustment",
    ":forward": "Forward",
    ":backward": "Backward",
    ":Decoder": "Decoder",
    ":TCNN_hashgrid_encoding": "TCNN hashgrid enc.",
    ":TCNN_oneblob_encoding": "TCNN oneblob enc.",
}

# Read CSVs
baseline = pd.read_csv(baseline_file)
morton   = pd.read_csv(morton_file)

def get_avg_ms(df, range_key):
    """Return Avg (ns) converted to ms for a given Range."""
    series = df.loc[df["Range"] == range_key, "Avg (ns)"]
    if series.empty:
        raise ValueError(f"Range {range_key} not found in dataframe")
    return series.iloc[0] / 1e6  # ns -> ms

results = {}

# Collect data + gains
for r_key, nice_label in wanted_ranges.items():
    b_ms = get_avg_ms(baseline, r_key)
    m_ms = get_avg_ms(morton, r_key)
    gain_pct = (b_ms - m_ms) / b_ms * 100.0  # positive = faster with morton

    results[r_key] = {
        "label": nice_label,
        "baseline_ms": b_ms,
        "morton_ms": m_ms,
        "gain_pct": gain_pct,
    }

    print(
        f"{nice_label}: baseline = {b_ms:.3f} ms, "
        f"morton sort complet = {m_ms:.3f} ms, "
        f"gain = {gain_pct:.2f}% faster"
    )

# Create a separate figure per function
for r_key, data in results.items():
    label = data["label"]
    b_ms = data["baseline_ms"]
    m_ms = data["morton_ms"]
    gain_pct = data["gain_pct"]

    x = [0, 1]
    heights = [b_ms, m_ms]
    bar_labels = ["Baseline", "Morton sort complet"]

    plt.figure(figsize=(5, 4))

    # Two separate bar calls so they get different colors
    bars_baseline = plt.bar(x[0], heights[0], label="Baseline")
    bars_morton   = plt.bar(x[1], heights[1], label="Morton sort complet")

    # Axis labels and title (show gain in %)
    plt.xticks(x, bar_labels)
    plt.ylabel("Avg time (ms, lower is better)")
    plt.title(
        f"{label} – office0, hash size 2^20\n"
        f"Morton sort complet gain: {gain_pct:.1f}% faster"
    )
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate each bar with its value
    for bar, val in zip([bars_baseline[0], bars_morton[0]], heights):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.2f} ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    # Safe filename from label
    safe_label = (
        label.lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("/", "_")
    )
    out_name = f"office0_T20_{safe_label}_baseline_vs_morton.png"
    plt.savefig(out_name, dpi=300)
    plt.close()


