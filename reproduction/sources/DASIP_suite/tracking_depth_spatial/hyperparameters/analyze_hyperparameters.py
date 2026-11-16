#!/usr/bin/env python3
"""Analyse the Replica depth-spatial hyperparameter sweep and build a report."""

from __future__ import annotations

import csv
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median, stdev

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
SAMPLE_CSV = ROOT / "Replica_depth_spatial_sample_count_sweep.csv"
QUALITY_CSV = ROOT / "replica_depth_spatial_hyperparameters.csv"
BASELINE_CSV = ROOT.parent / "replica_depth_spatial_results.csv"
SCENES = ("office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2")
CONFIG_KEYS = ("PoolFactor", "GridH", "GridW", "MinCoverage")
METRICS = ("Accuracy", "Completion", "CompletionRatio", "DepthL1")
UNIFORM_SAMPLES = 32

# A transparent engineering envelope used only to label recommendations.
LIMITS = {
    "Accuracy_Change_pct": 5.0,
    "Completion_Change_pct": 5.0,
    "CompletionRatio_Change_pp": -0.5,
    "DepthL1_Change_pct": 20.0,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def config(row: dict[str, str]) -> tuple[int, int, int, float]:
    return (
        int(row["PoolFactor"]), int(row["GridH"]), int(row["GridW"]),
        float(row["MinCoverage"]),
    )


def config_id(item: tuple[int, int, int, float]) -> str:
    pf, gh, gw, mc = item
    return f"PF{pf}-G{gh}x{gw}-MC{mc:g}"


def baseline_quality() -> dict[str, float]:
    rows = [row for row in read_rows(BASELINE_CSV) if row["Sampling"] == "baseline"]
    output: dict[str, float] = {}
    for metric in METRICS:
        run_macros = []
        for run in sorted({int(row["Run"]) for row in rows}):
            values = [
                float(row[metric]) * (1000.0 if metric == "DepthL1" else 1.0)
                for row in rows
                if int(row["Run"]) == run
                and not (metric == "DepthL1" and row["Scene"] == "office4")
            ]
            run_macros.append(fmean(values))
        output[metric] = fmean(run_macros)
    return output


def aggregate() -> tuple[list[dict[str, object]], dict[str, float]]:
    samples = read_rows(SAMPLE_CSV)
    quality = read_rows(QUALITY_CSV)
    baseline = baseline_quality()

    sample_configs = sorted({config(row) for row in samples})
    if len(sample_configs) != 32:
        raise ValueError(f"Expected 32 sample-count configurations, found {len(sample_configs)}")
    if {int(row["Run"]) for row in samples} != {1, 2, 3}:
        raise ValueError("Expected sample-count runs 1--3")

    rows: list[dict[str, object]] = []
    for cfg in sample_configs:
        run_macros = []
        for run in (1, 2, 3):
            observations = [
                float(row["MeanUniformSamples"])
                for row in samples if config(row) == cfg and int(row["Run"]) == run
            ]
            if len(observations) != len(SCENES):
                raise ValueError(f"Incomplete sample-count coverage for {config_id(cfg)}, run {run}")
            run_macros.append(fmean(observations))

        qrows = [row for row in quality if config(row) == cfg]
        present_scenes = {row["Scene"] for row in qrows}
        complete = present_scenes == set(SCENES)
        result: dict[str, object] = {
            "Configuration": config_id(cfg),
            "PoolFactor": cfg[0],
            "GridH": cfg[1],
            "GridW": cfg[2],
            "MinCoverage": cfg[3],
            "MeanUniformSamples": fmean(run_macros),
            "SampleSD_AcrossRuns": stdev(run_macros),
            "UniformSampleReduction_pct": 100.0 * (1.0 - fmean(run_macros) / UNIFORM_SAMPLES),
            "SampleRuns": 3,
            "QualityComplete": complete,
            "QualitySceneCount": len(present_scenes),
            "QualityObservations": len(qrows),
        }

        replicate_counts = [sum(row["Scene"] == scene for row in qrows) for scene in present_scenes]
        result["QualityReplicatesPerScene_Min"] = min(replicate_counts) if replicate_counts else 0
        result["QualityReplicatesPerScene_Max"] = max(replicate_counts) if replicate_counts else 0

        for metric in METRICS:
            scene_values = []
            for scene in SCENES:
                if scene not in present_scenes or (metric == "DepthL1" and scene == "office4"):
                    continue
                observations = [
                    float(row[metric]) * (1000.0 if metric == "DepthL1" else 1.0)
                    for row in qrows if row["Scene"] == scene
                ]
                scene_values.append(fmean(observations))
            result[metric] = fmean(scene_values) if scene_values else math.nan

        result["Accuracy_Change_pct"] = 100.0 * (
            float(result["Accuracy"]) / baseline["Accuracy"] - 1.0
        )
        result["Completion_Change_pct"] = 100.0 * (
            float(result["Completion"]) / baseline["Completion"] - 1.0
        )
        result["CompletionRatio_Change_pp"] = (
            float(result["CompletionRatio"]) - baseline["CompletionRatio"]
        )
        result["DepthL1_Change_pct"] = 100.0 * (
            float(result["DepthL1"]) / baseline["DepthL1"] - 1.0
        )
        terms = (
            float(result["Accuracy_Change_pct"]) / LIMITS["Accuracy_Change_pct"],
            float(result["Completion_Change_pct"]) / LIMITS["Completion_Change_pct"],
            -float(result["CompletionRatio_Change_pp"]) / -LIMITS["CompletionRatio_Change_pp"],
            float(result["DepthL1_Change_pct"]) / LIMITS["DepthL1_Change_pct"],
        )
        result["QualityEnvelopeScore"] = max(terms)
        result["WithinQualityEnvelope"] = complete and max(terms) <= 1.0 + 1e-9
        rows.append(result)

    complete_rows = [row for row in rows if row["QualityComplete"]]
    for row in rows:
        row["SpeedQualityPareto"] = False
    for candidate in complete_rows:
        dominated = any(
            float(other["UniformSampleReduction_pct"]) >= float(candidate["UniformSampleReduction_pct"])
            and float(other["QualityEnvelopeScore"]) <= float(candidate["QualityEnvelopeScore"])
            and (
                float(other["UniformSampleReduction_pct"]) > float(candidate["UniformSampleReduction_pct"])
                or float(other["QualityEnvelopeScore"]) < float(candidate["QualityEnvelopeScore"])
            )
            for other in complete_rows if other is not candidate
        )
        candidate["SpeedQualityPareto"] = not dominated
    return rows, baseline


def choose_recommendations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    complete = [row for row in rows if row["QualityComplete"]]
    balanced = max(
        (row for row in complete if row["WithinQualityEnvelope"]),
        key=lambda row: float(row["UniformSampleReduction_pct"]),
    )
    quality_first = min(complete, key=lambda row: float(row["QualityEnvelopeScore"]))
    aggressive = max(
        (row for row in complete if float(row["QualityEnvelopeScore"]) <= 1.15),
        key=lambda row: float(row["UniformSampleReduction_pct"]),
    )
    selections = (
        ("Balanced", balanced, "Fastest configuration inside the stated quality envelope"),
        ("Quality-first", quality_first, "Lowest worst-case quality-envelope utilisation"),
        ("Aggressive", aggressive, "Fastest configuration within 15% of the quality envelope"),
    )
    output = []
    for label, row, rationale in selections:
        item = dict(row)
        item["Preset"] = label
        item["Rationale"] = rationale
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: f"{value:.9f}" if isinstance(value, float) else value
                for key, value in row.items()
            })


def factor_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for pf in (4, 8):
        for mc in (0.25, 0.5, 0.75, 0.9):
            group = [
                row for row in rows
                if row["QualityComplete"] and row["PoolFactor"] == pf
                and math.isclose(float(row["MinCoverage"]), mc)
            ]
            output.append({
                "PoolFactor": pf,
                "MinCoverage": mc,
                "CompleteGridConfigurations": len(group),
                "MeanUniformSamples": fmean(float(row["MeanUniformSamples"]) for row in group),
                "MeanUniformSampleReduction_pct": fmean(
                    float(row["UniformSampleReduction_pct"]) for row in group
                ),
                "MedianQualityEnvelopeScore": median(
                    float(row["QualityEnvelopeScore"]) for row in group
                ),
                "ConfigurationsWithinEnvelope": sum(bool(row["WithinQualityEnvelope"]) for row in group),
            })
    return output


def plot_tradeoff(rows: list[dict[str, object]], recommendations: list[dict[str, object]]) -> None:
    viable = [
        row for row in rows if row["QualityComplete"] and float(row["MinCoverage"]) >= 0.75
    ]
    rec_by_config = {row["Configuration"]: row["Preset"] for row in recommendations}
    colors = {0.75: "#E69F00", 0.9: "#0072B2"}
    markers = {4: "o", 8: "s"}
    panels = (
        ("Accuracy_Change_pct", "Accuracy error change (%)", 5.0, "upper"),
        ("Completion_Change_pct", "Completion error change (%)", 5.0, "upper"),
        ("CompletionRatio_Change_pp", "Completion-ratio change (pp)", -0.5, "lower"),
        ("DepthL1_Change_pct", "Depth L1 change (%)", 20.0, "upper"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)
    for ax, (field, ylabel, limit, direction) in zip(axes.flat, panels):
        ys = [float(row[field]) for row in viable]
        ymin, ymax = min(ys), max(ys)
        pad = max((ymax - ymin) * 0.13, 0.5)
        ax.set_ylim(ymin - pad, ymax + pad)
        if direction == "upper":
            ax.axhspan(ymin - pad, limit, color="#009E73", alpha=0.08)
        else:
            ax.axhspan(limit, ymax + pad, color="#009E73", alpha=0.08)
        ax.axhline(0, color="0.45", lw=0.9, ls="--")
        ax.axhline(limit, color="#009E73", lw=1.1, ls=":")
        for row in viable:
            x = float(row["UniformSampleReduction_pct"])
            y = float(row[field])
            ax.scatter(
                x, y, s=54, marker=markers[int(row["PoolFactor"])],
                color=colors[float(row["MinCoverage"])], edgecolor="white",
                linewidth=0.6, zorder=3,
            )
            if row["Configuration"] in rec_by_config:
                short = {"Balanced": "B", "Quality-first": "Q", "Aggressive": "A"}[
                    rec_by_config[str(row["Configuration"])]
                ]
                ax.annotate(short, (x, y), xytext=(4, 5), textcoords="offset points",
                            fontsize=8, weight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="0.9", lw=0.7)
    for ax in axes[1]:
        ax.set_xlabel("Uniform samples removed relative to 32 (%)  →")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[0.75],
               markeredgecolor="white", markersize=8, label="min coverage 0.75"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[0.9],
               markeredgecolor="white", markersize=8, label="min coverage 0.90"),
        Line2D([0], [0], marker="o", color="0.3", linestyle="none", markersize=7,
               label="pool factor 4"),
        Line2D([0], [0], marker="s", color="0.3", linestyle="none", markersize=7,
               label="pool factor 8"),
        Line2D([0], [0], color="#009E73", ls=":", label="quality-envelope boundary"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
               ncol=5, frameon=False, fontsize=8.5)
    fig.suptitle("Speed–quality trade-off in the viable coverage region", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / f"quality_speed_tradeoff.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_phase_transition(rows: list[dict[str, object]]) -> None:
    complete = [row for row in rows if row["QualityComplete"]]
    grid_colors = {(4, 4): "#0072B2", (4, 8): "#E69F00", (8, 4): "#009E73", (8, 8): "#CC79A7"}
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    for column, pf in enumerate((4, 8)):
        for grid, color in grid_colors.items():
            group = sorted(
                (row for row in complete if row["PoolFactor"] == pf
                 and (row["GridH"], row["GridW"]) == grid),
                key=lambda row: float(row["MinCoverage"]),
            )
            axes[0, column].plot(
                [float(row["MinCoverage"]) for row in group],
                [float(row["QualityEnvelopeScore"]) for row in group],
                marker="o", lw=1.7, color=color, label=f"{grid[0]}×{grid[1]}",
            )
            axes[1, column].plot(
                [float(row["MinCoverage"]) for row in group],
                [float(row["UniformSampleReduction_pct"]) for row in group],
                marker="o", lw=1.7, color=color,
            )
        axes[0, column].set_yscale("log")
        axes[0, column].axhline(1.0, color="#009E73", ls=":", lw=1.2)
        axes[0, column].set_title(f"Pool factor {pf}")
        axes[0, column].grid(True, which="both", color="0.9", lw=0.7)
        axes[1, column].grid(True, color="0.9", lw=0.7)
        axes[1, column].set_xlabel("Minimum grid coverage")
        axes[1, column].set_xticks((0.25, 0.5, 0.75, 0.9))
    axes[0, 0].set_ylabel("Quality-envelope score (≤1 passes; log scale)")
    axes[1, 0].set_ylabel("Uniform samples removed (%)")
    axes[0, 0].legend(title="Grid H×W", frameon=False, fontsize=8, title_fontsize=8)
    fig.suptitle("Minimum coverage controls the failure-to-quality transition", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / f"coverage_phase_transition.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def tex_escape(text: object) -> str:
    return str(text).replace("%", r"\%").replace("×", r"$\times$")


def make_report(
    rows: list[dict[str, object]], baseline: dict[str, float],
    recommendations: list[dict[str, object]], factors: list[dict[str, object]],
) -> None:
    rec_lines = []
    for row in recommendations:
        rec_lines.append(
            f"{row['Preset']} & {row['PoolFactor']} & {row['GridH']}$\\times${row['GridW']} & "
            f"{float(row['MinCoverage']):.2f} & {float(row['MeanUniformSamples']):.2f} & "
            f"{float(row['UniformSampleReduction_pct']):.1f} & "
            f"{float(row['Accuracy_Change_pct']):+.1f} & "
            f"{float(row['Completion_Change_pct']):+.1f} & "
            f"{float(row['CompletionRatio_Change_pp']):+.3f} & "
            f"{float(row['DepthL1_Change_pct']):+.1f} \\\\"
        )
    factor_lines = []
    for row in factors:
        factor_lines.append(
            f"{row['PoolFactor']} & {float(row['MinCoverage']):.2f} & "
            f"{float(row['MeanUniformSamples']):.2f} & "
            f"{float(row['MeanUniformSampleReduction_pct']):.1f} & "
            f"{float(row['MedianQualityEnvelopeScore']):.2f} & "
            f"{row['ConfigurationsWithinEnvelope']}/{row['CompleteGridConfigurations']} \\\\"
        )
    incomplete = [row for row in rows if not row["QualityComplete"]]
    balanced = next(row for row in recommendations if row["Preset"] == "Balanced")
    quality_first = next(row for row in recommendations if row["Preset"] == "Quality-first")
    aggressive = next(row for row in recommendations if row["Preset"] == "Aggressive")

    tex = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=17mm]{geometry}
\usepackage{booktabs,tabularx,array,graphicx,xcolor}
\usepackage{float}
\pagestyle{plain}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
\begin{document}
\begin{center}
{\Large\bfseries Replica Depth-Spatial Sampling Hyperparameter Study}\par
\small Pool factor, spatial-grid size, and minimum coverage
\end{center}

\section*{Main finding}
Minimum coverage is the dominant hyperparameter. Coverage values 0.25 and 0.50 retain fewer samples but frequently cause severe reconstruction failure; 0.75 is the speed--quality knee, while 0.90 is the quality-first region. Under the stated quality envelope, the recommended balanced setting is \textbf{pool factor @@BALANCED_PF@@, grid @@BALANCED_GH@@$\times$@@BALANCED_GW@@, minimum coverage @@BALANCED_MC@@}, retaining @@BALANCED_SAMPLES@@ of 32 uniform samples (@@BALANCED_REMOVED@@\% removed).

The quality envelope is an explicit engineering criterion: at most +5\% accuracy error, +5\% completion error, $-0.5$ percentage points completion ratio, and +20\% Depth L1 relative to the unpruned baseline. It is used for recommendation labels, not as a statistical significance test.

\begin{table}[H]
\centering
\caption{Recommended operating points. Quality columns show change from the baseline; lower is better except completion ratio.}
\scriptsize
\setlength{\tabcolsep}{3.0pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
Preset & PF & Grid & Coverage & Samples & Removed (\%) & Acc. (\%) & Comp. (\%) & Ratio (pp) & Depth L1 (\%) \\
\midrule
@@REC_LINES@@
\bottomrule
\end{tabular}
\end{table}

\textbf{Preset interpretation.} Balanced is the fastest setting inside the envelope. Quality-first minimises the worst normalised quality degradation and retains @@QUALITY_SAMPLES@@ samples. Aggressive removes @@AGGRESSIVE_REMOVED@@\% of uniform samples but slightly exceeds the envelope, principally in completion error and completion ratio.

\begin{figure}[H]
\centering
\includegraphics[width=\linewidth]{quality_speed_tradeoff.pdf}
\caption{Detailed trade-off for the viable coverage values. Green shading marks the stated quality envelope. B, Q, and A identify the balanced, quality-first, and aggressive presets. Moving right removes more of the 32 uniform samples and is the proxy for greater speed.}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.96\linewidth]{coverage_phase_transition.pdf}
\caption{Coverage sweep. The quality-envelope score is the worst normalised degradation across the four quality metrics; a score at or below one meets every limit. Low coverage gives greater sample reduction but sharply increases failure risk.}
\end{figure}

\begin{table}[H]
\centering
\caption{Factor-level summary across complete grid configurations. The score is reported as a median because failures at low coverage are highly skewed.}
\small
\begin{tabular}{@{}rrrrrr@{}}
\toprule
PF & Coverage & Mean samples & Removed (\%) & Median score & Passing grids \\
\midrule
@@FACTOR_LINES@@
\bottomrule
\end{tabular}
\end{table}

\section*{Data handling and limitations}
\begin{tabularx}{\linewidth}{@{}lX@{}}
\toprule
Sample counts & 32 configurations, eight Replica scenes, and three complete runs. Scene means are macro-averaged within each run; reported variability in the CSV is the sample SD across the three run macros. \\
Quality & The source labels all quality observations as run 1 and repeats scene/configuration pairs unevenly. Replicates are first averaged within each scene/configuration, then the eight scenes are macro-averaged so repeated rows cannot overweight a scene. \\
Baseline & Five-run baseline from \texttt{../replica\_depth\_spatial\_results.csv}: accuracy @@BASE_ACC@@ cm, completion @@BASE_COMP@@ cm, completion ratio @@BASE_RATIO@@\%, and Depth L1 @@BASE_DEPTH@@ mm. \\
Depth L1 & \texttt{office4} is excluded, matching the main experiment, because its values contain invalid multi-metre outliers. \\
Incomplete cell & @@INCOMPLETE@@ lacks the room2 quality result and is excluded from ranking and plots. \\
Timing interpretation & Sample reduction is a speed proxy, not measured wall-clock acceleration. Direct timing is required before claiming the same percentage as runtime speedup. \\
\bottomrule
\end{tabularx}

\end{document}
"""
    replacements = {
        "@@BALANCED_PF@@": str(balanced["PoolFactor"]),
        "@@BALANCED_GH@@": str(balanced["GridH"]),
        "@@BALANCED_GW@@": str(balanced["GridW"]),
        "@@BALANCED_MC@@": f"{float(balanced['MinCoverage']):.2f}",
        "@@BALANCED_SAMPLES@@": f"{float(balanced['MeanUniformSamples']):.2f}",
        "@@BALANCED_REMOVED@@": f"{float(balanced['UniformSampleReduction_pct']):.1f}",
        "@@REC_LINES@@": "\n".join(rec_lines),
        "@@QUALITY_SAMPLES@@": f"{float(quality_first['MeanUniformSamples']):.2f}",
        "@@AGGRESSIVE_REMOVED@@": f"{float(aggressive['UniformSampleReduction_pct']):.1f}",
        "@@FACTOR_LINES@@": "\n".join(factor_lines),
        "@@BASE_ACC@@": f"{baseline['Accuracy']:.3f}",
        "@@BASE_COMP@@": f"{baseline['Completion']:.3f}",
        "@@BASE_RATIO@@": f"{baseline['CompletionRatio']:.3f}",
        "@@BASE_DEPTH@@": f"{baseline['DepthL1']:.3f}",
        "@@INCOMPLETE@@": tex_escape(
            incomplete[0]["Configuration"] if incomplete else "None"
        ),
    }
    for token, value in replacements.items():
        tex = tex.replace(token, value)
    path = ROOT / "hyperparameter_study_report.tex"
    path.write_text(tex, encoding="utf-8")
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", path.name],
        cwd=ROOT, check=True,
    )
    subprocess.run(["latexmk", "-c", path.name], cwd=ROOT, check=True)


def main() -> None:
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })
    rows, baseline = aggregate()
    recommendations = choose_recommendations(rows)
    factors = factor_summary(rows)
    write_csv(ROOT / "hyperparameter_configuration_summary.csv", rows)
    write_csv(ROOT / "recommended_configurations.csv", recommendations)
    write_csv(ROOT / "factor_summary.csv", factors)
    plot_tradeoff(rows, recommendations)
    plot_phase_transition(rows)
    make_report(rows, baseline, recommendations, factors)


if __name__ == "__main__":
    main()
