#!/usr/bin/env python3
"""Aggregate the DASIP depth-sampling experiment and render its LaTeX tables."""

from __future__ import annotations

import csv
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev


ROOT = Path(__file__).resolve().parent
SCENES = (
    "office0", "office1", "office2", "office3", "office4",
    "room0", "room1", "room2",
)
MODES = ("baseline", "depth_spatial", "depth_spatial_morton")
MODE_LABELS = {
    "baseline": "Baseline",
    "depth_spatial": "Depth-spatial",
    "depth_spatial_morton": "Depth-spatial + Morton",
}
QUALITY_METRICS = (
    ("Accuracy", "Accuracy error", "cm", 1.0, SCENES),
    ("Completion", "Completion error", "cm", 1.0, SCENES),
    ("CompletionRatio", "Completion ratio", r"\%", 1.0, SCENES),
    ("DepthL1", "Depth L1", "mm", 1000.0,
     tuple(scene for scene in SCENES if scene != "office4")),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_unique(rows: list[dict[str, str]], keys: tuple[str, ...], source: Path) -> None:
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row[column] for column in keys)
        if key in seen:
            raise ValueError(f"Duplicate {key} in {source}")
        seen.add(key)


def complete_runs(
    rows: list[dict[str, str]], variant_column: str
) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["Run"])].append(row)
    expected = {(scene, mode) for scene in SCENES for mode in MODES}
    return {
        run: run_rows
        for run, run_rows in sorted(grouped.items())
        if {(row["Scene"], row[variant_column]) for row in run_rows} == expected
    }


def aggregate_quality() -> list[dict[str, object]]:
    source = ROOT / "replica_depth_spatial_results.csv"
    rows = read_rows(source)
    require_unique(rows, ("Run", "Scene", "Sampling"), source)
    runs = complete_runs(rows, "Sampling")
    if not runs:
        raise ValueError("No complete quality runs")

    output: list[dict[str, object]] = []
    for metric, label, unit, scale, scenes in QUALITY_METRICS:
        by_mode: dict[str, tuple[float, float]] = {}
        for mode in MODES:
            run_macros = []
            for run_rows in runs.values():
                values = [
                    float(row[metric]) * scale
                    for row in run_rows
                    if row["Sampling"] == mode and row["Scene"] in scenes
                ]
                if len(values) != len(scenes):
                    raise ValueError(f"Incomplete {metric}/{mode} run")
                run_macros.append(fmean(values))
            by_mode[mode] = (fmean(run_macros), stdev(run_macros))

        baseline = by_mode["baseline"][0]
        output.append({
            "Metric": metric,
            "Label": label,
            "Unit": unit.replace("\\", ""),
            "Scene_Count": len(scenes),
            "Excluded_Scenes": "office4" if "office4" not in scenes else "",
            "Run_Count": len(runs),
            **{
                f"{mode}_{field}": f"{value:.9f}"
                for mode, (mean, sd) in by_mode.items()
                for field, value in (("Mean", mean), ("Sample_SD", sd))
            },
            "depth_spatial_Change": f"{by_mode['depth_spatial'][0] - baseline:.9f}",
            "depth_spatial_Change_pct": (
                "" if metric == "CompletionRatio" else
                f"{100.0 * (by_mode['depth_spatial'][0] - baseline) / baseline:.9f}"
            ),
            "depth_spatial_morton_Change": (
                f"{by_mode['depth_spatial_morton'][0] - baseline:.9f}"
            ),
            "depth_spatial_morton_Change_pct": (
                "" if metric == "CompletionRatio" else
                f"{100.0 * (by_mode['depth_spatial_morton'][0] - baseline) / baseline:.9f}"
            ),
        })
    return output


def aggregate_timing() -> list[dict[str, object]]:
    source = ROOT / "Replica_depth_spatial_three_modes_runs_15W_run2.csv"
    rows = read_rows(source)
    require_unique(rows, ("Run", "Scene", "Mode"), source)
    runs = complete_runs(rows, "Mode")
    if not runs:
        raise ValueError("No complete timing runs")

    scopes = (
        ("tracking", "Tracking stage", "ms per stage", "tr_iter_profile_total_ms"),
        ("bundle_adjustment", "Bundle-adjustment invocation", "ms per invocation",
         "ba_iter_profile_total_ms"),
        ("frame", "Frame-wise Replica average", "ms per frame",
         "total_iter_profile_ms"),
    )
    values: dict[str, dict[str, list[float]]] = {
        scope: {mode: [] for mode in MODES} for scope, *_ in scopes
    }
    frames_root = ROOT / "depth_spatial_three_modes_15W_run2" / "Replica" / "frames"
    for run, run_rows in runs.items():
        for mode in MODES:
            scene_values = {scope: [] for scope, *_ in scopes}
            for manifest_row in run_rows:
                if manifest_row["Mode"] != mode:
                    continue
                frame_path = frames_root / Path(manifest_row["FrameTimingCsv"]).name
                frame_rows = read_rows(frame_path)
                require_unique(frame_rows, ("frame_id",), frame_path)
                by_frame = {int(row["frame_id"]): row for row in frame_rows}
                if sorted(by_frame) != list(range(1, 2000)):
                    raise ValueError(f"Expected frame ids 1--1999 in {frame_path}")
                for frame_id, frame_row in by_frame.items():
                    total = float(frame_row["total_iter_profile_ms"])
                    tracking = float(frame_row["tr_iter_profile_total_ms"])
                    ba = float(frame_row["ba_iter_profile_total_ms"])
                    if min(total, tracking, ba) < 0 or abs(total - tracking - ba) > 1e-6:
                        raise ValueError(f"Invalid timing at frame {frame_id} in {frame_path}")
                tracking_ids = [fid for fid, row in by_frame.items()
                                if float(row["tr_iter_profile_total_ms"]) > 0]
                ba_ids = [fid for fid, row in by_frame.items()
                          if float(row["ba_iter_profile_total_ms"]) > 0]
                if tracking_ids != list(range(1, 2000)):
                    raise ValueError(f"Unexpected tracking support in {frame_path}")
                if ba_ids != list(range(5, 2000, 5)):
                    raise ValueError(f"Unexpected BA support in {frame_path}")
                scene_values["tracking"].append(fmean(
                    float(by_frame[fid]["tr_iter_profile_total_ms"])
                    for fid in tracking_ids
                ))
                scene_values["bundle_adjustment"].append(fmean(
                    float(by_frame[fid]["ba_iter_profile_total_ms"])
                    for fid in ba_ids
                ))
                scene_values["frame"].append(fmean(
                    float(row["total_iter_profile_ms"]) for row in by_frame.values()
                ))
            for scope in scene_values:
                if len(scene_values[scope]) != len(SCENES):
                    raise ValueError(f"Incomplete {scope}/{mode}/run {run}")
                values[scope][mode].append(fmean(scene_values[scope]))

    output: list[dict[str, object]] = []
    for scope, label, unit, _ in scopes:
        baseline = fmean(values[scope]["baseline"])
        for mode in MODES:
            run_macros = values[scope][mode]
            mean = fmean(run_macros)
            output.append({
                "Scope": scope,
                "Scope_Label": label,
                "Unit": unit,
                "Mode": mode,
                "Mode_Label": MODE_LABELS[mode],
                "Power_Mode": "15 W",
                "Mean": f"{mean:.9f}",
                "Sample_SD": f"{stdev(run_macros):.9f}",
                "Reduction_vs_Baseline_pct": f"{100.0 * (baseline - mean) / baseline:.9f}",
                "Run_Count": len(runs),
                "Run_IDs": ";".join(map(str, runs)),
                "Scene_Count": len(SCENES),
                "Observations_Per_Scene_Run": 399 if scope == "bundle_adjustment" else 1999,
                "Iterations_Per_Observation": (
                    "10 tracking; +10 bundle adjustment every fifth frame"
                    if scope == "frame" else 10
                ),
                "Source": "depth_spatial_three_modes_15W_run2/Replica/frames/*.csv",
                "Aggregation": (
                    "observation mean within scene; 8-scene macro mean within run; "
                    "mean and sample SD across complete runs"
                ),
            })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pm(mean: float, sd: float, digits: int) -> str:
    return f"{mean:.{digits}f}$\\pm${sd:.{digits}f}"


def quality_change(row: dict[str, object], mode: str) -> str:
    change = float(row[f"{mode}_Change"])
    if row["Metric"] == "CompletionRatio":
        return rf"{change:+.3f} pp"
    pct = float(row[f"{mode}_Change_pct"])
    return rf"{change:+.3f} ({pct:+.1f}\%)"


def render_tex(quality: list[dict[str, object]], timing: list[dict[str, object]]) -> str:
    quality_lines = []
    for row in quality:
        digits = 3
        tex_unit = r"\%" if row["Unit"] == "%" else str(row["Unit"])
        quality_lines.append(
            f"{row['Label']} ({tex_unit}, "
            + (r"$\uparrow$" if row["Metric"] == "CompletionRatio" else r"$\downarrow$")
            + ") & "
            + pm(float(row["baseline_Mean"]), float(row["baseline_Sample_SD"]), digits)
            + " & "
            + pm(float(row["depth_spatial_Mean"]), float(row["depth_spatial_Sample_SD"]), digits)
            + " & " + quality_change(row, "depth_spatial")
            + " & "
            + pm(float(row["depth_spatial_morton_Mean"]), float(row["depth_spatial_morton_Sample_SD"]), digits)
            + " & " + quality_change(row, "depth_spatial_morton") + r" \\"
        )

    timing_lookup = {
        (str(row["Scope"]), str(row["Mode"])): row for row in timing
    }

    def timing_row(scope: str, label: str) -> str:
        cells = [label]
        for mode in MODES:
            row = timing_lookup[(scope, mode)]
            cells.append(pm(float(row["Mean"]), float(row["Sample_SD"]), 1))
        for mode in MODES[1:]:
            row = timing_lookup[(scope, mode)]
            cells.append(f"{float(row['Reduction_vs_Baseline_pct']):.1f}")
        return " & ".join(cells) + r" \\"

    return r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=16mm]{geometry}
\usepackage{booktabs,tabularx,array,xcolor}
\pagestyle{empty}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
\begin{document}
\begin{table}[ht]
\centering
\caption{Replica reconstruction quality for depth sampling during tracking. Values are mean $\pm$ sample standard deviation across five run-level scene averages. All metrics use eight scenes except Depth L1, which excludes \texttt{office4} because of invalid outliers. Changes are relative to the baseline; completion-ratio changes are percentage points.}
\label{tab:depth-spatial-quality}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrrrr@{}}
\toprule
Metric & Baseline & \shortstack{Depth-\\spatial} & Change & \shortstack{Depth-spatial\\+ Morton} & Change \\
\midrule
""" + "\n".join(quality_lines) + r"""
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[ht]
\centering
\caption{Optimisation runtime on Replica in AGX Orin 15\,W mode. Values are mean $\pm$ sample standard deviation across five complete run-level scene averages (runs 1--5; eight scenes per run). Component values sum ten profiled iterations; frame-wise values average the noninitial frames. Reductions are relative to the baseline.}
\label{tab:depth-spatial-runtime}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrrrr@{}}
\toprule
Scope & Baseline & \shortstack{Depth-\\spatial} & \shortstack{+ Morton} & \shortstack{Spatial red.\\(\%)} & \shortstack{+ Morton red.\\(\%)} \\
\midrule
\multicolumn{6}{@{}l}{\textbf{(a) Ten-iteration component execution time (ms, $\downarrow$)}} \\
""" + timing_row("tracking", "Tracking stage") + "\n" + timing_row(
        "bundle_adjustment", "Bundle-adjustment invocation"
    ) + r"""
\midrule
\multicolumn{6}{@{}l}{\textbf{(b) Frame-wise optimisation execution time (ms/frame, $\downarrow$)}} \\
""" + timing_row("frame", "Replica average") + r"""
\bottomrule
\end{tabularx}
\end{table}
\end{document}
"""


def main() -> None:
    quality = aggregate_quality()
    timing = aggregate_timing()
    write_csv(ROOT / "quality_summary.csv", quality)
    write_csv(ROOT / "timing_summary_15W.csv", timing)
    tex_path = ROOT / "results_tables.tex"
    tex_path.write_text(render_tex(quality, timing), encoding="utf-8")
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(["latexmk", "-c", tex_path.name], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
