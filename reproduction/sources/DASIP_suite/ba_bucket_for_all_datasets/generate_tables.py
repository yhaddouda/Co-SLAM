#!/usr/bin/env python3
"""Render one paper-style quality table per dataset from the collected CSVs."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev


ROOT = Path(__file__).resolve().parent
MODES = ("baseline", "depth_spatial", "depth_spatial_morton")
MODE_LABELS = ("Baseline", "Depth-spatial", "Depth-spatial + Morton")
PAPER = (
    "https://openaccess.thecvf.com/content/CVPR2023/papers/"
    "Wang_Co-SLAM_Joint_Coordinate_and_Sparse_Parametric_Encodings_for_"
    "Neural_Real-Time_CVPR_2023_paper.pdf"
)


@dataclass(frozen=True)
class Metric:
    column: str
    label: str
    unit: str = "cm"
    scale: float = 1.0
    digits: int = 2
    higher_is_better: bool = False

    @property
    def heading(self) -> str:
        arrow = "↑" if self.higher_is_better else "↓"
        return f"{self.label} ({self.unit}) {arrow}"


ATE = Metric("ATE_RMSE_m", "ATE RMSE", scale=100.0)
# The reconstruction evaluator prints all distance metrics, including Depth L1,
# in cm. Preserve its values; only the explicitly metre-valued ATE columns scale.
RECONSTRUCTION = (
    Metric("DepthL1", "Depth L1", digits=5),
    Metric("Accuracy", "Accuracy"),
    Metric("Completion", "Completion"),
    Metric("CompletionRatio", "Completion ratio", "%", higher_is_better=True),
)


@dataclass(frozen=True)
class Estimate:
    mean: float
    sample_sd: float | None


def estimate(values: list[float]) -> Estimate:
    return Estimate(fmean(values), stdev(values) if len(values) > 1 else None)


@dataclass
class Dataset:
    key: str
    label: str
    scenes: tuple[str, ...]
    scene_labels: tuple[str, ...]
    metrics: tuple[Metric, ...]
    repetitions: int = 0
    hash_size: str = ""
    repeated_run_labels: bool = False

    @property
    def source(self) -> Path:
        return ROOT / f"{self.key}_depth_spatial_results.csv"


def datasets() -> list[Dataset]:
    return [
        Dataset(
            "scannet", "ScanNet",
            ("scene0000", "scene0059", "scene0106", "scene0169", "scene0181", "scene0207"),
            ("0000", "0059", "0106", "0169", "0181", "0207"),
            (ATE,),
        ),
        Dataset(
            "tum", "TUM RGB-D", ("fr1_desk", "fr2_xyz", "fr3_office"),
            ("fr1/desk", "fr2/xyz", "fr3/office"), (ATE,),
        ),
        Dataset(
            "synthetic", "Synthetic RGB-D", ("br", "ck", "gr", "gwr", "ma", "tg", "wr"),
            ("BR", "CK", "GR", "GWR", "MA", "TG", "WR"),
            (*RECONSTRUCTION, ATE),
        ),
    ]


def aggregate(dataset: Dataset) -> tuple[dict, list[dict], list[dict]]:
    with dataset.source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Run", "Scene", "Sampling", "HashSize"}
        required.update(metric.column for metric in dataset.metrics)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{dataset.source.name}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{dataset.source.name}: no observations")
    hash_sizes = {row["HashSize"] for row in rows}
    if len(hash_sizes) != 1:
        raise ValueError(f"{dataset.source.name}: cannot combine hash sizes {hash_sizes}")
    dataset.hash_size = hash_sizes.pop()

    grouped = defaultdict(list)
    run_keys = Counter()
    for row in rows:
        grouped[row["Scene"], row["Sampling"]].append(row)
        run_keys[row["Run"], row["Scene"], row["Sampling"]] += 1
    expected = {(scene, mode) for scene in dataset.scenes for mode in MODES}
    if set(grouped) != expected:
        raise ValueError(
            f"{dataset.source.name}: missing {sorted(expected - set(grouped))}; "
            f"unexpected {sorted(set(grouped) - expected)}"
        )
    counts = {len(observations) for observations in grouped.values()}
    if len(counts) != 1:
        raise ValueError(f"{dataset.source.name}: unequal repetition counts {counts}")
    dataset.repetitions = counts.pop()
    dataset.repeated_run_labels = any(count > 1 for count in run_keys.values())

    # The CSVs store complete run blocks in order. A reused Run label starts a
    # distinct repetition, so ScanNet's blocks 1, 2, 1, 2, 3 remain five samples.
    repetitions = []
    previous_run = None
    for row in rows:
        key = (row["Scene"], row["Sampling"])
        if (not repetitions or row["Run"] != previous_run
                or key in repetitions[-1]):
            repetitions.append({})
        repetitions[-1][key] = row
        previous_run = row["Run"]
    if len(repetitions) != dataset.repetitions or any(
        set(repetition) != expected for repetition in repetitions
    ):
        raise ValueError(f"{dataset.source.name}: expected complete contiguous run blocks")

    estimates = {}
    scene_summary = []
    dataset_summary = []
    for metric in dataset.metrics:
        for mode in MODES:
            for scene in dataset.scenes:
                values = [float(row[metric.column]) * metric.scale for row in grouped[scene, mode]]
                if any(not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError(f"{dataset.source.name}: invalid {metric.column}/{scene}/{mode}")
                if metric.unit == "%" and any(value > 100 for value in values):
                    raise ValueError(f"{dataset.source.name}: completion ratio exceeds 100%")
                stats = estimate(values)
                estimates[metric.column, mode, scene] = stats
                scene_summary.append({
                    "Dataset": dataset.label, "Scene": scene, "Sampling": mode,
                    "HashSize": dataset.hash_size, "Metric": metric.column,
                    "Unit": metric.unit, "Mean": stats.mean,
                    "Sample_SD": stats.sample_sd if stats.sample_sd is not None else "",
                    "Observation_Count": len(values), "Source": dataset.source.name,
                })
            # Average all scenes within each repetition, then measure repeat
            # variability of that dataset average (rather than scene variability).
            repetition_means = [
                fmean(float(repetition[scene, mode][metric.column]) * metric.scale
                      for scene in dataset.scenes)
                for repetition in repetitions
            ]
            stats = estimate(repetition_means)
            estimates[metric.column, mode, "Avg."] = stats
            dataset_summary.append({
                "Dataset": dataset.label, "Sampling": mode, "HashSize": dataset.hash_size,
                "Metric": metric.column, "Unit": metric.unit, "Mean": stats.mean,
                "Sample_SD": stats.sample_sd if stats.sample_sd is not None else "",
                "Scene_Count": len(dataset.scenes),
                "Observations_Per_Scene": dataset.repetitions,
                "Observation_Count": len(dataset.scenes) * dataset.repetitions,
                "Repetition_Count": len(repetition_means),
                "SD_Aggregation": "sample SD across repetition-level scene averages",
                "Source": dataset.source.name,
            })
    return estimates, scene_summary, dataset_summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def best_value(estimates: dict, metric: Metric, scene: str) -> float:
    values = [estimates[metric.column, mode, scene].mean for mode in MODES]
    return max(values) if metric.higher_is_better else min(values)


def cell(estimates: dict, metric: Metric, mode: str, scene: str, tex: bool) -> str:
    stats = estimates[metric.column, mode, scene]
    value = stats.mean
    mean = f"{value:.{metric.digits}f}"
    sd = f"{stats.sample_sd:.{metric.digits}f}" if stats.sample_sd is not None else "n/a"
    best = best_value(estimates, metric, scene)
    if tex:
        # Stack the longer Depth L1 values to retain five decimals legibly.
        formatted = (rf"\shortstack{{{mean}\\$\pm${sd}}}" if metric.column == "DepthL1"
                     else rf"{mean}$\pm${sd}")
    else:
        formatted = f"{mean} ± {sd}"
    if math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-12):
        return rf"\textbf{{{formatted}}}" if tex else f"**{formatted}**"
    return formatted


def caption(dataset: Dataset) -> str:
    return (
        f"Quality comparison on {dataset.label}. Scene values are mean "
        f"$\\pm$ sample standard deviation across {dataset.repetitions} repetitions. "
        f"Avg. reports mean $\\pm$ sample standard deviation of repetition-level "
        f"averages over all {len(dataset.scenes)} scenes. Bold marks the best unrounded mean "
        "among the three variants in each metric and column."
    )


def render_table(dataset: Dataset, estimates: dict) -> str:
    columns = len(dataset.scenes) + 2
    lines = [
        r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption(dataset)}}}",
        rf"\label{{tab:{dataset.key}-quality}}", r"\vspace{5pt}", r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}", r"\renewcommand{\arraystretch}{1.18}",
        r"\begin{tabularx}{\linewidth}{@{}l*{" + str(columns - 1) + r"}{Y}@{}}",
        r"\toprule",
        " & ".join(("Method", *dataset.scene_labels, r"\textbf{Avg.}")) + r" \\",
    ]
    for metric in dataset.metrics:
        arrow = r"$\uparrow$" if metric.higher_is_better else r"$\downarrow$"
        unit = r"\%" if metric.unit == "%" else metric.unit
        heading = f"{metric.label} ({unit}) {arrow}"
        lines.extend([r"\midrule", rf"\multicolumn{{{columns}}}{{@{{}}l}}{{\textit{{{heading}}}}} \\"])
        for mode, label in zip(MODES, MODE_LABELS):
            cells = [label]
            cells.extend(cell(estimates, metric, mode, scene, True)
                         for scene in (*dataset.scenes, "Avg."))
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def render_markdown(all_datasets: list[Dataset], all_estimates: dict) -> str:
    lines = [
        "# Quality comparison across datasets", "",
        "One table per dataset, following the scene columns and average column in "
        f"[the Co-SLAM paper]({PAPER}). Each metric compares Baseline, Depth-spatial, "
        "and Depth-spatial + Morton. HashSize = 19 in all input rows.", "",
        "Each cell shows mean ± sample standard deviation (SD, denominator n − 1). "
        "Scene SDs measure variability across repetitions. Avg. first averages all "
        "scenes within each repetition, then reports the mean and sample SD of those "
        "dataset averages. Each complete contiguous run block in CSV order is one "
        "repetition, including blocks with reused run labels. Dataset means give "
        "every scene equal weight and use unrounded values. No "
        "measurements or scenes are excluded. Bold marks the best unrounded mean "
        "among variants for each metric and column; ↓ means lower is better and ↑ "
        "means higher is better.", "",
        "ATE is converted from metres to centimetres. Accuracy, completion, and "
        "Depth L1 retain the centimetre values printed by "
        "`neural_slam_eval/eval_recon_headless.py`; completion ratio is a percentage. "
        "Depth L1 uses five decimal places to preserve the small values in the CSV.", "",
        "ScanNet contains five measurements per scene and variant, with run labels "
        "1 and 2 reused for different results. All five rows are retained. TUM and "
        "Synthetic each contain three measurements per scene and variant.", "",
    ]
    for dataset in all_datasets:
        estimates = all_estimates[dataset.key]
        lines.extend([f"## {dataset.label}", "", caption(dataset).replace("$\\pm$", "±"), "",
                      f"Source: `{dataset.source.name}`.", ""])
        lines.append("| " + " | ".join(("Metric", "Method", *dataset.scene_labels, "Avg.")) + " |")
        lines.append("| " + " | ".join(("---", "---", *(["---:"] * (len(dataset.scenes) + 1)))) + " |")
        for metric in dataset.metrics:
            for mode, label in zip(MODES, MODE_LABELS):
                cells = [metric.heading, label]
                cells.extend(cell(estimates, metric, mode, scene, False)
                             for scene in (*dataset.scenes, "Avg."))
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend([
        "Synthetic scene abbreviations: BR = breakfast room; CK = complete kitchen; "
        "GR = green room; GWR = grey white room; MA = morning apartment; "
        "TG = thin geometry; WR = white room.", "",
        "## Regeneration", "", "```bash",
        "python DASIP_suite/ba_bucket_for_all_datasets/generate_tables.py", "```", "",
        "The script uses only the Python standard library and `pdflatex` for PDF "
        "generation. Use `--no-pdf` to generate Markdown, CSV, and LaTeX only. "
        "The three `*_quality_table.tex` fragments can be included in a manuscript "
        "with `booktabs`, `tabularx`, and `array`, plus "
        r"`\newcolumntype{Y}{>{\centering\arraybackslash}X}`.", "",
        "`scene_quality_summary.csv` preserves unrounded scene means, sample "
        "standard deviations, and observation counts. `dataset_quality_summary.csv` "
        "preserves unrounded dataset means, sample SDs, and counts. The three "
        "`*_quality_table.csv` files provide scene means and Avg. with separate "
        "sample SD columns.", "",
    ])
    return "\n".join(lines)


def render_document(all_datasets: list[Dataset]) -> str:
    lines = [
        r"\documentclass[10pt]{article}", r"\usepackage[a4paper,margin=17mm]{geometry}",
        r"\usepackage[T1]{fontenc}", r"\pdfmapfile{+lm.map}", r"\usepackage{lmodern}",
        r"\input{glyphtounicode}", r"\pdfgentounicode=1",
        r"\usepackage{booktabs,tabularx,array}", r"\usepackage[hidelinks]{hyperref}",
        r"\newcolumntype{Y}{>{\centering\arraybackslash}X}",
        r"\setlength{\parindent}{0pt}", r"\begin{document}",
        r"{\large\bfseries Quality comparison across datasets}\par\medskip",
        r"\small Three variants, with HashSize = 19 throughout. Values are mean "
        r"$\pm$ sample standard deviation (SD, denominator $n-1$). Scene means use "
        r"all measurements; dataset means give every scene equal weight. "
        r"$\downarrow$: lower is better. $\uparrow$: higher is better.\par",
    ]
    for dataset in all_datasets:
        if dataset.key == "synthetic":
            lines.append(r"\clearpage")
        lines.append(rf"\input{{{dataset.key}_quality_table.tex}}")
    lines.extend([
        r"\footnotesize BR: breakfast room; CK: complete kitchen; GR: green room; "
        r"GWR: grey white room; MA: morning apartment; TG: thin geometry; "
        r"WR: white room.\par\medskip",
        r"ATE is converted from metres to cm. Reconstruction distances retain the "
        r"cm values printed by the evaluation script. Depth L1 uses five decimal "
        r"places to preserve the small recorded values.\par\medskip",
        r"Scene SDs use the repeated measurements for each scene and variant. "
        r"Avg. SDs use the dataset averages calculated separately within each "
        r"repetition. Each complete contiguous run block in CSV order is treated "
        r"as one repetition.\par\medskip",
        r"ScanNet run labels 1 and 2 are reused for different results; all five "
        r"measurements per scene and variant are retained. TUM and Synthetic "
        r"have three measurements each. No measurements or scenes are excluded.\par\medskip",
        rf"Layout reference: \href{{{PAPER}}}{{Co-SLAM (CVPR 2023)}}. "
        r"All numerical results in these tables come from the supplied CSVs.",
        r"\end{document}", "",
    ])
    return "\n".join(lines)


def compile_pdf() -> None:
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex is required for the PDF; use --no-pdf to skip it")
    with tempfile.TemporaryDirectory(prefix="coslam-quality-tables-") as directory:
        result = subprocess.run(
            [executable, "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={directory}", "results_tables.tex"],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout[-6000:] + result.stderr)
        for line in result.stdout.splitlines():
            if "Overfull" in line:
                print(line)
        shutil.copyfile(Path(directory) / "results_tables.pdf", ROOT / "results_tables.pdf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF compilation")
    args = parser.parse_args()
    all_datasets = datasets()
    all_estimates = {}
    scene_summary = []
    dataset_summary = []
    # Validate all sources before writing any output.
    for dataset in all_datasets:
        estimates, scene_rows, dataset_rows = aggregate(dataset)
        all_estimates[dataset.key] = estimates
        scene_summary.extend(scene_rows)
        dataset_summary.extend(dataset_rows)
    for dataset in all_datasets:
        estimates = all_estimates[dataset.key]
        (ROOT / f"{dataset.key}_quality_table.tex").write_text(
            render_table(dataset, estimates), encoding="utf-8"
        )
        table_rows = []
        for metric in dataset.metrics:
            for mode, label in zip(MODES, MODE_LABELS):
                table_rows.append({
                    "Metric": metric.label, "Unit": metric.unit, "Method": label,
                    **{
                        column: value
                        for scene in (*dataset.scenes, "Avg.")
                        for column, value in (
                            (scene, estimates[metric.column, mode, scene].mean),
                            (f"{scene}_Sample_SD", estimates[metric.column, mode, scene].sample_sd),
                        )
                    },
                })
        write_csv(ROOT / f"{dataset.key}_quality_table.csv", table_rows)
        print(f"{dataset.label}: {len(dataset.scenes)} scenes, "
              f"{dataset.repetitions} measurements per scene/variant")
    write_csv(ROOT / "scene_quality_summary.csv", scene_summary)
    write_csv(ROOT / "dataset_quality_summary.csv", dataset_summary)
    (ROOT / "results_tables.md").write_text(render_markdown(all_datasets, all_estimates), encoding="utf-8")
    (ROOT / "results_tables.tex").write_text(render_document(all_datasets), encoding="utf-8")
    if not args.no_pdf:
        compile_pdf()
        print(f"PDF: {ROOT / 'results_tables.pdf'}")


if __name__ == "__main__":
    main()
