#!/usr/bin/env python3
"""Collect Co-SLAM-style forward breakdowns from EC-SLAM CUDA events.

No Nsight profiler is used.  EC-SLAM writes the raw component events with the
same frame/outer-stage/stage structure as Co-SLAM's ``scene_rep`` timing mode,
plus an ``iteration_id`` needed to group concurrent EC-SLAM worker activity.
The script then reports table-ready means for Tracking and Mapping.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SETUPS = {
    "A": {"hash": "CoherentPrime", "morton_sort": False},
    "B": {"hash": "Morton", "morton_sort": False},
    "C": {"hash": "CoherentPrime", "morton_sort": True},
    "D": {"hash": "Morton", "morton_sort": True},
}
OUTER_STAGES = {
    "Tracking": "Tracking",
    "Mapping": "Bundle_Adjustment",
}
RAW_COLUMNS = (
    "frame_id",
    "outer_stage",
    "iteration_id",
    "stage",
    "cuda_ms",
)
DENOMINATOR_STAGES = (
    "render_rays",
    "compute_sdf_losses",
    "compute_render_losses",
)
SAMPLING_RENDERING_STAGES = (
    "depth_sampling",
    "volume_rendering",
    "compute_sdf_losses",
    "compute_render_losses",
)
COMPONENTS = {
    "hash_grid": ("TCNN_hashgrid_encoding",),
    "decoder_mlp": ("Decoder",),
    "sampling_rendering": SAMPLING_RENDERING_STAGES,
}
ITERATION_COLUMNS = (
    "stage",
    "sample_index",
    "frame_id",
    "iteration_id",
    "forward_denominator_ms",
    "hash_grid_ms",
    "decoder_mlp_ms",
    "sampling_rendering_ms",
    "other_ms",
    "hash_grid_percent",
    "decoder_mlp_percent",
    "sampling_rendering_percent",
    "other_percent",
)
SUMMARY_COLUMNS = (
    "System",
    "Stage",
    "Hardware",
    "Scene",
    "Setup",
    "HashSizeLog2",
    "Iterations",
    "HashGridEncodingPercent",
    "DecoderMLPPercent",
    "SamplingRenderingPercent",
    "OtherPercent",
    "MeanForwardDenominatorMs",
    "HashGridPercentStdDev",
    "DecoderMLPPercentStdDev",
    "SamplingRenderingPercentStdDev",
    "OtherPercentStdDev",
)


def read_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required. Run this script in the EC-SLAM environment."
        )
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def merge_dicts(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_merged_config(path: Path, default: Path | None = None) -> dict:
    leaf = read_yaml(path)
    inherited = leaf.get("inherit_from")
    if inherited:
        inherited_path = Path(inherited)
        if not inherited_path.is_absolute():
            inherited_path = REPO_ROOT / inherited_path
        base = load_merged_config(inherited_path, default)
    elif default is not None:
        base = read_yaml(default)
    else:
        base = {}
    return merge_dicts(base, leaf)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def safe_label(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return result or "hardware"


def automatic_frame_count(config: dict, iterations: int, warmup_frames: int) -> int:
    mapping_iters = int(config["mapping"]["iters"])
    every_frame = int(config["mapping"]["every_frame"])
    tracking_iters = int(config["tracking"]["iters"])
    if min(mapping_iters, every_frame, tracking_iters) <= 0:
        raise ValueError("tracking/mapping iterations and mapping.every_frame must be positive")

    calls_needed = math.ceil(iterations / mapping_iters)
    first_ba_frame = max(
        every_frame,
        math.ceil(warmup_frames / every_frame) * every_frame,
    )
    last_ba_frame = first_ba_frame + (calls_needed - 1) * every_frame
    frame_count = last_ba_frame + 1
    tracking_available = max(0, frame_count - max(1, warmup_frames)) * tracking_iters
    if tracking_available < iterations:
        frame_count += math.ceil(
            (iterations - tracking_available) / tracking_iters
        )
    return frame_count


def create_capture_config(
    args: argparse.Namespace,
    artifact_dir: Path,
    raw_csv: Path,
) -> tuple[Path, int]:
    scene_path = REPO_ROOT / "configs/Replica" / f"{args.scene}.yaml"
    if not scene_path.is_file():
        raise FileNotFoundError(f"Replica scene config not found: {scene_path}")
    merged = load_merged_config(scene_path, REPO_ROOT / "configs/EC_SLAM.yaml")
    minimum_frames = automatic_frame_count(
        merged, args.iterations, args.warmup_frames
    )
    frames = args.frames or minimum_frames
    if frames < minimum_frames:
        raise ValueError(
            f"--frames {frames} is too small for {args.iterations} mapping "
            f"iterations; use at least {minimum_frames}"
        )

    overlay = {
        "inherit_from": str(scene_path),
        "HashGrid": {
            **SETUPS[args.setup],
            "hash_size": args.hash_size,
        },
        "timing": {
            "mode": "scene_rep",
            "warmup_frames": args.warmup_frames,
            "max_frames": frames,
            "disable_eval": True,
            "scene_output_csv": str(raw_csv),
            "scene_max_iterations_per_stage": args.iterations,
        },
        "data": {
            "output": str(artifact_dir / "slam_output"),
            "exp_name": artifact_dir.name,
        },
    }
    config_path = artifact_dir / "scene_rep_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(overlay, handle, sort_keys=False)
    return config_path, frames


def read_raw_events(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"CUDA-event CSV not found: {path}")
    events = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise RuntimeError(
                f"Unexpected scene_rep CSV header in {path}: {reader.fieldnames}; "
                f"expected {list(RAW_COLUMNS)}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                frame_id = int(row["frame_id"])
                iteration_id = int(row["iteration_id"])
                cuda_ms = float(row["cuda_ms"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Invalid CUDA-event row at {path}:{line_number}"
                ) from error
            if frame_id < 0 or iteration_id < 0 or not math.isfinite(cuda_ms) or cuda_ms < 0:
                raise RuntimeError(
                    f"Invalid CUDA-event value at {path}:{line_number}"
                )
            events.append(
                {
                    "frame_id": frame_id,
                    "outer_stage": row["outer_stage"],
                    "iteration_id": iteration_id,
                    "stage": row["stage"],
                    "cuda_ms": cuda_ms,
                }
            )
    if not events:
        raise RuntimeError(f"CUDA-event CSV is empty: {path}")
    return events


def aggregate_iterations(
    events: list[dict[str, object]], iterations: int
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for event in events:
        key = (
            str(event["outer_stage"]),
            int(event["frame_id"]),
            int(event["iteration_id"]),
        )
        grouped[key][str(event["stage"])] += float(event["cuda_ms"])

    required = set(DENOMINATOR_STAGES)
    required.update(COMPONENTS["hash_grid"])
    required.update(COMPONENTS["decoder_mlp"])
    records = []
    for table_stage, outer_stage in OUTER_STAGES.items():
        candidates = sorted(
            (
                (frame_id, iteration_id, values)
                for (outer, frame_id, iteration_id), values in grouped.items()
                if outer == outer_stage
            ),
            key=lambda item: (item[0], item[1]),
        )
        if len(candidates) < iterations:
            raise RuntimeError(
                f"Requested {iterations} {table_stage} iterations, but {len(candidates)} "
                "were captured. Increase --frames."
            )

        for sample_index, (frame_id, iteration_id, stages) in enumerate(
            candidates[:iterations], start=1
        ):
            missing = sorted(required - stages.keys())
            if missing:
                raise RuntimeError(
                    f"{table_stage} frame {frame_id} iteration {iteration_id} is "
                    "missing CUDA stages: " + ", ".join(missing)
                )
            denominator = sum(stages[name] for name in DENOMINATOR_STAGES)
            components = {
                name: sum(stages[stage] for stage in stage_names)
                for name, stage_names in COMPONENTS.items()
            }
            other = denominator - sum(components.values())
            if denominator <= 0:
                raise RuntimeError(
                    f"Non-positive forward duration in {table_stage} frame {frame_id}"
                )
            tolerance = max(0.001, denominator * 0.005)
            if other < -tolerance:
                raise RuntimeError(
                    f"Overlapping CUDA categories in {table_stage} frame {frame_id} "
                    f"iteration {iteration_id}: Other={other:.6f} ms"
                )
            other = max(0.0, other)
            components["other"] = other
            record: dict[str, object] = {
                "stage": table_stage,
                "sample_index": sample_index,
                "frame_id": frame_id,
                "iteration_id": iteration_id,
                "forward_denominator_ms": denominator,
            }
            for name, value in components.items():
                record[f"{name}_ms"] = value
                record[f"{name}_percent"] = value / denominator * 100.0
            records.append(record)
    return records


def summarize(
    records: list[dict[str, object]], args: argparse.Namespace
) -> list[dict[str, object]]:
    output = []
    names = ("hash_grid", "decoder_mlp", "sampling_rendering", "other")
    for stage in OUTER_STAGES:
        rows = [row for row in records if row["stage"] == stage]
        denominator = statistics.fmean(
            float(row["forward_denominator_ms"]) for row in rows
        )
        means = {
            name: statistics.fmean(float(row[f"{name}_ms"]) for row in rows)
            for name in names
        }
        percentages = {name: means[name] / denominator * 100.0 for name in names}
        output.append(
            {
                "System": "EC-SLAM",
                "Stage": stage,
                "Hardware": args.hardware,
                "Scene": args.scene,
                "Setup": args.setup,
                "HashSizeLog2": args.hash_size,
                "Iterations": len(rows),
                "HashGridEncodingPercent": percentages["hash_grid"],
                "DecoderMLPPercent": percentages["decoder_mlp"],
                "SamplingRenderingPercent": percentages["sampling_rendering"],
                "OtherPercent": percentages["other"],
                "MeanForwardDenominatorMs": denominator,
                "HashGridPercentStdDev": statistics.stdev(
                    float(row["hash_grid_percent"]) for row in rows
                ),
                "DecoderMLPPercentStdDev": statistics.stdev(
                    float(row["decoder_mlp_percent"]) for row in rows
                ),
                "SamplingRenderingPercentStdDev": statistics.stdev(
                    float(row["sampling_rendering_percent"]) for row in rows
                ),
                "OtherPercentStdDev": statistics.stdev(
                    float(row["other_percent"]) for row in rows
                ),
            }
        )
    return output


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.9f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def print_summary(rows: list[dict[str, object]]) -> None:
    print("\nTable-ready CUDA-event means:")
    print("Stage       Hash-grid  Decoder/MLP  Sampling/rendering  Other")
    for row in rows:
        print(
            f"{row['Stage']:<11} "
            f"{row['HashGridEncodingPercent']:>8.1f}%  "
            f"{row['DecoderMLPPercent']:>10.1f}%  "
            f"{row['SamplingRenderingPercent']:>17.1f}%  "
            f"{row['OtherPercent']:>5.1f}%"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--scene", default="office0")
    parser.add_argument("--setup", choices=sorted(SETUPS), default="D")
    parser.add_argument("--hash-size", type=int, default=19, metavar="LOG2")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup-frames", type=int, default=1)
    parser.add_argument(
        "--frames",
        type=int,
        help="captured frames (default: minimum needed for requested mapping iterations)",
    )
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--summarize-only",
        type=Path,
        metavar="SCENE_REP_CSV",
        help="skip EC-SLAM and summarize an existing EC-SLAM CUDA-event CSV",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.iterations < 2 or args.hash_size <= 0:
        raise ValueError("--iterations must be at least 2 and --hash-size positive")
    if args.warmup_frames < 0 or (args.frames is not None and args.frames <= 1):
        raise ValueError("warm-up must be nonnegative and frames greater than one")

    run_name = f"{args.scene}_{args.setup}_T{args.hash_size}_{safe_label(args.hardware)}"
    artifact_dir = resolve(
        args.output_dir or Path("output/cuda_scene_rep") / run_name
    )
    if (
        not args.summarize_only
        and artifact_dir.exists()
        and any(artifact_dir.iterdir())
        and not args.overwrite
    ):
        raise FileExistsError(
            f"Output directory is not empty: {artifact_dir}; use --overwrite"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = (
        resolve(args.summarize_only)
        if args.summarize_only
        else artifact_dir / "scene_rep_timing.csv"
    )

    if not args.summarize_only:
        config_path, frames = create_capture_config(args, artifact_dir, raw_csv)
        command = [args.python_exec, str(REPO_ROOT / "run.py"), str(config_path)]
        print(
            f"CUDA-event capture: scene={args.scene}, setup={args.setup}, "
            f"frames={frames}, warmup_frames={args.warmup_frames}"
        )
        print("Running:", " ".join(command))
        if args.dry_run:
            return 0
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    records = aggregate_iterations(read_raw_events(raw_csv), args.iterations)
    summary = summarize(records, args)
    iteration_path = artifact_dir / "per_iteration_breakdown.csv"
    summary_path = artifact_dir / "table_summary.csv"
    write_csv(iteration_path, ITERATION_COLUMNS, records)
    write_csv(summary_path, SUMMARY_COLUMNS, summary)
    print_summary(summary)
    print(f"\nRaw component events: {raw_csv}")
    print(f"Per-iteration values: {iteration_path}")
    print(f"Table summary:        {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
