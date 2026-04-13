import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_sampling_stats(stats_csv: Path):
    batch_stats = {}
    per_batch_selected = []

    with open(stats_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {
            "batch_id",
            "frame_id",
            "outer_stage",
            "selected_uniform_samples",
            "requested_uniform_samples",
            "ray_samples_before_depth",
        }
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing columns in {stats_csv}: {sorted(missing)}"
            )

        for row in reader:
            batch_id = int(row["batch_id"])
            selected_uniform_samples = int(row["selected_uniform_samples"])
            counts = json.loads(row["ray_samples_before_depth"])
            batch_stats[batch_id] = {
                "frame_id": int(row["frame_id"]),
                "outer_stage": row["outer_stage"],
                "selected_uniform_samples": selected_uniform_samples,
                "requested_uniform_samples": int(row["requested_uniform_samples"]),
                "ray_counts": [int(count) for count in counts if count is not None],
            }
            per_batch_selected.append(selected_uniform_samples)

    return batch_stats, per_batch_selected


def integer_hist_bins(values):
    min_value = min(values)
    max_value = max(values)
    return [value - 0.5 for value in range(min_value, max_value + 2)]


def format_batch_id_summary(batch_ids):
    sorted_ids = sorted(batch_ids)
    if not sorted_ids:
        return "no batches available"

    min_batch = sorted_ids[0]
    max_batch = sorted_ids[-1]
    count = len(sorted_ids)

    if sorted_ids == list(range(min_batch, max_batch + 1)):
        return f"{min_batch}..{max_batch} ({count} total)"

    preview_count = min(10, count)
    preview = ", ".join(str(batch_id) for batch_id in sorted_ids[:preview_count])
    if count > preview_count:
        preview += ", ..."
    return f"{preview} ({count} total, min={min_batch}, max={max_batch})"


def set_integer_x_axis(ax, values):
    min_value = min(values)
    max_value = max(values)
    ax.set_xticks(list(range(min_value, max_value + 1)))
    ax.set_xlim(min_value - 0.5, max_value + 0.5)


def save_histogram(values, title: str, xlabel: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=integer_hist_bins(values), edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    set_integer_x_axis(ax, values)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot histograms for uniform_samples_until_depth statistics."
    )
    parser.add_argument("--stats-csv", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default="uniform_samples_until_depth")
    parser.add_argument(
        "--batch-ids",
        nargs="+",
        type=int,
        default=None,
        help="Batch IDs to plot separately for the per-ray histogram.",
    )
    args = parser.parse_args()

    stats_csv = args.stats_csv
    if not stats_csv.exists():
        raise SystemExit(f"Stats CSV not found: {stats_csv}")

    output_dir = args.output_dir or stats_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_stats, per_batch_selected = load_sampling_stats(stats_csv)
    if not batch_stats:
        raise SystemExit("No cropped-sampling batch data found in the stats CSV.")
    if not per_batch_selected:
        raise SystemExit("No batch-level cropped-sampling counts found in the stats CSV.")

    available_batch_ids = sorted(batch_stats)
    print(f"Available batch IDs: {format_batch_id_summary(available_batch_ids)}")

    batch_hist_path = output_dir / f"{args.prefix}_per_batch_selected_hist.png"

    saved_ray_hist_paths = []
    if args.batch_ids:
        for batch_id in args.batch_ids:
            if batch_id not in batch_stats:
                raise SystemExit(
                    f"Batch ID {batch_id} not found in {stats_csv}. "
                    f"Available batch IDs: {format_batch_id_summary(available_batch_ids)}"
                )
            ray_counts = batch_stats[batch_id]["ray_counts"]
            if not ray_counts:
                raise SystemExit(f"Batch ID {batch_id} has no valid per-ray counts to plot.")

            ray_hist_path = output_dir / f"{args.prefix}_batch_{batch_id}_per_ray_hist.png"
            save_histogram(
                ray_counts,
                title=(
                    f"Per-ray samples before depth - batch {batch_id}\n"
                    f"frame={batch_stats[batch_id]['frame_id']}, "
                    f"stage={batch_stats[batch_id]['outer_stage']}"
                ),
                xlabel="Number of uniform samples before depth",
                output_path=ray_hist_path,
            )
            saved_ray_hist_paths.append((batch_id, ray_hist_path, ray_counts))

    save_histogram(
        per_batch_selected,
        title="Per-batch selected uniform sample count",
        xlabel="Selected maximum samples kept for batch",
        output_path=batch_hist_path,
    )

    for batch_id, ray_hist_path, ray_counts in saved_ray_hist_paths:
        print(f"Saved per-ray histogram for batch {batch_id} to {ray_hist_path}")
        print(
            f"Batch {batch_id} per-ray counts: {len(ray_counts)} values, "
            f"min={min(ray_counts)}, max={max(ray_counts)}"
        )
    print(f"Saved per-batch histogram to {batch_hist_path}")
    if not args.batch_ids:
        print("No --batch-ids were provided, so only the per-batch histogram was generated.")
    print(
        f"Per-batch selected counts: {len(per_batch_selected)} values, "
        f"min={min(per_batch_selected)}, max={max(per_batch_selected)}"
    )


if __name__ == "__main__":
    main()
