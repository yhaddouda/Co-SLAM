# Cropped Sampling Statistics Guide

This guide explains how to collect and analyze the statistics produced by the cropped sampling path in [`model/scene_rep.py`](../model/scene_rep.py).

## What Is Collected

When cropped sampling is active, the code can collect two kinds of statistics:

1. Per-ray counts inside a batch:
   the number of uniform samples that fall before the target depth for each valid ray in that batch.
2. Per-batch selected counts across a run:
   the maximum per-ray count chosen for the batch, which becomes the batch-wide `n_uniform` value.

The collection logic is opt-in and stays disabled unless the dedicated flag is enabled.

## Enable Statistics Collection

In your config, use the `training` block in [`configs/Replica/replica.yaml`](../configs/Replica/replica.yaml) or in a scene-specific config:

```yaml
training:
  uniform_samples_until_depth: True
  collect_uniform_samples_until_depth_stats: True
  uniform_samples_until_depth_stats_csv: null
```

Important notes:

- `uniform_samples_until_depth: True` enables the cropped sampling behavior itself.
- `collect_uniform_samples_until_depth_stats: True` enables statistics recording.
- `uniform_samples_until_depth_stats_csv: null` uses the default output path.
- If `collect_uniform_samples_until_depth_stats` is `True` but `uniform_samples_until_depth` is `False`, the collector is present but no cropped-sampling rows are generated.

## Default Output File

If `uniform_samples_until_depth_stats_csv` is left as `null`, the CSV is saved to:

```text
<data.output>/<data.exp_name>/uniform_samples_until_depth_stats.csv
```

Example:

```text
output/Replica/office0/run_1/baseline_A_Size19_Run1/uniform_samples_until_depth_stats.csv
```

## CSV Schema

The collector writes one row per render batch with the following columns:

- `frame_id`
- `outer_stage`
- `batch_id`
- `selected_uniform_samples`
- `requested_uniform_samples`
- `valid_ray_count`
- `total_ray_count`
- `ray_samples_before_depth`

### Meaning Of The Key Fields

- `selected_uniform_samples`:
  the maximum count selected to represent the whole batch.
- `requested_uniform_samples`:
  the configured upper limit, for example `32`.
- `ray_samples_before_depth`:
  a JSON list with one entry per ray in the batch.
  Valid rays store an integer count, and invalid-depth rays store `null`.

## Run Co-SLAM

Once the flags are enabled, run Co-SLAM normally. For example:

```bash
python coslam.py --config configs/Replica/office0.yaml
```

Or through the batch runner:

```bash
python auto_run_replica_coslam_many_runs.py \
  --mode cropped \
  --runs 1 \
  --setups A \
  --sizes 19 \
  --scenes office0
```

## Plot The Histograms

Use [`scripts/plot/plot_uniform_samples_until_depth_histograms.py`](../scripts/plot/plot_uniform_samples_until_depth_histograms.py):

```bash
python scripts/plot/plot_uniform_samples_until_depth_histograms.py \
  --stats-csv output/Replica/.../uniform_samples_until_depth_stats.csv \
  --batch-ids 0 10 25
```

Optional flags:

- `--output-dir` to choose where plots are written
- `--prefix` to customize the output file prefix

## What The Plot Script Produces

### 1. Per-batch Histogram Across The Run

This always produces:

```text
<prefix>_per_batch_selected_hist.png
```

This histogram shows how often each batch-level selected value was used during the run.

### 2. Per-ray Histogram For Specific Batches

When you pass `--batch-ids`, the script produces one plot per requested batch:

```text
<prefix>_batch_<batch_id>_per_ray_hist.png
```

Each of these plots shows the per-ray counts only for that batch.

## Batch-ID Discovery

The plot script prints the available batch IDs before plotting, for example:

```text
Available batch IDs: 0..377 (378 total)
```

If a requested batch ID is missing, the script prints the available range in the error message.

## Histogram Behavior

Both histogram types use integer-valued x-axis ticks so the axis matches the discrete sample counts.

## Typical Workflow

1. Enable `uniform_samples_until_depth`.
2. Enable `collect_uniform_samples_until_depth_stats`.
3. Run the experiment.
4. Locate `uniform_samples_until_depth_stats.csv`.
5. Run the plotting script for the full-run batch histogram and for any batch IDs you want to inspect in detail.
