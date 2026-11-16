#!/usr/bin/env python3
"""Evaluate Replica, TUM, Synthetic, or ScanNet with depth-spatial tracking and BA.

Select one or more datasets with --dataset (default: replica). --scenes takes
scene config stems, e.g. office0, fr1_desk, ck, or scene0059. Replica/Synthetic
use mesh metrics; TUM/ScanNet record final trajectory ATE RMSE in meters.
ScanNet defaults to ../datasets/data/scans/<scene>_00 with exported color,
depth, and pose folders. --dataset-root can select another exported scans root.

Examples (run from the repository root):
    python scripts/run/auto_eval_coslam_all_datasets_depth_spatial.py --dataset scannet tum synthetic --runs 3 --sizes 19
    python scripts/run/auto_eval_coslam_all_datasets_depth_spatial.py --dataset synthetic --scenes ck wr --sizes 19
"""

if __package__:
    from . import auto_eval_coslam_all_datasets_bucket as common
else:
    import auto_eval_coslam_all_datasets_bucket as common


MODE_DEFINITIONS = {
    "baseline": {
        "tracking_depth_spatial": False,
        "ba_bucket": False,
        "hash": "CoherentPrime",
        "morton_sort": False,
    },
    "depth_spatial": {
        "tracking_depth_spatial": True,
        "ba_bucket": True,
        "hash": "CoherentPrime",
        "morton_sort": False,
    },
    "depth_spatial_morton": {
        "tracking_depth_spatial": True,
        "ba_bucket": True,
        "hash": "Morton",
        "morton_sort": True,
    },
}


def apply_mode(cfg: dict, mode: str) -> None:
    mode_cfg = MODE_DEFINITIONS[mode]

    cfg.setdefault("grid", {})
    cfg.setdefault("training", {})
    cfg.setdefault("tracking_bucket", {})
    cfg.setdefault("tracking_naive_pruning", {})
    cfg.setdefault("tracking_depth_spatial", {})
    cfg.setdefault("ba_bucket", {})

    cfg["grid"]["hash"] = mode_cfg["hash"]
    cfg["grid"]["morton_sort"] = mode_cfg["morton_sort"]
    cfg["training"]["uniform_samples_until_depth"] = False
    cfg["tracking_bucket"]["enabled"] = False
    cfg["tracking_bucket"]["morton_inside_bucket"] = False
    cfg["tracking_naive_pruning"]["enabled"] = False
    cfg["tracking_depth_spatial"]["enabled"] = mode_cfg[
        "tracking_depth_spatial"
    ]
    cfg["ba_bucket"]["enabled"] = mode_cfg["ba_bucket"]


def main() -> None:
    # Reuse the bucket runner's run/cull/evaluate/CSV workflow with this
    # experiment's modes and isolated default outputs.
    common.DEFAULT_SAMPLING = list(MODE_DEFINITIONS)
    common.SAMPLING_DEFINITIONS = MODE_DEFINITIONS
    common.apply_sampling_mode = apply_mode

    common.main(
        dataset_support=True,
        default_output_csv="output/{dataset}_depth_spatial_results.csv",
        default_output_prefix="depth_spatial_eval",
    )


if __name__ == "__main__":
    main()
