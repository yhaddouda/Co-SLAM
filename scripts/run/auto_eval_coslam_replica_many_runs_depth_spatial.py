#!/usr/bin/env python3
"""Evaluate Replica runs with depth-spatial tracking and bucketed BA modes."""

import sys

if __package__:
    from . import auto_eval_coslam_replica_many_runs_bucket as common
else:
    import auto_eval_coslam_replica_many_runs_bucket as common


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


def add_default_argument(flag: str, value: str) -> None:
    if not any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv):
        sys.argv.extend([flag, value])


def main() -> None:
    # Reuse the bucket runner's run/cull/evaluate/CSV workflow with this
    # experiment's modes and isolated default outputs.
    common.DEFAULT_SAMPLING = list(MODE_DEFINITIONS)
    common.SAMPLING_DEFINITIONS = MODE_DEFINITIONS
    common.apply_sampling_mode = apply_mode

    add_default_argument(
        "--output-csv", "output/replica_depth_spatial_results.csv"
    )
    add_default_argument("--output-prefix", "depth_spatial_eval")
    common.main()


if __name__ == "__main__":
    main()
