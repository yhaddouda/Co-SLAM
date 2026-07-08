#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import load_config


SYNTHETIC_SCENES = ["br", "ck", "gr", "gwr", "ma", "tg", "wr"]
TUM_SCENES = ["fr1_desk", "fr2_xyz", "fr3_office"]


def natural_key(path):
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", path.name)
    ]


def read_timestamp_list(path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            rows.append((float(fields[0]), fields[1:]))
    return rows


def tum_depth_paths(datadir, frame_rate=32):
    image_data = read_timestamp_list(datadir / "rgb.txt")
    depth_data = read_timestamp_list(datadir / "depth.txt")
    pose_filename = "groundtruth.txt" if (datadir / "groundtruth.txt").is_file() else "pose.txt"
    pose_data = read_timestamp_list(datadir / pose_filename)

    image_times = np.asarray([row[0] for row in image_data])
    depth_times = np.asarray([row[0] for row in depth_data])
    pose_times = np.asarray([row[0] for row in pose_data])

    associations = []
    for image_index, timestamp in enumerate(image_times):
        depth_index = int(np.argmin(np.abs(depth_times - timestamp)))
        pose_index = int(np.argmin(np.abs(pose_times - timestamp)))
        if (
            abs(depth_times[depth_index] - timestamp) < 0.08
            and abs(pose_times[pose_index] - timestamp) < 0.08
        ):
            associations.append((image_index, depth_index, pose_index))

    selected = [0]
    for index in range(1, len(associations)):
        previous_time = image_times[associations[selected[-1]][0]]
        current_time = image_times[associations[index][0]]
        if current_time - previous_time > 1.0 / frame_rate:
            selected.append(index)

    return [
        datadir / depth_data[associations[index][1]][1][0]
        for index in selected
    ]


def preprocess_depth(path, config):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")

    depth = depth.astype(np.float32)
    depth /= float(config["cam"]["png_depth_scale"])
    depth *= float(config["data"]["sc_factor"])

    downsample = int(config["data"]["downsample"])
    if downsample > 1:
        height, width = depth.shape
        depth = cv2.resize(
            depth,
            (width // downsample, height // downsample),
            interpolation=cv2.INTER_NEAREST,
        )

    crop_size = config["cam"].get("crop_size")
    if crop_size is not None:
        depth = cv2.resize(
            depth,
            (int(crop_size[1]), int(crop_size[0])),
            interpolation=cv2.INTER_NEAREST,
        )

    crop_edge = int(config["cam"].get("crop_edge", 0))
    if crop_edge > 0:
        depth = depth[crop_edge:-crop_edge, crop_edge:-crop_edge]

    return depth


def scene_depth_paths(dataset, scene, config):
    datadir = Path(config["data"]["datadir"])
    if dataset == "Synthetic":
        return sorted((datadir / "depth_filtered").glob("*.png"), key=natural_key)
    return tum_depth_paths(datadir)


def summarize_scene(dataset, scene, config):
    paths = scene_depth_paths(dataset, scene, config)
    if not paths:
        raise RuntimeError(f"No depth images found for {dataset}/{scene}")

    available_pixels = 0
    usable_pixels = 0
    within_far_pixels = 0
    total_pixels = 0
    available_frame_ratios = []
    usable_frame_ratios = []
    shapes = set()

    depth_trunc = float(config["cam"]["depth_trunc"])
    far = float(config["cam"]["far"])

    for path in paths:
        depth = preprocess_depth(path, config)
        finite = np.isfinite(depth)
        available = finite & (depth > 0)
        usable = available & (depth < depth_trunc)
        within_far = available & (depth <= far)

        pixels = depth.size
        total_pixels += pixels
        available_pixels += int(np.count_nonzero(available))
        usable_pixels += int(np.count_nonzero(usable))
        within_far_pixels += int(np.count_nonzero(within_far))
        available_frame_ratios.append(float(np.count_nonzero(available)) / pixels)
        usable_frame_ratios.append(float(np.count_nonzero(usable)) / pixels)
        shapes.add(depth.shape)

    if len(shapes) != 1:
        shape_text = ";".join(f"{height}x{width}" for height, width in sorted(shapes))
    else:
        height, width = next(iter(shapes))
        shape_text = f"{height}x{width}"

    available_array = np.asarray(available_frame_ratios)
    usable_array = np.asarray(usable_frame_ratios)
    return {
        "Dataset": dataset,
        "Scene": scene,
        "DepthSource": "depth_filtered" if dataset == "Synthetic" else "associated depth",
        "Frames": len(paths),
        "ProcessedShape": shape_text,
        "TotalPixels": total_pixels,
        "AvailablePixels": available_pixels,
        "AvailableDepthRatio": available_pixels / total_pixels,
        "MissingDepthRatio": 1.0 - available_pixels / total_pixels,
        "UsablePixels": usable_pixels,
        "UsableDepthRatio": usable_pixels / total_pixels,
        "WithinFarPixels": within_far_pixels,
        "WithinFarRatio": within_far_pixels / total_pixels,
        "FrameAvailableMin": float(available_array.min()),
        "FrameAvailableMedian": float(np.median(available_array)),
        "FrameAvailableMean": float(available_array.mean()),
        "FrameAvailableStd": float(available_array.std()),
        "FrameAvailableMax": float(available_array.max()),
        "FrameUsableMin": float(usable_array.min()),
        "FrameUsableMedian": float(np.median(usable_array)),
        "FrameUsableMean": float(usable_array.mean()),
        "FrameUsableStd": float(usable_array.std()),
        "FrameUsableMax": float(usable_array.max()),
        "DepthTrunc": depth_trunc,
        "Far": far,
    }


def aggregate_dataset(dataset, rows):
    dataset_rows = [row for row in rows if row["Dataset"] == dataset]
    total_pixels = sum(row["TotalPixels"] for row in dataset_rows)
    available_pixels = sum(row["AvailablePixels"] for row in dataset_rows)
    usable_pixels = sum(row["UsablePixels"] for row in dataset_rows)
    within_far_pixels = sum(row["WithinFarPixels"] for row in dataset_rows)
    return {
        "Dataset": dataset,
        "Scene": "ALL",
        "DepthSource": "configured scene inputs",
        "Frames": sum(row["Frames"] for row in dataset_rows),
        "ProcessedShape": "mixed",
        "TotalPixels": total_pixels,
        "AvailablePixels": available_pixels,
        "AvailableDepthRatio": available_pixels / total_pixels,
        "MissingDepthRatio": 1.0 - available_pixels / total_pixels,
        "UsablePixels": usable_pixels,
        "UsableDepthRatio": usable_pixels / total_pixels,
        "WithinFarPixels": within_far_pixels,
        "WithinFarRatio": within_far_pixels / total_pixels,
        "FrameAvailableMin": min(row["FrameAvailableMin"] for row in dataset_rows),
        "FrameAvailableMedian": "",
        "FrameAvailableMean": "",
        "FrameAvailableStd": "",
        "FrameAvailableMax": max(row["FrameAvailableMax"] for row in dataset_rows),
        "FrameUsableMin": min(row["FrameUsableMin"] for row in dataset_rows),
        "FrameUsableMedian": "",
        "FrameUsableMean": "",
        "FrameUsableStd": "",
        "FrameUsableMax": max(row["FrameUsableMax"] for row in dataset_rows),
        "DepthTrunc": "",
        "Far": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Quality/depth_availability_synthetic_tum.csv"),
    )
    args = parser.parse_args()

    rows = []
    for scene in SYNTHETIC_SCENES:
        config = load_config(f"configs/Synthetic/{scene}.yaml")
        rows.append(summarize_scene("Synthetic", scene, config))
    for scene in TUM_SCENES:
        config = load_config(f"configs/Tum/{scene}.yaml")
        rows.append(summarize_scene("TUM", scene, config))

    output_rows = []
    for dataset in ("Synthetic", "TUM"):
        output_rows.extend(row for row in rows if row["Dataset"] == dataset)
        output_rows.append(aggregate_dataset(dataset, rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
