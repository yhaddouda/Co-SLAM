"""Shared artifact, evaluation and statistics helpers for execution experiments."""
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch


def seed_phase(seed, frame=0, phase=0):
    value = seed * 1000003 + frame * 17 + phase
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def statistics(values):
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        return None
    return dict(count=len(data), mean_ms=float(data.mean()), median_ms=float(np.median(data)),
                p95_ms=float(np.percentile(data, 95)), std_ms=float(data.std()))


def write_csv(path, records):
    if not records:
        return
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def snapshot(output, source_root, names):
    hashes = {}
    for name in names:
        source = Path(source_root) / name
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (output / 'source_hashes.json').write_text(json.dumps(hashes, indent=2))


def device_metadata():
    properties = torch.cuda.get_device_properties(0)
    return dict(device=properties.name, torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                compute_capability=[properties.major, properties.minor],
                multiprocessors=properties.multi_processor_count,
                total_device_memory_bytes=properties.total_memory)


def rigid_ate(ground_truth, estimated):
    """Float64 Horn rigid alignment, identical to the prior BA evaluator.

    No fitted scale. Report all frames in their saved frame order.
    """
    truth = np.asarray(ground_truth[:, :3, 3].T, dtype=np.float64)
    prediction = np.asarray(estimated[:, :3, 3].T, dtype=np.float64)
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('Nonfinite trajectory')
    centered_truth = truth - truth.mean(1, keepdims=True)
    centered_prediction = prediction - prediction.mean(1, keepdims=True)
    # Match the repository evaluator's accumulation order, including float64.
    cross = np.zeros((3, 3))
    for column in range(prediction.shape[1]):
        cross += np.outer(centered_prediction[:, column], centered_truth[:, column])
    u, _, vh = np.linalg.svd(cross.T)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        correction[2, 2] = -1
    rotation = u @ correction @ vh
    translation = truth.mean(1, keepdims=True) - rotation @ prediction.mean(1, keepdims=True)
    errors = rotation @ prediction + translation - truth
    return float(np.sqrt(np.mean(np.sum(errors * errors, axis=0))))
