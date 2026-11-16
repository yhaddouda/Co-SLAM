"""Private spatial/depth ordering for complementary tracking-ray groups.

Every group is omitted once per cycle; sorted original indices preserve the
native order of the retained rays. Selection does not alter global RNG state.
"""
import numpy as np


def _image_depth_order(population, group_ids, metadata):
    """Order rays by group, coarse image cell, depth, and original row index."""
    original = np.arange(population, dtype=np.int64)
    if metadata is None:
        return np.lexsort((original, group_ids))
    rays = np.asarray(metadata)
    if rays.ndim != 2 or rays.shape[0] != population or rays.shape[1] not in (4, 7):
        raise ValueError('metadata must have shape (population, 4) or (population, 7)')
    directions = rays[:, :3].astype(np.float64, copy=False)
    valid = np.isfinite(directions).all(axis=1) & (np.abs(directions[:, 2]) > 1e-12)
    image = np.zeros((population, 2), dtype=np.float64)
    np.divide(directions[:, :2], np.abs(directions[:, 2:3]),
              out=image, where=valid[:, None])
    # Frame intrinsics are common to each dataset. Bounding the observed image
    # coordinates gives deterministic, inexpensive spatial strata for any batch.
    if np.any(valid):
        lower = image[valid].min(axis=0)
        width = image[valid].max(axis=0) - lower
        width = np.where(width > 0., width, 1.)
        cells = np.floor((image - lower) / width * 8.)
        cells = np.clip(cells, 0, 7).astype(np.int64)
        cell = cells[:, 1] * 8 + cells[:, 0]
        cell[~valid] = 64
    else:
        cell = np.full(population, 64, dtype=np.int64)
    depth = rays[:, -1].astype(np.float64, copy=False)
    depth_key = np.where(np.isfinite(depth) & (depth > 0.), depth, np.inf)
    return np.lexsort((original, depth_key, cell, group_ids))


def cyclic_ray_subsets(population, seed, metadata=None, groups=4):
    """Spatial/depth interleaved complementary subsets for one tracking frame.

    Each ray is omitted exactly once per cycle. Group sizes differ by at most
    one, so budgets can differ by one when N is not divisible by the group
    count. A private random initial phase changes which complement comes first.
    Singletons remain dense; for other small populations use at most N groups.
    """
    if isinstance(population, (bool, np.bool_)) or int(population) != population or population < 0:
        raise ValueError('population must be a nonnegative integer')
    if isinstance(groups, (bool, np.bool_)) or int(groups) != groups or groups < 2:
        raise ValueError('cycle groups must be an integer >= 2')
    population, groups = int(population), int(groups)
    order = _image_depth_order(population, np.zeros(population, dtype=np.int64), metadata)
    if population < 2:
        return (order,)
    groups = min(groups, population)
    phase = int(np.random.default_rng(int(seed)).integers(groups))
    rank_group = np.arange(population) % groups
    return tuple(np.sort(order[rank_group != (phase + j) % groups]).astype(np.int64)
                 for j in range(groups))

