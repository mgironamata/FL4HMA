"""Mask generation utilities for station and boundary masks."""

import json
import os
from typing import Dict, List, Optional

import numpy as np
from matplotlib.path import Path


def generate_random_split_masks(
    centralised_mask: np.ndarray,
    n_clients: int,
    out_dir: str,
    seed: int = 42,
    prefix: str = "random",
) -> Dict[str, str]:
    """Randomly partition station pixels into *n_clients* disjoint masks.

    Parameters
    ----------
    centralised_mask : 2-D bool/int array
    n_clients : number of disjoint groups
    out_dir : output directory
    seed : random seed for reproducibility
    prefix : filename prefix

    Returns
    -------
    {client_name: path_to_mask.npy}  e.g. {"client_0": "...", ...}
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    positions = np.argwhere(centralised_mask.astype(bool))
    n_stations = len(positions)
    perm = rng.permutation(n_stations)
    splits = np.array_split(perm, n_clients)

    mask_paths: Dict[str, str] = {}
    for i, indices in enumerate(splits):
        m = np.zeros_like(centralised_mask, dtype=int)
        rows, cols = positions[indices, 0], positions[indices, 1]
        m[rows, cols] = 1
        name = f"client_{i}"
        path = os.path.join(out_dir, f"{prefix}_{name}_mask.npy")
        np.save(path, m)
        mask_paths[name] = path

    # Verify
    union = np.zeros_like(centralised_mask, dtype=int)
    for p in mask_paths.values():
        union += np.load(p)
    assert np.array_equal(
        union, centralised_mask.astype(int)
    ), "Random split union does not equal centralised mask"

    return mask_paths
