""" TO DO: switch to Natural Earth boundaries and disputed areas

Boundary mask generation"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
from matplotlib.path import Path

# ---------------------------------------------------------------------------
# Country name → geoBoundaries "shapeName" field mapping
# ---------------------------------------------------------------------------
COUNTRY_NAME_MAP = {
    "afghanistan": "Afghanistan",
    "china": "China",
    "india": "India",
    "kazakhstan": "Kazakhstan",
    "kyrgyzstan": "Kyrgyzstan",
    "nepal": "Nepal",
    "pakistan": "Pakistan",
    "tajikistan": "Tajikistan",
    "uzbekistan": "Uzbekistan",
}

_GEOBOUNDARIES_GEOJSON = "data/country_masks/geoBoundariesCGAZ_ADM0.geojson"


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


def _polygon_coords_to_paths(geometry: dict) -> List[Path]:
    """Convert a GeoJSON Polygon/MultiPolygon to matplotlib Paths."""
    paths = []
    if geometry["type"] == "Polygon":
        paths.append(Path(geometry["coordinates"][0]))
    elif geometry["type"] == "MultiPolygon":
        for poly in geometry["coordinates"]:
            paths.append(Path(poly[0]))
    return paths


def generate_country_boundary_masks(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    countries: List[str],
    out_dir: str,
    land_mask: Optional[np.ndarray] = None,
    geojson_path: str = _GEOBOUNDARIES_GEOJSON,
    name_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Rasterise country polygons onto a lat/lon grid.

    Uses geoBoundaries CGAZ ADM0 boundaries (local GeoJSON).
    Point-in-polygon is done via ``matplotlib.path.Path``.

    Parameters
    ----------
    lat_vals, lon_vals : 1-D arrays of grid coordinates.
    countries : list of lowercase country names (keys in *name_map*).
    out_dir : directory where ``<country>_boundary_mask.npy`` is saved.
    land_mask : optional 2-D array; boundary masks are intersected with it.
    geojson_path : path to the geoBoundaries GeoJSON file.
                   Defaults to ``data/country_masks/geoBoundariesCGAZ_ADM0.geojson``.
    name_map : {lowercase_name: shapeName}. Defaults to HMA set.

    Returns
    -------
    {country_name: path_to_boundary_mask.npy}
    """
    if name_map is None:
        name_map = COUNTRY_NAME_MAP

    os.makedirs(out_dir, exist_ok=True)

    # Load local geoBoundaries GeoJSON
    with open(geojson_path) as f:
        geojson = json.load(f)

    # Build lookup: shapeName → geometry
    ne_geom = {}
    for feat in geojson["features"]:
        ne_geom[feat["properties"]["shapeName"]] = feat["geometry"]

    # Build grid points (lon, lat) — shape (N, 2) for Path.contains_points
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    mask_paths: Dict[str, str] = {}
    for country in countries:
        path_file = os.path.join(out_dir, f"{country}_boundary_mask.npy")
        if os.path.exists(path_file):
            print(
                f"    {country:15s}: cached ({int(np.load(path_file).sum()):5d} pixels)"
            )
            mask_paths[country] = path_file
            continue

        ne_name = name_map[country]
        if ne_name not in ne_geom:
            raise ValueError(
                f"Country '{ne_name}' not found in geoBoundaries data. "
                f"Available: {sorted(ne_geom.keys())}"
            )

        paths = _polygon_coords_to_paths(ne_geom[ne_name])
        inside = np.zeros(grid_points.shape[0], dtype=bool)
        for p in paths:
            inside |= p.contains_points(grid_points)
        mask = inside.reshape(lon_grid.shape).astype(int)

        if land_mask is not None:
            mask = mask & land_mask.astype(int)

        np.save(path_file, mask)
        print(f"    {country:15s}: {int(mask.sum()):5d} pixels")
        mask_paths[country] = path_file

    return mask_paths
