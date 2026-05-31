"""Country boundary mask generation.

Supports two boundary sources:
- geoBoundaries CGAZ ADM0 GeoJSON (legacy)
- Natural Earth 10m with disputed areas merged by de-facto administrator
"""

import json
import os
from pathlib import Path as _Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
from matplotlib.path import Path
from shapely import vectorized
from shapely.geometry import box
from shapely.ops import unary_union

# ---------------------------------------------------------------------------
# Country name → geoBoundaries "shapeName" field mapping (legacy)
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

# ---------------------------------------------------------------------------
# Natural Earth ADM0_A3 → lowercase country key
# ---------------------------------------------------------------------------
_ADM0_A3_TO_COUNTRY: Dict[str, str] = {
    "AFG": "afghanistan",
    "CHN": "china",
    "IND": "india",
    "KAZ": "kazakhstan",
    "KGZ": "kyrgyzstan",
    "NPL": "nepal",
    "PAK": "pakistan",
    "TJK": "tajikistan",
    "UZB": "uzbekistan",
    "KAS": "india",  # Siachen Glacier – de-facto Indian administration
}

_NE_COUNTRIES_SHP = (
    "data/country_masks/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp"
)
_NE_DISPUTED_SHP = (
    "data/country_masks/ne_10m_admin_0_disputed_areas/ne_10m_admin_0_disputed_areas.shp"
)


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


def generate_geo_country_boundary_masks(
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


# ---------------------------------------------------------------------------
# Natural Earth boundaries with disputed areas
# ---------------------------------------------------------------------------


def load_natural_earth_boundaries(
    countries_shp: str = _NE_COUNTRIES_SHP,
    disputed_shp: str = _NE_DISPUTED_SHP,
    adm0_a3_map: Optional[Dict[str, str]] = None,
) -> Dict[str, gpd.GeoDataFrame]:
    """Load Natural Earth countries and merge disputed areas by de-facto admin.

    Disputed areas are assigned to the country that administers them
    according to the ``ADM0_A3`` field.  The mapping from ADM0_A3 codes
    to lowercase country keys is given by *adm0_a3_map* (defaults to
    ``_ADM0_A3_TO_COUNTRY``).

    Parameters
    ----------
    countries_shp : path to ne_10m_admin_0_countries shapefile.
    disputed_shp : path to ne_10m_admin_0_disputed_areas shapefile.
    adm0_a3_map : {ADM0_A3_code: country_key}. Defaults to HMA set.

    Returns
    -------
    Dict mapping lowercase country key → GeoDataFrame with merged geometry.
    """
    if adm0_a3_map is None:
        adm0_a3_map = _ADM0_A3_TO_COUNTRY

    gdf_countries = gpd.read_file(countries_shp)
    gdf_disputed = gpd.read_file(disputed_shp)

    # Build {country_key: [geometry, ...]} collecting base + disputed polygons
    geom_parts: Dict[str, list] = {}
    for country_key in adm0_a3_map.values():
        geom_parts.setdefault(country_key, [])

    # Add base country geometries (matched by NAME)
    name_to_key = {v: k for k, v in COUNTRY_NAME_MAP.items()}
    for _, row in gdf_countries.iterrows():
        key = name_to_key.get(row["NAME"])
        if key is not None:
            geom_parts.setdefault(key, []).append(row.geometry)

    # Add disputed areas by ADM0_A3
    for _, row in gdf_disputed.iterrows():
        a3 = row["ADM0_A3"]
        key = adm0_a3_map.get(a3)
        if key is not None:
            geom_parts[key].append(row.geometry)

    # Merge into unified geometries
    result: Dict[str, gpd.GeoDataFrame] = {}
    for key, geoms in geom_parts.items():
        if geoms:
            merged = unary_union(geoms)
            result[key] = gpd.GeoDataFrame(
                {"country": [key]}, geometry=[merged], crs=gdf_countries.crs
            )

    return result


def _rasterize_geometry(
    geometry,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
) -> np.ndarray:
    """Rasterize a shapely geometry onto a regular lat/lon grid.

    Uses ``shapely.vectorized.contains`` for fast point-in-polygon.

    Returns
    -------
    2-D bool array of shape (len(lat_vals), len(lon_vals)).
    """
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    return vectorized.contains(geometry, lon_grid, lat_grid)


def generate_country_boundary_masks(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    countries: List[str],
    out_dir: str,
    land_mask: Optional[np.ndarray] = None,
    countries_shp: str = _NE_COUNTRIES_SHP,
    disputed_shp: str = _NE_DISPUTED_SHP,
    adm0_a3_map: Optional[Dict[str, str]] = None,
    force: bool = False,
) -> Dict[str, str]:
    """Rasterise Natural Earth country polygons (with disputed areas) onto a grid.

    Disputed areas are merged into the country that administers them
    de-facto, based on the ``ADM0_A3`` field in the NE disputed-areas
    shapefile.

    Parameters
    ----------
    lat_vals, lon_vals : 1-D arrays of grid coordinates.
    countries : list of lowercase country keys to generate masks for.
    out_dir : directory where ``<country>_boundary_mask.npy`` is saved.
    land_mask : optional 2-D array; boundary masks are intersected with it.
    countries_shp : path to ne_10m_admin_0_countries shapefile.
    disputed_shp : path to ne_10m_admin_0_disputed_areas shapefile.
    adm0_a3_map : {ADM0_A3_code: country_key}. Defaults to HMA set.
    force : regenerate masks even if cached files exist.

    Returns
    -------
    {country_key: path_to_boundary_mask.npy}
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load and merge boundaries
    merged = load_natural_earth_boundaries(
        countries_shp=countries_shp,
        disputed_shp=disputed_shp,
        adm0_a3_map=adm0_a3_map,
    )

    mask_paths: Dict[str, str] = {}
    for country in countries:
        path_file = os.path.join(out_dir, f"{country}_boundary_mask.npy")

        if not force and os.path.exists(path_file):
            print(
                f"    {country:15s}: cached ({int(np.load(path_file).sum()):5d} pixels)"
            )
            mask_paths[country] = path_file
            continue

        if country not in merged:
            raise ValueError(
                f"Country '{country}' not found in Natural Earth data. "
                f"Available: {sorted(merged.keys())}"
            )

        geom = merged[country].geometry.iloc[0]
        mask = _rasterize_geometry(geom, lat_vals, lon_vals).astype(int)

        if land_mask is not None:
            mask = mask & land_mask.astype(int)

        np.save(path_file, mask)
        print(f"    {country:15s}: {int(mask.sum()):5d} pixels")
        mask_paths[country] = path_file

    return mask_paths
