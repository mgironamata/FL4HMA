"""Station mask generation from GHCN and LTM station networks.

Generates:
- Static 2-D masks (lat × lon): station presence on the APHRODITE grid
- Nonstationary 3-D masks (year × lat × lon): time-varying station coverage
- Land/sea (output) masks from APHRODITE data
- Random client splits from any centralised mask
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# GHCN station ID prefix → country
STATION_PREFIX_MAP: Dict[str, str] = {
    "IN": "india",
    "CH": "china",
    "NP": "nepal",
    "AF": "afghanistan",
    "PK": "pakistan",
    "TI": "tajikistan",
    "KG": "kyrgyzstan",
    "UZ": "uzbekistan",
    "KZ": "kazakhstan",
}


def generate_output_mask(
    ds_path: str,
    variable: str,
    lon_slice: Tuple[float, float] = (60, 105),
    lat_slice: Tuple[float, float] = (20, 40),
    out_path: Optional[str] = None,
) -> np.ndarray:
    """Derive a land/sea mask from the first timestep of an APHRODITE file.

    Args:
        ds_path: Path to NetCDF file.
        variable: Variable name (e.g. "precip", "tave").
        lon_slice: (min_lon, max_lon) for spatial subsetting.
        lat_slice: (min_lat, max_lat) for spatial subsetting.
        out_path: If given, save the mask to this path.

    Returns:
        2-D int array (lat, lon) with 1 = land, 0 = ocean/missing.
    """
    ds = xr.open_dataset(ds_path).sel(lon=slice(*lon_slice), lat=slice(*lat_slice))
    mask = np.where(np.isnan(ds[variable][0].values), 0, 1).astype(np.int8)
    ds.close()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, mask)
    return mask


def _load_ghcn_stations(stations_path: str) -> pd.DataFrame:
    """Load GHCN-Daily station metadata (fixed-width format)."""
    col_widths = [12, 9, 10, 8, 30, 8, 8]
    col_names = ["id", "lat", "lon", "elevation", "name", "gsn_flag", "wmo_id"]
    df = pd.read_fwf(stations_path, widths=col_widths, header=None, names=col_names)
    df["country"] = df["id"].str[:2].map(STATION_PREFIX_MAP)
    return df


def _load_ghcn_inventory(inventory_path: str) -> pd.DataFrame:
    """Load GHCN-Daily inventory (station temporal coverage)."""
    col_widths = [12, 9, 10, 5, 5, 5]
    col_names = ["id", "lat", "lon", "var", "start", "end"]
    return pd.read_fwf(inventory_path, header=None, widths=col_widths, names=col_names)


def _load_ltm_stations(ltm_path: str) -> pd.DataFrame:
    """Load LTM HMA research station locations.

    Reads the LTM_HMA.csv format with columns including Name, LAT, LON,
    TimeStart, TimeEnd, and OpenData. Stations marked as "Restricted" are
    excluded. Stations missing temporal coverage are kept but with NaN
    start/end.

    Returns:
        DataFrame with columns: name, lat, lon, start, end, country.
    """
    df = pd.read_csv(ltm_path, index_col=False)
    # Filter out restricted stations
    df = df[df["OpenData"].str.strip().str.lower() != "restricted"].copy()
    # Coerce temporal columns to numeric (handles '?' and blanks → NaN)
    df["start"] = pd.to_numeric(df["TimeStart"], errors="coerce")
    df["end"] = pd.to_numeric(df["TimeEnd"], errors="coerce")
    df = df.rename(columns={"Name": "name", "LAT": "lat", "LON": "lon"})
    df = df.dropna(subset=["lat", "lon"])
    # Drop stations without temporal coverage
    df = df.dropna(subset=["start", "end"])
    df["country"] = None
    return df[["name", "lat", "lon", "start", "end", "country"]]


def generate_static_station_masks(
    ghcn_stations_path: str,
    output_mask: np.ndarray,
    lon_slice: Tuple[float, float] = (60, 105),
    lat_slice: Tuple[float, float] = (20, 40),
    res: float = 0.25,
    ltm_path: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Generate static 2-D station masks from GHCN (+ optional LTM) stations.

    Creates a centralised mask (all stations) and per-country masks.
    Masks are intersected with the output (land) mask.

    Args:
        ghcn_stations_path: Path to GHCN-D stations file.
        output_mask: 2-D land mask to intersect with.
        lon_slice: Longitude extent.
        lat_slice: Latitude extent.
        res: Grid resolution in degrees.
        ltm_path: Optional path to LTM_HMA.csv research stations.
        out_dir: If given, save masks as .npy files here.

    Returns:
        Dict mapping mask name → 2-D int array.
        Keys: "centralised", and each country name.
    """
    lon_bins = np.arange(lon_slice[0], lon_slice[1] + res, res)
    lat_bins = np.arange(lat_slice[0], lat_slice[1] + res, res)

    ghcn_df = _load_ghcn_stations(ghcn_stations_path)

    # Combine with LTM research stations if provided
    if ltm_path is not None:
        ltm_df = _load_ltm_stations(ltm_path)
        combined = pd.concat(
            [ghcn_df[["lat", "lon", "country"]], ltm_df[["lat", "lon", "country"]]],
            ignore_index=True,
        )
    else:
        combined = ghcn_df[["lat", "lon", "country"]].copy()

    # Centralised mask (all stations)
    density, _, _ = np.histogram2d(
        combined["lon"], combined["lat"], bins=[lon_bins, lat_bins]
    )
    centralised = np.where(density.T > 0, 1, 0) & output_mask.astype(int)

    masks: Dict[str, np.ndarray] = {"centralised": centralised}

    # Per-country masks
    for country in combined["country"].dropna().unique():
        country_stations = combined[combined["country"] == country]
        density_c, _, _ = np.histogram2d(
            country_stations["lon"],
            country_stations["lat"],
            bins=[lon_bins, lat_bins],
        )
        country_mask = np.where(density_c.T > 0, 1, 0) & output_mask.astype(int)
        masks[country] = country_mask

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        for name, m in masks.items():
            np.save(os.path.join(out_dir, f"{name}_mask.npy"), m)

    return masks


def generate_nonstationary_station_masks(
    ghcn_stations_path: str,
    ghcn_inventory_path: str,
    output_mask: np.ndarray,
    variable: str = "PRCP",
    lon_slice: Tuple[float, float] = (60, 105),
    lat_slice: Tuple[float, float] = (20, 40),
    year_range: Tuple[int, int] = (1998, 2015),
    res: float = 0.25,
    ltm_path: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Generate nonstationary 3-D masks (year × lat × lon) from GHCN inventory.

    Uses the station temporal coverage (start/end years) to produce
    year-varying masks showing which grid cells had active stations each year.
    LTM research stations (if provided) are added to every year since they
    lack temporal metadata.

    Args:
        ghcn_stations_path: Path to GHCN-D stations file.
        ghcn_inventory_path: Path to GHCN-D inventory file.
        output_mask: 2-D land mask to intersect with.
        variable: GHCN variable code ("PRCP", "TAVG", "TMAX", "TMIN").
        lon_slice: Longitude extent.
        lat_slice: Latitude extent.
        year_range: (start_year, end_year) inclusive.
        res: Grid resolution in degrees.
        ltm_path: Optional path to LTM_HKH.csv research stations.
            These are assumed present in all years.
        out_dir: If given, save masks here.

    Returns:
        Dict mapping mask name → 3-D int array (year × lat × lon).
        Keys: "centralised", and each country name.
    """
    lon_bins = np.arange(lon_slice[0], lon_slice[1] + res, res)
    lat_bins = np.arange(lat_slice[0], lat_slice[1] + res, res)
    year_bins = np.arange(year_range[0], year_range[1] + 2, 1)  # +2 for inclusive end

    stations = _load_ghcn_stations(ghcn_stations_path)
    inventory = _load_ghcn_inventory(ghcn_inventory_path)

    merged = pd.merge(stations[["id", "country"]], inventory, on="id", how="inner")
    var_df = merged[merged["var"] == variable].copy()

    # Expand rows: one row per (station, year) where station was active
    var_df = var_df.loc[var_df.index.repeat(var_df["end"] - var_df["start"] + 1)].copy()
    var_df["year"] = (
        var_df.groupby(level=0)
        .apply(lambda x: np.arange(x["start"].iloc[0], x["end"].iloc[0] + 1))
        .explode()
        .values.astype(int)
    )
    var_df = var_df[
        (var_df["year"] >= year_range[0]) & (var_df["year"] <= year_range[1])
    ]

    # Add LTM research stations using their actual temporal coverage
    if ltm_path is not None:
        ltm_df = _load_ltm_stations(ltm_path)
        ltm_rows = []
        for _, row in ltm_df.iterrows():
            start = int(row["start"])
            end = int(row["end"])
            for y in range(max(start, year_range[0]), min(end, year_range[1]) + 1):
                ltm_rows.append(
                    {
                        "year": y,
                        "lat": row["lat"],
                        "lon": row["lon"],
                        "country": row["country"],
                    }
                )
        if ltm_rows:
            ltm_expanded = pd.DataFrame(ltm_rows)
            var_df = pd.concat(
                [var_df[["year", "lat", "lon", "country"]], ltm_expanded],
                ignore_index=True,
            )

    # Broadcast output_mask to 3D (n_years = len(year_bins) - 1 histogram bins)
    n_years = len(year_bins) - 1
    output_mask_3d = np.broadcast_to(
        output_mask[np.newaxis, :, :],
        (n_years, output_mask.shape[0], output_mask.shape[1]),
    )

    def _make_3d_mask(df: pd.DataFrame) -> np.ndarray:
        arr = df[["year", "lat", "lon"]].values
        density, _ = np.histogramdd(arr, bins=[year_bins, lat_bins, lon_bins])
        mask = np.where(density >= 1, 1, 0)
        return mask & output_mask_3d

    # Centralised
    masks: Dict[str, np.ndarray] = {"centralised": _make_3d_mask(var_df)}

    # Per-country
    for country in var_df["country"].dropna().unique():
        masks[country] = _make_3d_mask(var_df[var_df["country"] == country])

    if out_dir is not None:
        var_label = variable.lower()
        if var_label == "tavg":
            var_label = "tave"
        os.makedirs(out_dir, exist_ok=True)
        for name, m in masks.items():
            np.save(os.path.join(out_dir, f"{var_label}_{name}_mask.npy"), m)

    return masks


def generate_random_split_masks(
    centralised_mask: np.ndarray,
    n_clients: int,
    out_dir: str,
    seed: int = 42,
    prefix: str = "random",
) -> Dict[str, str]:
    """Randomly partition station pixels into *n_clients* disjoint masks.

    Args:
        centralised_mask: 2-D bool/int array.
        n_clients: Number of disjoint groups.
        out_dir: Output directory.
        seed: Random seed for reproducibility.
        prefix: Filename prefix.

    Returns:
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

    # Verify disjoint union
    union = np.zeros_like(centralised_mask, dtype=int)
    for p in mask_paths.values():
        union += np.load(p)
    assert np.array_equal(
        union, centralised_mask.astype(int)
    ), "Random split union does not equal centralised mask"

    return mask_paths
