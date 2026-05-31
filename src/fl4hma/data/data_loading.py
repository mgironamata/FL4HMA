"""APHRODITE data loading """

from typing import Optional, Tuple

import numpy as np
import xarray as xr


def load_aphro_data(
    train_path: str,
    test_path: str,
    variable: str = "precip",
    lon_slice: Tuple[float, float] = (60, 105),
    lat_slice: Tuple[float, float] = (20, 40),
    seasonality: bool = False,
    elevation_path: str = None,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Load APHRODITE train/test data via :func:`~fl4hma.data.create.create_data_array`.

    Parameters
    ----------
    train_path, test_path : str
        Paths to training and test NetCDF files.
    variable : str
        NetCDF variable name, e.g. ``"precip"`` or ``"tave"``.
    lon_slice, lat_slice : tuple of float
        Spatial bounds for subsetting.
    seasonality : bool
        Include year/dayofyear features.
    elevation_path : str or None
        Optional elevation NetCDF to append as a channel.
    """
    da_train = create_data_array(
        variable=variable,
        aphro_path=train_path,
        elevation_path=elevation_path,
        seasonality=seasonality,
        lon_slice=lon_slice,
        lat_slice=lat_slice,
    )
    da_test = create_data_array(
        variable=variable,
        aphro_path=test_path,
        elevation_path=elevation_path,
        seasonality=seasonality,
        lon_slice=lon_slice,
        lat_slice=lat_slice,
    )
    return da_train, da_test


def create_data_array(
    variable: str,
    aphro_path: str,
    elevation_path: Optional[str] = None,
    seasonality: bool = False,
    lon_slice: Tuple[float, float] = (60, 105),
    lat_slice: Tuple[float, float] = (20, 40),
) -> xr.DataArray:
    """
    Load an APHRODITE NetCDF file and return a DataArray with coordinate channels.

    Always includes lat/lon channels. Optionally adds seasonality features
    (year, dayofyear_cos, dayofyear_sin) and/or elevation.

    Args:
        variable: NetCDF variable name (e.g. "tave" or "precip").
        aphro_path: Path to the APHRODITE dataset file.
        elevation_path: Path to the elevation dataset file (optional).
        seasonality: Whether to include seasonality features.
        lon_slice: (min_lon, max_lon) for spatial subsetting.
        lat_slice: (min_lat, max_lat) for spatial subsetting.

    Returns:
        xarray.DataArray with dims (variable, time, lat, lon).
    """

    ds = xr.open_dataset(aphro_path).sel(
        lon=slice(*lon_slice),
        lat=slice(*lat_slice),
    )

    ds["lats"] = (
        ("time", "lat", "lon"),
        np.tile(ds.lat.values, (ds.time.size, ds.lon.size, 1)).transpose(0, 2, 1),
    )
    ds["lons"] = (
        ("time", "lat", "lon"),
        np.tile(ds.lon.values, (ds.time.size, ds.lat.size, 1)),
    )

    channels = [variable, "lats", "lons"]

    if seasonality:
        ds["year"] = (
            ("time", "lat", "lon"),
            np.tile(
                ds.time.dt.year.values[:, None, None],
                (1, ds.sizes["lat"], ds.sizes["lon"]),
            ),
        )
        ds["dayofyear_cos"] = (
            ("time", "lat", "lon"),
            np.tile(
                np.cos(2 * np.pi * ds.time.dt.dayofyear.values / 365.0)[:, None, None],
                (1, ds.sizes["lat"], ds.sizes["lon"]),
            ),
        )
        ds["dayofyear_sin"] = (
            ("time", "lat", "lon"),
            np.tile(
                np.sin(2 * np.pi * ds.time.dt.dayofyear.values / 365.0)[:, None, None],
                (1, ds.sizes["lat"], ds.sizes["lon"]),
            ),
        )
        channels += ["year", "dayofyear_cos", "dayofyear_sin"]

    # Create DataArray with dimensions (variable, time, lat, lon)
    da = ds[channels].to_array().fillna(0)

    if elevation_path is not None:
        elev_ds = xr.open_dataset(elevation_path)
        elev_da = elev_ds.data.transpose("time", "lat", "lon")
        elev_arr = np.tile(elev_da.values, (da.time.size, 1, 1))
        new_elev_da = xr.DataArray(
            elev_arr,
            coords={"time": da.time, "lat": da.lat, "lon": da.lon},
            dims=elev_da.dims,
        )
        new_elev_da = new_elev_da.expand_dims(dim={"variable": ["elevation"]})
        da = xr.concat([da, new_elev_da], dim="variable")

    return da
