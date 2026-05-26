import numpy as np
import xarray as xr


def create_data_array(
    variable: str,
    aphro_path: str,
    elevation_path: str = None,
    seasonality: bool = False,
):
    """
    Load data from the specified paths. Assumes the data shares the same regular grids.

    Args:
        variable (str): The name of the variable to load from the dataset ("tave" or "precip").
        aphro_path (str): The path to the APHROSITE dataset file.
        elevation_path (str, optional): The path to the elevation dataset file. Defaults to None.
        seasonality (bool, optional): Whether to include seasonality features (year, dayofyear_cos, dayofyear_sin). Defaults to False.

    Returns:
        xarray.DataArray: The created DataArray with dimensions (variable, time, lat, lon).
    """

    ds = xr.open_dataset(aphro_path).sel(lon=slice(60, 105), lat=slice(20, 40))

    ds["lats"] = (
        (
            "time",
            "lat",
            "lon",
        ),
        np.tile(ds.lat.values, (ds.time.size, ds.lon.size, 1)).transpose(0, 2, 1),
    )
    ds["lons"] = (
        (
            "time",
            "lat",
            "lon",
        ),
        np.tile(ds.lon.values, (ds.time.size, ds.lat.size, 1)),
    )

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

    # drop the rstn variable
    ds = ds.drop_vars("rstn")

    # Create DataArray with dimensions (variable, time, lat, lon)
    da = ds[
        [variable, "lats", "lons", "year", "dayofyear_cos", "dayofyear_sin"]
    ].to_array()

    # Remove NaN values (if any) and replace with zeros
    da = da.fillna(0)

    if elevation_path is not None:
        elev_ds = xr.open_dataset(
            elevation_path
        )  # .sel(lon=slice(60, 105), lat=slice(20, 40))
        elev_da = elev_ds.data.transpose("time", "lat", "lon")
        elev_arr = np.tile(elev_da.values, (da.time.size, 1, 1))
        new_elev_da = xr.DataArray(
            elev_arr,
            coords={"time": da.time, "lat": da.lat, "lon": da.lon},
            dims=elev_da.dims,
        )
        new_elev_da = new_elev_da.expand_dims(dim={"variable": ["time, lat, lon"]})
        da = xr.concat([da, new_elev_da], dim="variable")

    return da
