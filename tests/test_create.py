import numpy as np
import pytest
import xarray as xr

from fl4hma.data.create import create_data_array

# ── helpers ──────────────────────────────────────────────────────────────────


def _make_aphro_nc(path, variable="tave", n_time=5, lat=(18, 42), lon=(55, 110)):
    """Write a minimal APHRODITE-like NetCDF file."""
    rng = np.random.default_rng(0)
    lats = np.arange(lat[0], lat[1], dtype=np.float64)
    lons = np.arange(lon[0], lon[1], dtype=np.float64)
    times = xr.cftime_range("2000-01-01", periods=n_time, freq="D")
    shape = (n_time, len(lats), len(lons))

    ds = xr.Dataset(
        {
            variable: (
                ("time", "lat", "lon"),
                rng.standard_normal(shape).astype(np.float32),
            ),
            "rstn": (
                ("time", "lat", "lon"),
                rng.integers(0, 10, shape).astype(np.float32),
            ),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )
    ds.to_netcdf(path)
    return ds


def _make_elevation_nc(path, lat, lon):
    """Write a minimal elevation NetCDF with a single time step matching the given grid."""
    rng = np.random.default_rng(1)
    elev = rng.standard_normal((1, len(lat), len(lon))).astype(np.float32)
    ds = xr.Dataset(
        {"data": (("time", "lat", "lon"), elev)},
        coords={"time": [0], "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def aphro_nc(tmp_path):
    """A standard APHRODITE-like NetCDF (tave, 5 timesteps)."""
    path = tmp_path / "aphro.nc"
    _make_aphro_nc(path, variable="tave")
    return str(path)


@pytest.fixture
def aphro_nc_precip(tmp_path):
    """APHRODITE-like NetCDF with precip variable."""
    path = tmp_path / "aphro_precip.nc"
    _make_aphro_nc(path, variable="precip")
    return str(path)


@pytest.fixture
def aphro_nc_365(tmp_path):
    """APHRODITE-like NetCDF with 365 timesteps."""
    path = tmp_path / "aphro_365.nc"
    _make_aphro_nc(path, variable="tave", n_time=365)
    return str(path)


@pytest.fixture
def tave_da(aphro_nc):
    """DataArray from create_data_array with seasonality=True."""
    return create_data_array("tave", aphro_nc, seasonality=True)


# ── basic output tests ───────────────────────────────────────────────────────


class TestCreateDataArrayBasic:
    def test_returns_dataarray(self, tave_da):
        assert isinstance(tave_da, xr.DataArray)

    def test_dimensions(self, tave_da):
        assert set(tave_da.dims) == {"variable", "time", "lat", "lon"}

    def test_lat_lon_slicing(self, tave_da):
        """Output lat/lon should be clipped to [20,40) x [60,105)."""
        assert float(tave_da.lat.min()) >= 20
        assert float(tave_da.lat.max()) <= 40
        assert float(tave_da.lon.min()) >= 60
        assert float(tave_da.lon.max()) <= 105

    def test_no_nan_values(self, tave_da):
        assert not np.isnan(tave_da.values).any()


# ── variable / channel tests ─────────────────────────────────────────────────


class TestVariables:
    def test_seasonality_variables_present(self, aphro_nc_precip):
        da = create_data_array("precip", aphro_nc_precip, seasonality=True)
        var_names = list(da.coords["variable"].values)
        for v in ["precip", "lats", "lons", "year", "dayofyear_cos", "dayofyear_sin"]:
            assert v in var_names

    def test_channel_count_with_seasonality(self, tave_da):
        # tave + lats + lons + year + cos + sin = 6
        assert tave_da.sizes["variable"] == 6

    def test_rstn_dropped(self, tave_da):
        assert "rstn" not in list(tave_da.coords["variable"].values)


# ── seasonality value tests ──────────────────────────────────────────────────


class TestSeasonality:
    def test_dayofyear_cos_sin_range(self, aphro_nc_365):
        da = create_data_array("tave", aphro_nc_365, seasonality=True)
        cos_vals = da.sel(variable="dayofyear_cos").values
        sin_vals = da.sel(variable="dayofyear_sin").values
        assert cos_vals.min() >= -1.0 and cos_vals.max() <= 1.0
        assert sin_vals.min() >= -1.0 and sin_vals.max() <= 1.0

    def test_cos_sin_unit_circle(self, tave_da):
        """cos^2 + sin^2 ≈ 1 for every pixel."""
        cos_vals = tave_da.sel(variable="dayofyear_cos").values
        sin_vals = tave_da.sel(variable="dayofyear_sin").values
        np.testing.assert_allclose(cos_vals**2 + sin_vals**2, 1.0, atol=1e-6)

    def test_lats_broadcast_correctly(self, tave_da):
        lats_var = tave_da.sel(variable="lats")
        # Each row (lat index) should be constant across lon and time
        for lat_idx in range(lats_var.sizes["lat"]):
            row_vals = lats_var.isel(lat=lat_idx).values
            assert np.all(row_vals == row_vals.flat[0])

    def test_lons_broadcast_correctly(self, tave_da):
        lons_var = tave_da.sel(variable="lons")
        # Each column (lon index) should be constant across lat and time
        for lon_idx in range(lons_var.sizes["lon"]):
            col_vals = lons_var.isel(lon=lon_idx).values
            assert np.all(col_vals == col_vals.flat[0])


# ── seasonality=False ────────────────────────────────────────────────────────


class TestNoSeasonality:
    def test_seasonality_false_raises(self, aphro_nc):
        """seasonality=False doesn't create year/dayofyear_* vars, but the
        function unconditionally selects them ‒ expect a KeyError."""
        with pytest.raises(KeyError):
            create_data_array("tave", aphro_nc, seasonality=False)


# ── elevation tests ──────────────────────────────────────────────────────────


class TestElevation:
    def test_adds_elevation_channel(self, aphro_nc, tave_da, tmp_path):
        elev_nc = tmp_path / "elev.nc"
        _make_elevation_nc(elev_nc, lat=tave_da.lat.values, lon=tave_da.lon.values)

        da = create_data_array(
            "tave", aphro_nc, elevation_path=str(elev_nc), seasonality=True
        )
        assert da.sizes["variable"] == tave_da.sizes["variable"] + 1

    def test_elevation_no_nan(self, aphro_nc, tave_da, tmp_path):
        elev_nc = tmp_path / "elev.nc"
        _make_elevation_nc(elev_nc, lat=tave_da.lat.values, lon=tave_da.lon.values)

        da = create_data_array(
            "tave", aphro_nc, elevation_path=str(elev_nc), seasonality=True
        )
        assert not np.isnan(da.values).any()
