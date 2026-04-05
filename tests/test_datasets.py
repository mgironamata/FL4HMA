import numpy as np
import pytest
import torch
import xarray as xr

from fl4hma.data.dataset import StationPatchDataset


def make_dataarray(n_vars=3, n_time=4, n_lat=64, n_lon=64):
    """Create a synthetic DataArray with the expected dimensions."""
    rng = np.random.default_rng(42)
    data = rng.standard_normal((n_vars, n_time, n_lat, n_lon)).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("variable", "time", "lat", "lon"),
        coords={
            "variable": [f"v{i}" for i in range(n_vars)],
            "time": np.arange(n_time),
            "lat": np.linspace(25, 40, n_lat),
            "lon": np.linspace(60, 105, n_lon),
        },
    )


# ── constructor tests ─────────────────────────────────────────────────────────


class TestInit:
    def test_wrong_dims_raises(self):
        bad = xr.DataArray(np.zeros((3, 4, 64, 64)), dims=("a", "b", "c", "d"))
        with pytest.raises(AssertionError, match="must have dims"):
            StationPatchDataset(bad, input_sparsity=0.5, output_sparsity=0.5)

    def test_stores_patch_size_and_stride(self):
        ds = StationPatchDataset(
            make_dataarray(),
            patch_size=16,
            stride=8,
            input_sparsity=0.5,
            output_sparsity=0.5,
        )
        assert ds.patch_size == 16
        assert ds.stride == 8

    def test_sparsity_attributes(self):
        ds = StationPatchDataset(
            make_dataarray(),
            input_sparsity=0.3,
            output_sparsity=0.7,
        )
        assert ds.input_sparsity == 0.3
        assert ds.output_sparsity == 0.7

    def test_input_mask_from_file(self, tmp_path):
        mask = np.ones((64, 64), dtype=bool)
        mask_path = tmp_path / "mask.npy"
        np.save(mask_path, mask)
        ds = StationPatchDataset(
            make_dataarray(),
            input_sparsity=str(mask_path),
            output_sparsity=0.5,
        )
        np.testing.assert_array_equal(ds.station_mask, mask)


# ── normalization tests ───────────────────────────────────────────────────────


class TestNormalization:
    def test_normalized_mean_near_zero(self):
        ds = StationPatchDataset(
            make_dataarray(),
            input_sparsity=0.5,
            output_sparsity=0.5,
            normalize=True,
        )
        for c in range(ds.channels):
            assert abs(np.nanmean(ds.data[c])) < 1e-5

    def test_no_normalization(self):
        da = make_dataarray()
        ds = StationPatchDataset(
            da,
            input_sparsity=0.5,
            output_sparsity=0.5,
            normalize=False,
        )
        np.testing.assert_array_equal(ds.data, da.values.astype(np.float32))


# ── indexing & length tests ───────────────────────────────────────────────────


class TestIndexing:
    def test_length_non_overlapping(self):
        # 64x64 grid, patch 32, stride 32 → 2x2 spatial, 4 time steps → 16
        ds = StationPatchDataset(
            make_dataarray(n_time=4, n_lat=64, n_lon=64),
            patch_size=32,
            stride=32,
            input_sparsity=0.5,
            output_sparsity=0.5,
        )
        assert len(ds) == 4 * 2 * 2

    def test_length_overlapping(self):
        # 64x64 grid, patch 32, stride 16 → 3x3 spatial, 2 time steps → 18
        ds = StationPatchDataset(
            make_dataarray(n_time=2, n_lat=64, n_lon=64),
            patch_size=32,
            stride=16,
            input_sparsity=0.5,
            output_sparsity=0.5,
        )
        lat_patches = len(range(0, 64 - 32 + 1, 16))
        lon_patches = len(range(0, 64 - 32 + 1, 16))
        assert len(ds) == 2 * lat_patches * lon_patches

    def test_length_single_patch(self):
        ds = StationPatchDataset(
            make_dataarray(n_time=1, n_lat=32, n_lon=32),
            patch_size=32,
            stride=32,
            input_sparsity=0.5,
            output_sparsity=0.5,
        )
        assert len(ds) == 1


# ── __getitem__ shape & dtype tests ──────────────────────────────────────────


class TestGetItem:
    @pytest.fixture
    def ds(self):
        return StationPatchDataset(
            make_dataarray(n_vars=3, n_time=2, n_lat=64, n_lon=64),
            patch_size=32,
            stride=32,
            input_sparsity=0.5,
            output_sparsity=0.5,
            normalize=True,
        )

    def test_output_count(self, ds):
        result = ds[0]
        assert len(result) == 4

    def test_sparse_input_shape(self, ds):
        sparse_input, _, _, _ = ds[0]
        assert sparse_input.shape == (3, 32, 32)

    def test_sparse_target_shape(self, ds):
        _, sparse_target, _, _ = ds[0]
        assert sparse_target.shape == (1, 32, 32)

    def test_input_mask_shape(self, ds):
        _, _, input_mask, _ = ds[0]
        assert input_mask.shape == (32, 32)

    def test_output_mask_shape(self, ds):
        _, _, _, output_mask = ds[0]
        assert output_mask.shape == (32, 32)

    def test_dtypes(self, ds):
        sparse_input, sparse_target, input_mask, output_mask = ds[0]
        assert sparse_input.dtype == torch.float32
        assert sparse_target.dtype == torch.float32
        assert input_mask.dtype == torch.float32
        assert output_mask.dtype == torch.float32

    def test_valid_index_range(self, ds):
        # first and last index should work without error
        _ = ds[0]
        _ = ds[len(ds) - 1]

    def test_out_of_range_raises(self, ds):
        with pytest.raises(IndexError):
            _ = ds[len(ds)]


# ── masking behaviour ────────────────────────────────────────────────────────


class TestMasking:
    def test_sparse_input_zeros_where_unmasked(self):
        ds = StationPatchDataset(
            make_dataarray(n_vars=3, n_time=1, n_lat=32, n_lon=32),
            patch_size=32,
            stride=32,
            input_sparsity=0.5,
            output_sparsity=1.0,
            normalize=False,
        )
        sparse_input, _, input_mask, _ = ds[0]
        # Where mask is 0 (False), channel-0 of sparse_input must be 0
        assert (sparse_input[0][input_mask == 0] == 0).all()

    def test_sparse_target_minus_one_where_unmasked(self):
        ds = StationPatchDataset(
            make_dataarray(n_vars=3, n_time=1, n_lat=32, n_lon=32),
            patch_size=32,
            stride=32,
            input_sparsity=1.0,
            output_sparsity=0.5,
            normalize=False,
        )
        _, sparse_target, _, output_mask = ds[0]
        # Where output mask is 0, target should be -1
        assert (sparse_target[0][output_mask == 0] == -1.0).all()

    def test_full_input_sparsity_keeps_all(self):
        ds = StationPatchDataset(
            make_dataarray(n_vars=3, n_time=1, n_lat=32, n_lon=32),
            patch_size=32,
            stride=32,
            input_sparsity=1.0,
            output_sparsity=1.0,
            normalize=False,
        )
        _, _, input_mask, output_mask = ds[0]
        # sparsity=1.0 → all pixels kept
        assert input_mask.sum() == 32 * 32
        assert output_mask.sum() == 32 * 32

    def test_zero_input_sparsity_keeps_none(self):
        ds = StationPatchDataset(
            make_dataarray(n_vars=3, n_time=1, n_lat=32, n_lon=32),
            patch_size=32,
            stride=32,
            input_sparsity=0.0,
            output_sparsity=0.0,
            normalize=False,
        )
        _, _, input_mask, output_mask = ds[0]
        assert input_mask.sum() == 0
        assert output_mask.sum() == 0


# ── reproducibility ──────────────────────────────────────────────────────────


class TestReproducibility:
    def test_file_mask_is_deterministic(self, tmp_path):
        """File-based masks should return the same result on repeated access."""
        mask = np.random.default_rng(0).random((64, 64)) > 0.5
        mask_path = tmp_path / "mask.npy"
        np.save(mask_path, mask)
        ds = StationPatchDataset(
            make_dataarray(n_time=2, n_lat=64, n_lon=64),
            input_sparsity=str(mask_path),
            output_sparsity=0.5,
        )
        _, _, m1, _ = ds[0]
        _, _, m2, _ = ds[0]
        torch.testing.assert_close(m1, m2)

    def test_different_indices_can_differ(self):
        ds = StationPatchDataset(
            make_dataarray(n_time=2, n_lat=64, n_lon=64),
            input_sparsity=0.5,
            output_sparsity=0.5,
        )
        _, _, m0, _ = ds[0]
        _, _, m1, _ = ds[1]
        # With 50% sparsity on different seeds the masks should (almost certainly) differ
        assert not torch.equal(m0, m1)

    def test_random_mask_not_reproducible_flag(self):
        """Note: np.random.seed in __getitem__ does not seed torch.rand used
        in generate_mask, so random sparsity masks are NOT reproducible across
        repeated calls. This test documents that behaviour."""
        ds = StationPatchDataset(
            make_dataarray(n_time=1, n_lat=32, n_lon=32),
            input_sparsity=0.5,
            output_sparsity=0.5,
        )
        _, _, m1, _ = ds[0]
        _, _, m2, _ = ds[0]
        # Masks may differ — just assert shapes are correct
        assert m1.shape == m2.shape == (32, 32)


# ── DataLoader compatibility ────────────────────────────────────────────────


class TestDataLoader:
    def test_batched_loading(self):
        ds = StationPatchDataset(
            make_dataarray(n_time=2, n_lat=64, n_lon=64),
            patch_size=32,
            stride=32,
            input_sparsity=0.5,
            output_sparsity=0.5,
        )
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        sparse_input, sparse_target, input_mask, output_mask = batch
        assert sparse_input.shape == (4, 3, 32, 32)
        assert sparse_target.shape == (4, 1, 32, 32)
        assert input_mask.shape == (4, 32, 32)
        assert output_mask.shape == (4, 32, 32)
