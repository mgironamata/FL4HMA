import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from fl4hma.data.torch_dataset import StationPatchDataset, training_patch_origins
from fl4hma.training.training import train_sparse_pixel
from tests.test_datasets import make_dataarray


def make_dataset(tmp_path, dataset_class=StationPatchDataset):
    da = make_dataarray(n_vars=3, n_time=2, n_lat=8, n_lon=9)
    mask = np.indices((8, 9)).sum(axis=0) % 2 == 0
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    return dataset_class(
        da,
        input_mask_path=str(path),
        output_mask_path=str(path),
        patch_size=4,
        stride=2,
    )


def test_origins_refresh_preserving_budget_and_fixed_grid(tmp_path):
    ds = make_dataset(tmp_path)
    original = ds.indices
    loader = DataLoader(ds, batch_size=2)
    torch.manual_seed(14)
    with training_patch_origins(loader):
        first = list(ds.indices)
        assert [t for t, _, _ in first] == [t for t, _, _ in original]
        assert len(first) == len(original)
        assert all(0 <= row <= 4 and 0 <= col <= 5 for _, row, col in first)
        assert any(row % 2 or col % 2 for _, row, col in first)
    assert ds.indices is original
    with training_patch_origins(loader):
        assert ds.indices != first
    torch.manual_seed(14)
    with training_patch_origins(loader):
        assert ds.indices == first


@pytest.mark.parametrize("daily_mask", [False, True])
def test_masks_and_all_channels_use_sampled_origin(tmp_path, daily_mask):
    ds = make_dataset(tmp_path)
    if daily_mask:
        ds.station_mask = np.stack([ds.station_mask, ~ds.station_mask])
    with training_patch_origins(DataLoader(ds)):
        for index, (day, row, col) in enumerate(ds.indices):
            inputs, target, in_mask, out_mask = ds[index]
            expected = torch.from_numpy(ds.data[:, day, row : row + 4, col : col + 4])
            station_mask = ds.station_mask[day] if daily_mask else ds.station_mask
            mask = torch.from_numpy(station_mask[row : row + 4, col : col + 4])
            output = torch.from_numpy(ds.output_mask[row : row + 4, col : col + 4])
            torch.testing.assert_close(inputs[1:], expected[1:])
            torch.testing.assert_close(inputs[0][mask], expected[0][mask])
            assert (inputs[0][~mask] == 0).all()
            torch.testing.assert_close(target[0][output], expected[0][output])
            assert (target[0][~output] == -1).all()
            torch.testing.assert_close(in_mask.bool(), mask)
            torch.testing.assert_close(out_mask.bool(), output)


def test_restores_grid_after_failure_and_handles_subset(tmp_path):
    ds = make_dataset(tmp_path)
    original = ds.indices
    loader = DataLoader(Subset(ds, [0, 1, 2]))
    with pytest.raises(RuntimeError, match="failure"):
        with training_patch_origins(loader):
            assert ds.indices is not original
            raise RuntimeError("failure")
    assert ds.indices is original


def test_persistent_workers_rejected(tmp_path):
    ds = make_dataset(tmp_path)
    loader = DataLoader(ds, num_workers=1, persistent_workers=True)
    with pytest.raises(ValueError, match="persistent_workers"):
        with training_patch_origins(loader):
            pass


def test_country_copy_keeps_its_fixed_indices(tmp_path):
    ds = make_dataset(tmp_path)
    other = ds.with_input_mask(str(tmp_path / "mask.npy"))
    original = ds.indices
    assert other.indices is original
    with training_patch_origins(DataLoader(ds)):
        assert other.indices is original
        assert ds.indices is not other.indices
    assert ds.indices is other.indices


def test_nonpersistent_workers_see_new_origins(tmp_path):
    ds = make_dataset(tmp_path)
    loader = DataLoader(ds, batch_size=len(ds), num_workers=1)
    with training_patch_origins(loader):
        actual = next(iter(loader))[0]
        expected = torch.stack([ds[index][0] for index in range(len(ds))])
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("epochs_per_call", [1, 2])
def test_training_refreshes_each_epoch_and_restores_grid(
    tmp_path, private, epochs_per_call
):
    seen = []

    class RecordingDataset(StationPatchDataset):
        def __getitem__(self, index):
            if index == 0:
                seen.append(list(self.indices))
            return super().__getitem__(index)

    ds = make_dataset(tmp_path, RecordingDataset)
    original = ds.indices
    model = torch.nn.Conv2d(3, 1, 1)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
    for _ in range(2 // epochs_per_call):
        if private:
            from fl4hma.federation.differential_privacy import (
                DPConfig,
                dp_train_sparse_pixel,
            )

            dp_train_sparse_pixel(
                model, loader, DPConfig(noise_multiplier=0), epochs=epochs_per_call
            )
        else:
            train_sparse_pixel(model, loader, epochs=epochs_per_call)
        assert ds.indices is original
    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert ds.indices is original
