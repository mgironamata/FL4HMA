from typing import Dict, Optional

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class StationPatchDataset(Dataset):
    """
    PyTorch Dataset that extracts spatial patches from an xarray DataArray
    and applies station-based or random masking.

    Parameters
    ----------
    dataarray : xr.DataArray
        Shape (variable, time, lat, lon).
    input_mask_path : str or None
        Path to a ``.npy`` binary mask for input masking.
        Supports 2D (lat, lon) or 3D (year, lat, lon) masks.
        If None, ``input_sparsity`` is used (random mask).
    output_mask_path : str or None
        Path to a ``.npy`` binary mask for output masking.
        If None, ``output_sparsity`` is used (random mask).
    input_sparsity : float or None
        Probability a pixel is visible when no mask file is given.
    output_sparsity : float or None
        Probability a pixel has a label when no mask file is given.
    patch_size : int
    stride : int
    normalize : bool
    mask_start_year : int
        The first year covered by a 3D yearly mask (default 1998).
    """

    def __init__(
        self,
        dataarray: xr.DataArray,
        input_mask_path: Optional[str] = None,
        output_mask_path: Optional[str] = None,
        input_sparsity: Optional[float] = None,
        output_sparsity: Optional[float] = None,
        patch_size: int = 32,
        stride: int = 32,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
        mask_start_year: int = 1998,
    ):
        assert set(dataarray.dims) == {
            "variable",
            "time",
            "lat",
            "lon",
        }, "DataArray must have dims ('variable', 'time', 'lat', 'lon')"

        self.da = dataarray
        self.patch_size = patch_size
        self.stride = stride
        self.dtype = dtype
        self.mask_start_year = mask_start_year

        # Convert to numpy
        self.data = dataarray.values.astype(np.float32)

        if normalize:
            self.mean = np.nanmean(self.data, axis=(1, 2, 3), keepdims=True)
            self.std = np.nanstd(self.data, axis=(1, 2, 3), keepdims=True) + 1e-6
            self.data = (self.data - self.mean) / self.std

        self.channels, self.time_len, self.lat_len, self.lon_len = self.data.shape

        # Precompute patch indices
        self.indices = []
        for t in range(self.time_len):
            for i in range(0, self.lat_len - patch_size + 1, stride):
                for j in range(0, self.lon_len - patch_size + 1, stride):
                    self.indices.append((t, i, j))

        # Input mask
        if input_mask_path is not None:
            self.station_mask = np.load(input_mask_path)
            if self.station_mask.ndim > 2:
                self.station_mask = self._expand_yearly_to_daily(self.station_mask)
        elif input_sparsity is not None:
            self.input_sparsity = input_sparsity
        else:
            raise ValueError("Provide input_mask_path or input_sparsity")

        # Output mask
        if output_mask_path is not None:
            self.output_mask = np.load(output_mask_path)
        elif output_sparsity is not None:
            self.output_sparsity = output_sparsity
        else:
            raise ValueError("Provide output_mask_path or output_sparsity")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if idx >= len(self.indices) or idx < -len(self.indices):
            raise IndexError(
                f"index {idx} out of range for dataset of length {len(self)}"
            )

        t, i, j = self.indices[idx]
        ps = self.patch_size

        patch = self.data[:, t, i : i + ps, j : j + ps]
        patch = torch.tensor(patch, dtype=self.dtype)

        np.random.seed(idx)

        # Input mask
        if hasattr(self, "station_mask"):
            if self.station_mask.ndim == 2:
                input_mask = torch.tensor(
                    self.station_mask[i : i + ps, j : j + ps],
                    dtype=torch.bool,
                )
            else:
                input_mask = torch.tensor(
                    self.station_mask[t, i : i + ps, j : j + ps],
                    dtype=torch.bool,
                )
        else:
            input_mask = torch.rand(ps, ps) < self.input_sparsity

        # Output mask
        if hasattr(self, "output_mask"):
            output_mask = torch.tensor(
                self.output_mask[i : i + ps, j : j + ps],
                dtype=torch.bool,
            )
        else:
            output_mask = torch.rand(ps, ps) < self.output_sparsity

        # Apply input mask (zero out missing pixels in channel 0)
        sparse_input = patch.clone()
        sparse_input[0, ~input_mask] = 0.0

        # Sparse target (only labelled pixels; unlabelled = -1)
        sparse_target = patch[0].clone().unsqueeze(0)
        sparse_target[0, ~output_mask] = -1.0

        return sparse_input, sparse_target, input_mask.float(), output_mask.float()

    def _expand_yearly_to_daily(self, yearly_mask: np.ndarray) -> np.ndarray:
        """
        Expand a yearly mask (year, lat, lon) to daily resolution (time, lat, lon),
        sliced to the dataset's time period and accounting for leap years.
        """
        start_year = self.da.time.dt.year.min().item()
        end_year = self.da.time.dt.year.max().item()
        yearly_mask = yearly_mask[
            start_year - self.mask_start_year : end_year - self.mask_start_year + 1
        ]

        daily_mask = []
        for year in range(start_year, end_year + 1):
            year_mask = yearly_mask[year - start_year]
            days_in_year = (
                366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
            )
            daily_mask.append(
                np.repeat(year_mask[np.newaxis, :, :], days_in_year, axis=0)
            )

        return np.concatenate(daily_mask, axis=0)


def build_country_datasets(
    da: xr.DataArray,
    country_masks: Dict[str, str],
    output_mask_path: str,
    patch_size: int = 32,
    stride: int = 32,
) -> Dict[str, StationPatchDataset]:
    """Create one ``StationPatchDataset`` per country."""
    datasets = {}
    for country, mask_path in country_masks.items():
        datasets[country] = StationPatchDataset(
            dataarray=da,
            input_mask_path=mask_path,
            output_mask_path=output_mask_path,
            patch_size=patch_size,
            stride=stride,
        )
    return datasets
