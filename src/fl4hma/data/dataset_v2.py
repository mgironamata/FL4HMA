from typing import Optional

import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np


class StationPatchDataset(Dataset):
    """
    PyTorch Dataset that extracts spatial patches from an xarray DataArray
    and applies station-based masking.

    Parameters
    ----------
    dataarray : xr.DataArray
        Shape (variable, time, lat, lon).
    input_mask_path : str or None
        Path to a ``.npy`` binary mask for input masking.
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
    ):
        assert set(dataarray.dims) == {"variable", "time", "lat", "lon"}, \
            "DataArray must have dims ('variable', 'time', 'lat', 'lon')"

        self.patch_size = patch_size
        self.stride = stride
        self.dtype = dtype

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
            self._station_mask = np.load(input_mask_path)
        elif input_sparsity is not None:
            self._input_sparsity = input_sparsity
        else:
            raise ValueError("Provide input_mask_path or input_sparsity")

        # Output mask
        if output_mask_path is not None:
            self._output_mask = np.load(output_mask_path)
        elif output_sparsity is not None:
            self._output_sparsity = output_sparsity
        else:
            raise ValueError("Provide output_mask_path or output_sparsity")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, i, j = self.indices[idx]
        ps = self.patch_size

        patch = self.data[:, t, i:i + ps, j:j + ps]
        patch = torch.tensor(patch, dtype=self.dtype)

        np.random.seed(idx)

        # Input mask
        if hasattr(self, "_station_mask"):
            input_mask = torch.tensor(
                self._station_mask[i:i + ps, j:j + ps], dtype=torch.bool,
            )
        else:
            input_mask = torch.rand(ps, ps) < self._input_sparsity

        # Output mask
        if hasattr(self, "_output_mask"):
            output_mask = torch.tensor(
                self._output_mask[i:i + ps, j:j + ps], dtype=torch.bool,
            )
        else:
            output_mask = torch.rand(ps, ps) < self._output_sparsity

        # Apply input mask (zero out missing pixels in channel 0)
        sparse_input = patch.clone()
        sparse_input[0, ~input_mask] = 0.0

        # Sparse target (only labelled pixels; unlabelled = -1)
        sparse_target = patch[0].clone().unsqueeze(0)
        sparse_target[0, ~output_mask] = -1.0

        return sparse_input, sparse_target, input_mask.float(), output_mask.float()