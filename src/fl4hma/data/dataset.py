from __future__ import annotations

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class StationPatchDataset(Dataset):
    """
    PyTorch Dataset that extracts 32x32 spatial patches.
    """

    def __init__(
        self,
        dataarray: xr.DataArray,
        patch_size: int = 32,
        stride: int = 32,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
        input_sparsity: float | str | None = None,
        output_sparsity: float | None = None,
        transform=None,
    ):
        """
        Create the dataset patches

        Args:
            dataarray (xr.DataArray): data of dimension (variable, time, lat, lon)
            patch_size (int, optional): Spatial patch size (default 32)
            stride (int, optional): Sliding window stride (default 32, non-overlapping)
            normalize (bool, optional): Whether to normalize data (mean/std over full array)
            dtype (torch.dtype, optional): Torch dtype
            input_sparsity (float | str | None, optional): Fraction of input pixels to keep (default None).
                If None, uses centralised stationary station mask. If a string, loads the mask from the given file path.
            output_sparsity (float | None, optional): Fraction of output pixels to keep (default None).
                If None, uses output mask.
            transform (_type_, optional): Optional transform to be applied on a sample. Defaults to None.
        """

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

        # Convert to numpy (lazy-safe)
        self.data = self.da.values.astype(
            np.float32
        )  # shape (channel, time, lon, lat,)

        if normalize:
            self.mean = np.nanmean(self.data, axis=(1, 2, 3), keepdims=True)
            self.std = np.nanstd(self.data, axis=(1, 2, 3), keepdims=True) + 1e-6
            self.data = (self.data - self.mean) / self.std

        self.channels, self.time_len, self.lat_len, self.lon_len = self.data.shape

        # Precompute all patch indices
        self.indices = []
        for t in range(self.time_len):
            for i in range(0, self.lat_len - patch_size + 1, stride):
                for j in range(0, self.lon_len - patch_size + 1, stride):
                    self.indices.append((t, i, j))

        # Input masking strategy
        if input_sparsity is None:
            self.station_mask = np.load(
                "station_data/masks/stat/centralised_mask.npy"
            )  # shape (lat, lon)
        elif isinstance(input_sparsity, str):
            self.station_mask = np.load(input_sparsity)  # shape (lat, lon)
        else:
            self.input_sparsity = input_sparsity

        # Output masking strategy
        if output_sparsity is None:
            self.output_mask = np.load(
                "station_data/masks/out_mask.npy"
            )  # shape (lat, lon)
        else:
            self.output_sparsity = output_sparsity

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, i, j = self.indices[idx]

        patch = self.data[
            :,
            t,
            i : i + self.patch_size,
            j : j + self.patch_size,
        ]

        # Convert to torch tensor
        # Shape: [3, 32, 32] (channel-first)
        patch = torch.tensor(patch, dtype=self.dtype)  # .unsqueeze(0)
        np.random.seed(idx)  # For reproducible masks per sample

        input_mask = self.generate_patch_mask("input", patch, i, j, t)
        output_mask = self.generate_patch_mask("output", patch, i, j)

        # Create sparse input (mask out some pixels)
        sparse_input = patch.clone()
        sparse_input[0, ~input_mask] = 0.0  # Set masked pixels to 0

        # Create sparse target (only some pixels have labels)
        sparse_target = patch[0].clone().unsqueeze(0)  # Shape [1, 32, 32]
        sparse_target[0, ~output_mask] = -1.0  # Mark unlabeled pixels as -1

        return sparse_input, sparse_target, input_mask.float(), output_mask.float()

    def generate_patch_mask(
        self, type: str, patch: torch.Tensor, i: int, j: int, t: int = None
    ) -> torch.Tensor:
        """
        Generate a mask for the input or output based on the specified sparsity strategy.

        Args:
            type (str): The type of mask to generate ("input" or "output").
            patch (torch.Tensor): The patch for which to generate the mask.
            i (int): The starting index for the patch in the latitude dimension.
            j (int): The starting index for the patch in the longitude dimension.
            t (int, optional): The time index for the patch. Defaults to None.

        Returns:
            torch.Tensor: The generated mask.
        """

        # If the sparsity is defined as a fraction, generate a random mask
        if hasattr(self, type + "_sparsity"):
            mask = torch.rand(patch.shape[-2], patch.shape[-1]) < self.__getattribute__(
                type + "_sparsity"
            )

        # Otherwise use station mask for the input
        elif type == "input":

            # Check the station mask dimensions
            if self.station_mask.ndim == 2:
                mask = self.station_mask[
                    i : i + self.patch_size, j : j + self.patch_size
                ]

            if self.station_mask.ndim == 3:
                mask = self.station_mask[
                    t, i : i + self.patch_size, j : j + self.patch_size
                ]

            mask = torch.tensor(mask, dtype=torch.bool)

        else:  # Otherwise use output mask for the output
            mask = self.output_mask[i : i + self.patch_size, j : j + self.patch_size]
            mask = torch.tensor(mask, dtype=torch.bool)

        return mask
