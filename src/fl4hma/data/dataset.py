import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np


class StationPatchDataset(Dataset):
    """
    PyTorch Dataset that extracts 32x32 spatial patches
    from an xarray DataArray with dimensions (time, lat, lon) 
    using the station mask.
    """

    def __init__(
        self,
        dataarray: xr.DataArray,
        patch_size: int = 32,
        stride: int = 32,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
        input_sparsity: float = None,
        output_sparsity: float = None, 
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
            input_sparsity (float, optional): Fraction of input pixels to keep (default None). If None, uses station mask.
            output_sparsity (float, optional): Fraction of output pixels to keep (default None). If None, uses output mask.
            transform (_type_, optional): Optional transform to be applied on a sample. Defaults to None.
        """

        assert set(dataarray.dims) == {"variable", "time", "lat", "lon"}, \
            "DataArray must have dims ('variable', 'time', 'lat', 'lon')"

        self.da = dataarray
        self.patch_size = patch_size
        self.stride = stride
        self.dtype = dtype

        # Convert to numpy (lazy-safe)
        self.data = self.da.values.astype(np.float32) # shape (channel, time, lon, lat,)

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

        if input_sparsity is None:
            self.station_mask = np.load("station_data/masks/centralised_mask.npy") # shape (lat, lon)
        else:
            self.input_sparsity = input_sparsity
        if output_sparsity is None:
            self.output_mask = np.load("station_data/masks/out_mask.npy") # shape (lat, lon)
        else:
            self.output_sparsity = output_sparsity

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t, i, j = self.indices[idx]

        patch = self.data[:,
            t,
            i : i + self.patch_size,
            j : j + self.patch_size,
        ]

        # Convert to torch tensor
        # Shape: [3, 32, 32] (channel-first)
        patch = torch.tensor(patch, dtype=self.dtype) #.unsqueeze(0)
        np.random.seed(idx)  # For reproducible masks per sample

        if hasattr(self, "input_sparsity"):
            # Create input mask: which pixels are visible in the input
            input_mask = torch.rand(patch.shape[-2], patch.shape[-1]) < self.input_sparsity

        else:
            input_mask = self.station_mask[i : i + self.patch_size,
                                 j : j + self.patch_size]
            input_mask = torch.tensor(input_mask, dtype=torch.bool)
        
        # Output mask: which pixels have target labels
        if hasattr(self, "output_sparsity"):
            output_mask = torch.rand(patch.shape[-2], patch.shape[-1]) < self.output_sparsity
        else:
            output_mask = self.output_mask[i : i + self.patch_size,
                                 j : j + self.patch_size]
            output_mask = torch.tensor(output_mask, dtype=torch.bool)
        
        # Create sparse input (mask out some pixels)
        sparse_input = patch.clone()
        sparse_input[0, ~input_mask] = 0.0  # Set masked pixels to 0
        
        # Create sparse target (only some pixels have labels)
        sparse_target = patch[0].clone().unsqueeze(0)  # Shape [1, 32, 32]
        sparse_target[0, ~output_mask] = -1.0  # Mark unlabeled pixels as -1
        
        return sparse_input, sparse_target, input_mask.float(), output_mask.float()