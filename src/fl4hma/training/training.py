"""Training and evaluation loops for sparse pixel models."""

from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xarray as xr
from torch.utils.data import DataLoader

from fl4hma.data.torch_dataset import StationPatchDataset
from fl4hma.models.unet import UNetCNN, sparse_pixel_loss


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train_sparse_pixel(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.001,
) -> float:
    """Train U-Net on sparse pixel data.  Returns average loss."""
    device = _get_device()
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    n_batches = 0

    for _ in range(epochs):
        for sparse_in, sparse_tgt, _, output_mask in loader:
            sparse_in = sparse_in.to(device)
            sparse_tgt = sparse_tgt.to(device)
            output_mask = output_mask.to(device)

            optimizer.zero_grad()
            pred = model(sparse_in)
            loss = sparse_pixel_loss(pred, sparse_tgt, output_mask)
            if loss is not None and not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

    return total_loss / max(1, n_batches)


def evaluate_sparse_pixel(
    model: nn.Module,
    loader: DataLoader,
) -> Dict[str, float]:
    """Evaluate on sparse pixel data.  Returns dict with mse, rmse, loss."""
    device = _get_device()
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for sparse_in, sparse_tgt, _, output_mask in loader:
            sparse_in = sparse_in.to(device)
            sparse_tgt = sparse_tgt.to(device)
            output_mask = output_mask.to(device)

            pred = model(sparse_in)
            loss = sparse_pixel_loss(pred, sparse_tgt, output_mask)
            if loss is not None and not torch.isnan(loss):
                total_loss += loss.item()

                # MSE on labelled pixels
                for b in range(pred.size(0)):
                    mask_b = output_mask[b].bool()
                    if mask_b.sum() > 0:
                        mse_b = F.mse_loss(
                            pred[b][:, mask_b],
                            sparse_tgt[b][:, mask_b],
                        )
                        total_mse += mse_b.item()
                n_batches += 1

    if n_batches == 0:
        return {"loss": float("nan"), "mse": float("nan"), "rmse": float("nan")}
    avg_loss = total_loss / n_batches
    avg_mse = total_mse / n_batches
    return {"loss": avg_loss, "mse": avg_mse, "rmse": np.sqrt(avg_mse)}


def evaluate_model_with_mask(
    model: nn.Module,
    da_test: xr.DataArray,
    input_mask_path: str,
    output_mask_path: str,
    batch_size: int = 16,
    patch_size: int = 32,
    stride: int = 32,
) -> Dict[str, float]:
    """Evaluate a model using a specific input mask on the test set.

    Standardised evaluation entry point used by all experiments.
    """
    ds = StationPatchDataset(
        da_test,
        input_mask_path=input_mask_path,
        output_mask_path=output_mask_path,
        patch_size=patch_size,
        stride=stride,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return evaluate_sparse_pixel(model, loader)
