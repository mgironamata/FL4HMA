"""Training and evaluation loops for sparse pixel models."""

from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from torch.utils.data import DataLoader

from fl4hma.data.torch_dataset import StationPatchDataset
from fl4hma.models.unet import (
    UNetCNN,
    enable_mc_dropout,
    ensemble_forward,
    sparse_pixel_loss,
)
from fl4hma.training.losses import LossConfig, sparse_crps_loss, sparse_tweedie_loss


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


def _forward_and_loss(
    model: nn.Module,
    sparse_in: torch.Tensor,
    sparse_tgt: torch.Tensor,
    output_mask: torch.Tensor,
    cfg: LossConfig,
):
    """Run the model and evaluate the configured objective.

    Args:
        model: Model to run.
        sparse_in: Masked input batch.
        sparse_tgt: Targets, with ``-1`` on unlabelled pixels.
        output_mask: Binary mask of labelled pixels.
        cfg: Objective selector.

    Returns:
        ``(loss, point_prediction, ensemble)`` where ``ensemble`` is ``None``
        for deterministic objectives. ``point_prediction`` is in the same units
        as ``sparse_tgt``: the ensemble mean for CRPS, ``exp(eta)`` for Tweedie.
    """
    if cfg.name == "crps":
        ensemble = ensemble_forward(model, sparse_in, cfg.n_members, cfg.chunk_size)
        loss = sparse_crps_loss(ensemble, sparse_tgt, output_mask, fair=cfg.fair)
        return loss, ensemble.mean(dim=1, keepdim=True), ensemble

    raw = model(sparse_in)
    if cfg.name == "tweedie":
        loss = sparse_tweedie_loss(
            raw, sparse_tgt, output_mask, p=cfg.tweedie_p, eta_clamp=cfg.eta_clamp
        )
        point = torch.exp(raw.clamp(-cfg.eta_clamp, cfg.eta_clamp))
        return loss, point, None

    return sparse_pixel_loss(raw, sparse_tgt, output_mask), raw, None


def train_sparse_pixel(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.001,
    loss_config: Optional[LossConfig] = None,
) -> float:
    """Train U-Net on sparse pixel data.

    Args:
        model: Model to train, updated in place.
        loader: Training loader.
        epochs: Local epochs over the loader.
        lr: Adam learning rate.
        loss_config: Objective selector. Defaults to MSE, reproducing the
            behaviour of every caller that predates this argument. Under CRPS
            the model stays in train mode, so dropout is already sampling and
            ``n_members`` replicas are drawn per step.

    Returns:
        Mean training loss over batches.
    """
    cfg = loss_config or LossConfig()
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
            loss, _, _ = _forward_and_loss(
                model, sparse_in, sparse_tgt, output_mask, cfg
            )
            if loss is not None and not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

    return total_loss / max(1, n_batches)


def evaluate_sparse_pixel(
    model: nn.Module,
    loader: DataLoader,
    loss_config: Optional[LossConfig] = None,
) -> Dict[str, float]:
    """Evaluate on sparse pixel data.

    Every score is pooled over the labelled pixels the loader yields, so all of
    them are independent of ``loader.batch_size``.

    The deterministic scores are computed on a single point prediction — the
    ensemble mean under CRPS, ``exp(eta)`` under Tweedie, the raw output under
    MSE — which makes them comparable across objectives. ``crps`` is likewise
    comparable: the CRPS of a point forecast is its absolute error, so a
    deterministic model scores ``crps == mae`` and an ensemble is only better
    if its spread is informative.

    Scores are in whatever units the loader serves. With
    ``StationPatchDataset(physical_target=True)`` that is mm; otherwise
    z-scores, which are not comparable between variables or runs.

    Args:
        model: Model to evaluate.
        loader: Loader yielding
            ``(sparse_input, sparse_target, input_mask, output_mask)``.
        loss_config: Objective selector. Defaults to MSE.

    Returns:
        Dict with ``loss``, ``mse``, ``rmse``, ``mae``, ``bias``, ``crps`` and
        ``spread``. ``bias`` is mean(prediction - observation), so positive
        means over-prediction. All are ``nan`` when no pixel is labelled.
    """
    cfg = loss_config or LossConfig()
    device = _get_device()
    model.to(device)
    if cfg.is_ensemble:
        enable_mc_dropout(model)
    else:
        model.eval()

    total_loss = 0.0
    n_batches = 0
    total_pixels = 0
    sq_error = abs_error = signed_error = crps_total = spread_total = 0.0

    with torch.no_grad():
        for sparse_in, sparse_tgt, _, output_mask in loader:
            sparse_in = sparse_in.to(device)
            sparse_tgt = sparse_tgt.to(device)
            output_mask = output_mask.to(device)

            loss, point, ensemble = _forward_and_loss(
                model, sparse_in, sparse_tgt, output_mask, cfg
            )
            if loss is None or torch.isnan(loss):
                continue
            total_loss += loss.item()
            n_batches += 1

            labelled = output_mask.bool().unsqueeze(1).expand_as(point)
            n_labelled = int(labelled.sum().item())
            if n_labelled == 0:
                continue

            error = (point - sparse_tgt)[labelled]
            sq_error += (error**2).sum().item()
            abs_error += error.abs().sum().item()
            signed_error += error.sum().item()
            total_pixels += n_labelled

            if ensemble is None:
                # A point forecast's CRPS is its absolute error.
                crps_total += error.abs().sum().item()
            else:
                crps_total += loss.item() * n_labelled
                spread_total += ensemble.std(dim=1, keepdim=True)[labelled].sum().item()

    nan = float("nan")
    if n_batches == 0:
        return {
            k: nan for k in ("loss", "mse", "rmse", "mae", "bias", "crps", "spread")
        }
    avg_loss = total_loss / n_batches
    if total_pixels == 0:
        return {
            "loss": avg_loss,
            "mse": nan,
            "rmse": nan,
            "mae": nan,
            "bias": nan,
            "crps": nan,
            "spread": nan,
        }

    mse = sq_error / total_pixels
    return {
        "loss": avg_loss,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": abs_error / total_pixels,
        "bias": signed_error / total_pixels,
        "crps": crps_total / total_pixels,
        "spread": spread_total / total_pixels,
    }


def evaluate_model_with_mask(
    model: nn.Module,
    da_test: xr.DataArray,
    input_mask_path: str,
    output_mask_path: str,
    batch_size: int = 16,
    patch_size: int = 32,
    stride: int = 32,
    loss_config: Optional[LossConfig] = None,
) -> Dict[str, float]:
    """Evaluate a model using a specific input mask on the test set.

    Standardised evaluation entry point used by all experiments. The target
    units follow *loss_config*, so a Tweedie run is scored in mm while an MSE
    run is scored in z-scores.

    Args:
        model: Model to evaluate.
        da_test: Test data array with dims (variable, time, lat, lon).
        input_mask_path: Station mask supplied to the model as input.
        output_mask_path: Mask of pixels the model is scored on.
        batch_size: Evaluation batch size; no score depends on it.
        patch_size: Spatial patch size.
        stride: Patch stride.
        loss_config: Objective selector. Defaults to MSE.

    Returns:
        The metric dict from :func:`evaluate_sparse_pixel`.
    """
    cfg = loss_config or LossConfig()
    ds = StationPatchDataset(
        da_test,
        input_mask_path=input_mask_path,
        output_mask_path=output_mask_path,
        patch_size=patch_size,
        stride=stride,
        physical_target=cfg.physical_target,
        target_min=cfg.target_min,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return evaluate_sparse_pixel(model, loader, loss_config=cfg)
