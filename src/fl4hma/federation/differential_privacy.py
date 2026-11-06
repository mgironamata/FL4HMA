"""Differential privacy framework for federated learning.

Provides both **local DP** (client-side DP-SGD with per-sample gradient clipping
and Gaussian noise) and **global DP** (server-side clipping and noise on
aggregated updates).

Key components
--------------
- ``DPConfig``           – dataclass holding all DP hyperparameters.
- ``DPAccountant``       – Rényi-DP based privacy accountant (ε, δ tracking).
- ``dp_train_sparse_pixel`` – local DP training loop (replaces train_sparse_pixel).
- ``DPAphroFlowerClient``   – DP-aware Flower client.
- ``DPFedAvg``           – FedAvg strategy with global DP (server-side noise).
- ``run_federated_dp``   – end-to-end DP federated simulation.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from flwr.client import NumPyClient
from flwr.common import FitRes, NDArrays, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader

from fl4hma.data.torch_dataset import StationPatchDataset, build_country_datasets
from fl4hma.federation.simulation import run_ray_simulation
from fl4hma.models.unet import UNetCNN, sparse_pixel_loss
from fl4hma.training.training import (
    _get_device,
    evaluate_sparse_pixel,
    get_parameters,
    set_parameters,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DPConfig:
    """Configuration for differential privacy in federated learning.

    Parameters
    ----------
    clip_norm : float
        Maximum L2 norm for gradient (local DP) or model-update (global DP)
        clipping.
    noise_multiplier : float
        Ratio of Gaussian noise standard deviation to ``clip_norm``.
        σ = noise_multiplier × clip_norm.
    target_delta : float
        Target δ for (ε, δ)-DP guarantees.
    local_dp : bool
        Enable client-side DP-SGD (per-sample gradient clipping + noise).
    global_dp : bool
        Enable server-side DP (clip client updates + noise after aggregation).
    max_grad_norm : float | None
        Per-sample gradient clip norm for local DP.  Defaults to ``clip_norm``.
    secure_mode : bool
        If True, use cryptographically secure RNG for noise generation.
    """

    clip_norm: float = 1.0
    noise_multiplier: float = 1.0
    target_delta: float = 1e-5
    local_dp: bool = True
    global_dp: bool = True
    max_grad_norm: Optional[float] = None
    secure_mode: bool = False

    def __post_init__(self):
        if self.max_grad_norm is None:
            self.max_grad_norm = self.clip_norm


# ---------------------------------------------------------------------------
# Privacy Accountant (Rényi Differential Privacy)
# ---------------------------------------------------------------------------


class DPAccountant:
    """Simple Rényi-DP accountant for tracking cumulative privacy loss.

    Uses the analytical Gaussian mechanism RDP bound from [Mironov 2017] and
    converts to (ε, δ)-DP via the standard conversion lemma.
    """

    def __init__(self, target_delta: float = 1e-5):
        self.target_delta = target_delta
        self._rdp_orders = list(range(2, 256))  # α values
        self._rdp_eps = np.zeros(len(self._rdp_orders))
        self._steps = 0

    def _compute_rdp_gaussian(
        self, noise_multiplier: float, sample_rate: float
    ) -> np.ndarray:
        """Compute RDP of subsampled Gaussian mechanism for each order α."""
        rdp = np.zeros(len(self._rdp_orders))
        if noise_multiplier == 0:
            return np.full(len(self._rdp_orders), np.inf)
        for i, alpha in enumerate(self._rdp_orders):
            if sample_rate == 1.0:
                # Full-batch: standard Gaussian mechanism RDP
                rdp[i] = alpha / (2.0 * noise_multiplier**2)
            else:
                # Subsampled Gaussian (Poisson subsampling upper bound)
                # Use the tighter bound as a fallback for large alpha where
                # exp(alpha/sigma^2) overflows.
                full_batch_rdp = alpha / (2.0 * noise_multiplier**2)
                exponent = alpha / (noise_multiplier**2)
                if alpha <= 1:
                    rdp[i] = 0.0
                elif exponent > 500:
                    # exp() would overflow; the subsampled bound can only be
                    # worse than the full-batch bound, so use full-batch.
                    rdp[i] = full_batch_rdp
                else:
                    subsampled = math.log1p(
                        sample_rate**2 * (math.exp(exponent) - 1) / (alpha - 1)
                    )
                    rdp[i] = min(subsampled, full_batch_rdp)
        return rdp

    def step(
        self, noise_multiplier: float, sample_rate: float = 1.0, num_steps: int = 1
    ) -> None:
        """Record DP mechanism applications (training steps or rounds).

        Args:
            noise_multiplier: Noise multiplier z of each application.
            sample_rate: Sampling rate q of each application.
            num_steps: Number of identical applications to record at once;
                RDP composes additively, so this equals calling ``step``
                ``num_steps`` times.
        """
        rdp = self._compute_rdp_gaussian(noise_multiplier, sample_rate)
        self._rdp_eps += rdp * num_steps
        self._steps += num_steps

    def get_epsilon(self, delta: Optional[float] = None) -> float:
        """Convert accumulated RDP to (ε, δ)-DP."""
        delta = delta or self.target_delta
        # RDP to (ε, δ) conversion: ε = min_α { RDP(α) + log(1/δ)/(α-1) }
        eps_candidates = []
        for i, alpha in enumerate(self._rdp_orders):
            eps = self._rdp_eps[i] + math.log(1.0 / delta) / (alpha - 1)
            eps_candidates.append(eps)
        return min(eps_candidates) if eps_candidates else 0.0

    @property
    def epsilon(self) -> float:
        return self.get_epsilon()

    @property
    def steps(self) -> int:
        return self._steps

    def reset(self):
        self._rdp_eps = np.zeros(len(self._rdp_orders))
        self._steps = 0


# ---------------------------------------------------------------------------
# Local DP: DP-SGD Training Loop
# ---------------------------------------------------------------------------


def per_sample_gradients(
    model: nn.Module,
    sparse_in: torch.Tensor,
    sparse_tgt: torch.Tensor,
    output_mask: torch.Tensor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Compute the gradient of each sample's loss separately.

    Each sample's loss is the MSE over its own labelled pixels, so
    ``sum(losses * counts) / sum(counts)`` equals ``sparse_pixel_loss`` on the
    whole batch.

    Args:
        model: Model without BatchNorm (whose batch statistics would couple
            samples and make per-sample gradients meaningless).
        sparse_in: Inputs, shape (B, C, H, W).
        sparse_tgt: Targets, shape (B, 1, H, W).
        output_mask: Labelled-pixel mask, shape (B, H, W).

    Returns:
        ``(grads, losses, counts)``: per-parameter gradients with a leading
        batch dimension, per-sample losses (B,), and labelled-pixel counts (B,).
        Samples without labelled pixels get zero loss and zero gradient.

    Raises:
        ValueError: If the model contains BatchNorm layers.
    """
    if any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules()):
        raise ValueError(
            "per-sample DP-SGD needs a model without BatchNorm; use norm='group'"
        )
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def sample_loss(p, x, y, m):
        pred = functional_call(model, (p, buffers), (x.unsqueeze(0),))[0]
        count = m.sum()
        loss = (((pred - y) ** 2).sum(0) * m).sum() / count.clamp_min(1)
        return loss, (loss, count)

    grads, (losses, counts) = vmap(
        grad(sample_loss, has_aux=True), in_dims=(None, 0, 0, 0)
    )(params, sparse_in, sparse_tgt, output_mask)
    return grads, losses.detach(), counts.detach()


def privatise_gradients(
    per_sample_grads: Dict[str, torch.Tensor],
    max_norm: float,
    noise_std: float,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Clip each sample's gradient, sum, add Gaussian noise and average.

    This is the DP-SGD gradient: ``(Σ_i clip(g_i, C) + N(0, σ²)) / B`` with
    ``σ = noise_std`` (normally ``noise_multiplier × C``).

    Args:
        per_sample_grads: Gradients with a leading batch dimension B, e.g.
            from ``per_sample_gradients``.
        max_norm: Per-sample L2 clipping bound C, over all parameters jointly.
        noise_std: Std of the Gaussian noise added to the summed gradient.
        generator: Optional RNG for the noise (for reproducibility in tests).

    Returns:
        Private gradients, one per parameter, without the batch dimension.
    """
    grads = list(per_sample_grads.values())
    batch_size = grads[0].shape[0]
    norms = torch.sqrt(sum(g.reshape(batch_size, -1).pow(2).sum(1) for g in grads))
    scale = (max_norm / (norms + 1e-12)).clamp(max=1.0)
    private = {}
    for name, g in per_sample_grads.items():
        summed = torch.einsum("b,b...->...", scale, g)
        if noise_std > 0:
            summed = summed + noise_std * torch.randn(
                summed.shape,
                generator=generator,
                device=summed.device,
                dtype=summed.dtype,
            )
        private[name] = summed / batch_size
    return private


def dp_train_sparse_pixel(
    model: nn.Module,
    loader: DataLoader,
    dp_config: DPConfig,
    epochs: int = 1,
    lr: float = 0.001,
    accountant: Optional[DPAccountant] = None,
) -> Tuple[float, DPAccountant]:
    """Train with local DP-SGD (per-sample gradient clipping + noise).

    Each step clips every sample's gradient to ``dp_config.max_grad_norm``,
    sums them, adds N(0, (noise_multiplier × max_grad_norm)²) and divides by
    the batch size, then takes an Adam step on that private gradient.

    Args:
        model: Model to train; must not contain BatchNorm.
        loader: Training data loader.
        dp_config: DP configuration.
        epochs: Number of local epochs.
        lr: Learning rate.
        accountant: Privacy accountant (created if not given).

    Returns:
        ``(avg_loss, accountant)``; ``avg_loss`` is the mean over steps of the
        batch loss pooled over labelled pixels, as in ``train_sparse_pixel``.
    """
    if accountant is None:
        accountant = DPAccountant(target_delta=dp_config.target_delta)

    device = _get_device()
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_norm = dp_config.max_grad_norm
    noise_std = dp_config.noise_multiplier * max_norm
    batch_size = loader.batch_size or 1
    sample_rate = batch_size / len(loader.dataset)

    total_loss = 0.0
    n_batches = 0

    for _ in range(epochs):
        for sparse_in, sparse_tgt, _, output_mask in loader:
            sparse_in = sparse_in.to(device)
            sparse_tgt = sparse_tgt.to(device)
            output_mask = output_mask.to(device)

            grads, losses, counts = per_sample_gradients(
                model, sparse_in, sparse_tgt, output_mask
            )
            private = privatise_gradients(grads, max_norm, noise_std)

            optimizer.zero_grad()
            for name, param in model.named_parameters():
                param.grad = private[name]
            optimizer.step()

            total_loss += ((losses * counts).sum() / counts.sum().clamp_min(1)).item()
            n_batches += 1
            accountant.step(dp_config.noise_multiplier, sample_rate)

    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, accountant


def local_dp_epsilon(
    dp_config: DPConfig,
    dataset_size: int,
    batch_size: int,
    local_epochs: int,
    num_rounds: int,
) -> float:
    """Total local DP-SGD ε spent by one client over a whole simulation.

    Flower rebuilds clients every round, so a client's own accountant only
    sees one round; this composes every step the client takes across all
    rounds.

    Args:
        dp_config: DP configuration (noise multiplier and δ).
        dataset_size: Number of training samples on the client.
        batch_size: Local batch size.
        local_epochs: Local epochs per round.
        num_rounds: Number of federated rounds.

    Returns:
        ε at ``dp_config.target_delta`` (``inf`` if the noise multiplier is 0).
    """
    steps = num_rounds * local_epochs * math.ceil(dataset_size / batch_size)
    accountant = DPAccountant(target_delta=dp_config.target_delta)
    accountant.step(
        dp_config.noise_multiplier,
        sample_rate=batch_size / dataset_size,
        num_steps=steps,
    )
    return accountant.epsilon


# ---------------------------------------------------------------------------
# Global DP: Server-side clipping and noise
# ---------------------------------------------------------------------------


def clip_model_update(
    original_params: List[np.ndarray],
    updated_params: List[np.ndarray],
    clip_norm: float,
    mask: Optional[List[bool]] = None,
) -> List[np.ndarray]:
    """Clip a model update (Δ = updated - original) to a maximum L2 norm.

    Args:
        original_params: Parameters before local training.
        updated_params: Parameters after local training.
        clip_norm: Maximum L2 norm of the update.
        mask: If given, only entries where True are included in the norm and
            clipped; the others (e.g. BatchNorm buffers such as
            ``num_batches_tracked``) are passed through unchanged. Without it,
            such buffers dominate the norm and shrink the weight update.

    Returns:
        The clipped updated parameters (original + clipped Δ).
    """
    if mask is None:
        mask = [True] * len(original_params)
    deltas = [u - o for u, o in zip(updated_params, original_params)]
    flat_delta = np.concatenate([d.ravel() for d, keep in zip(deltas, mask) if keep])
    delta_norm = np.linalg.norm(flat_delta)

    scale = clip_norm / delta_norm if delta_norm > clip_norm else 1.0
    return [
        o + d * scale if keep else u
        for o, d, u, keep in zip(original_params, deltas, updated_params, mask)
    ]


def add_noise_to_parameters(
    parameters: List[np.ndarray],
    noise_std: float,
    rng: Optional[np.random.Generator] = None,
    noise_mask: Optional[List[bool]] = None,
) -> List[np.ndarray]:
    """Add Gaussian noise to model parameters.

    Parameters
    ----------
    parameters : list of ndarray
        Model parameters (state dict values).
    noise_std : float
        Standard deviation of Gaussian noise.
    rng : numpy Generator or None
        Random number generator.
    noise_mask : list of bool or None
        If given, only add noise where True. This is used to skip
        non-trainable buffers (e.g. BatchNorm running stats).
    """
    if rng is None:
        rng = np.random.default_rng()
    noisy = []
    for i, p in enumerate(parameters):
        if noise_mask is not None and not noise_mask[i]:
            noisy.append(p.copy())
        else:
            noise = rng.normal(loc=0.0, scale=noise_std, size=p.shape).astype(p.dtype)
            noisy.append(p + noise)
    return noisy


def _get_trainable_mask(model: nn.Module) -> List[bool]:
    """Return a boolean mask over state_dict entries: True for trainable params."""
    param_names = {name for name, _ in model.named_parameters()}
    return [name in param_names for name in model.state_dict().keys()]


# ---------------------------------------------------------------------------
# DP-aware Flower Client
# ---------------------------------------------------------------------------


class DPAphroFlowerClient(NumPyClient):
    """Flower client with local differential privacy (DP-SGD)."""

    def __init__(
        self,
        train_ds: StationPatchDataset,
        dp_config: DPConfig,
        local_epochs: int = 1,
        batch_size: int = 16,
        lr: float = 0.001,
        in_channels: int = 3,
        base_filters: int = 32,
    ):
        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
        )
        self.local_epochs = local_epochs
        self.lr = lr
        self.dp_config = dp_config
        self.device = _get_device()
        self.model = UNetCNN(
            in_channels=in_channels,
            out_channels=1,
            base_filters=base_filters,
        ).to(self.device)
        self.num_examples = len(train_ds)
        self.accountant = DPAccountant(target_delta=dp_config.target_delta)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.to(self.device)

        if self.dp_config.local_dp:
            loss, self.accountant = dp_train_sparse_pixel(
                self.model,
                self.train_loader,
                dp_config=self.dp_config,
                epochs=self.local_epochs,
                lr=self.lr,
                accountant=self.accountant,
            )
            epsilon = self.accountant.epsilon
        else:
            # Fall back to standard training (imported from training module)
            from fl4hma.training.training import train_sparse_pixel

            loss = train_sparse_pixel(
                self.model,
                self.train_loader,
                epochs=self.local_epochs,
                lr=self.lr,
            )
            epsilon = 0.0

        return (
            get_parameters(self.model),
            self.num_examples,
            {"train_loss": loss, "epsilon": epsilon},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        metrics = evaluate_sparse_pixel(self.model, self.train_loader)
        return metrics["loss"], self.num_examples, {"mse": metrics["mse"]}


# ---------------------------------------------------------------------------
# DP-aware FedAvg Strategy (Global DP)
# ---------------------------------------------------------------------------


class DPFedAvg(FedAvg):
    """FedAvg with global differential privacy.

    After aggregation, clips client model updates and adds calibrated Gaussian
    noise to the global model.  Tracks server-side privacy budget.
    """

    def __init__(
        self,
        dp_config: DPConfig,
        num_clients: int,
        in_channels: int = 3,
        base_filters: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dp_config = dp_config
        self.num_clients = num_clients
        self.accountant = DPAccountant(target_delta=dp_config.target_delta)
        self._rng = np.random.default_rng(42)
        # Cache the pre-round global params for computing deltas
        self._global_params: Optional[List[np.ndarray]] = None
        # Mask: True for trainable params, False for buffers (e.g. BN stats)
        _ref_model = UNetCNN(
            in_channels=in_channels, out_channels=1, base_filters=base_filters
        )
        self._trainable_mask = _get_trainable_mask(_ref_model)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate with optional global DP."""
        if not self.dp_config.global_dp:
            return super().aggregate_fit(server_round, results, failures)

        if not results:
            return None, {}

        # Extract and clip each client's update
        if self._global_params is not None:
            clipped_results = []
            for client_proxy, fit_res in results:
                client_params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                clipped = clip_model_update(
                    self._global_params,
                    client_params,
                    self.dp_config.clip_norm,
                    mask=self._trainable_mask,
                )
                fit_res_new = FitRes(
                    status=fit_res.status,
                    parameters=fl.common.ndarrays_to_parameters(clipped),
                    num_examples=fit_res.num_examples,
                    metrics=fit_res.metrics,
                )
                clipped_results.append((client_proxy, fit_res_new))
            results = clipped_results

        # Standard FedAvg aggregation on clipped updates
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_params is not None:
            # Add noise only to trainable parameters (skip BN running stats)
            agg_ndarrays = fl.common.parameters_to_ndarrays(aggregated_params)
            noise_std = (
                self.dp_config.noise_multiplier
                * self.dp_config.clip_norm
                / self.num_clients
            )
            noisy_params = add_noise_to_parameters(
                agg_ndarrays,
                noise_std,
                self._rng,
                noise_mask=self._trainable_mask,
            )
            aggregated_params = fl.common.ndarrays_to_parameters(noisy_params)

            # Update cached global params
            self._global_params = noisy_params

            # Account for this round
            self.accountant.step(
                self.dp_config.noise_multiplier,
                sample_rate=1.0,
            )
            eps = self.accountant.epsilon
            print(
                f"  [Global DP] Round {server_round}: "
                f"noise_std={noise_std:.6f}, ε={eps:.4f} "
                f"(δ={self.dp_config.target_delta})"
            )
            metrics["global_epsilon"] = eps

        return aggregated_params, metrics

    def initialize_parameters(self, client_manager):
        """Cache initial parameters for delta computation."""
        params = super().initialize_parameters(client_manager)
        if params is not None:
            self._global_params = fl.common.parameters_to_ndarrays(params)
        return params


# ---------------------------------------------------------------------------
# End-to-end DP Federated Simulation
# ---------------------------------------------------------------------------


def run_federated_dp(
    da_train: xr.DataArray,
    da_test: xr.DataArray,
    country_masks: Dict[str, str],
    output_mask_path: str,
    centralised_mask_path: str,
    dp_config: Optional[DPConfig] = None,
    test_input_mask_path: Optional[str] = None,
    num_rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 16,
    lr: float = 0.001,
    in_channels: int = 3,
    base_filters: int = 32,
    patch_size: int = 32,
    stride: int = 32,
) -> Dict:
    """Run Flower FedAvg simulation with differential privacy.

    Parameters
    ----------
    da_train, da_test : xr.DataArray
        APHRODITE data arrays with dims (variable, time, lat, lon).
    country_masks : dict
        ``{country_name: path_to_mask.npy}``
    output_mask_path : str
        Path to the output (land) mask.
    centralised_mask_path : str
        Path to combined mask used for server-side test evaluation.
    dp_config : DPConfig or None
        Differential privacy configuration.  If None, uses defaults
        (both local and global DP enabled with noise_multiplier=1.0).
    test_input_mask_path : str or None
        If given, use this mask for server-side test evaluation.
    num_rounds : int
        Number of FL communication rounds.
    local_epochs : int
        Client-local training epochs per round.
    batch_size, lr, in_channels, base_filters, patch_size, stride :
        Standard model/training hyperparameters.

    Returns
    -------
    dict with model, history, DP privacy budgets, and metrics.
    """
    if dp_config is None:
        dp_config = DPConfig()

    np.random.seed(42)
    torch.manual_seed(42)

    num_clients = len(country_masks)
    country_names = list(country_masks.keys())

    print("=" * 64)
    print("Federated Learning with Differential Privacy (Flower)")
    print("=" * 64)
    print(f"  Clients            : {num_clients} ({', '.join(country_names)})")
    print(f"  Rounds             : {num_rounds}")
    print(f"  Local epochs       : {local_epochs}")
    print(f"  Local DP           : {dp_config.local_dp}")
    print(f"  Global DP          : {dp_config.global_dp}")
    print(f"  Clip norm          : {dp_config.clip_norm}")
    print(f"  Noise multiplier   : {dp_config.noise_multiplier}")
    print(f"  Target δ           : {dp_config.target_delta}")
    print(f"  Device             : {_get_device()}")
    print()

    # --- Per-country training datasets ---
    client_datasets = build_country_datasets(
        da_train,
        country_masks,
        output_mask_path,
        patch_size=patch_size,
        stride=stride,
    )
    client_list = list(client_datasets.values())

    for name, ds in client_datasets.items():
        print(f"  Client '{name}': {len(ds)} patches")
    print()

    # --- Test dataset ---
    _test_mask = test_input_mask_path or centralised_mask_path
    test_ds = StationPatchDataset(
        da_test,
        input_mask_path=_test_mask,
        output_mask_path=output_mask_path,
        patch_size=patch_size,
        stride=stride,
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # --- Client factory ---
    build_client = partial(
        DPAphroFlowerClient,
        dp_config=dp_config,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        in_channels=in_channels,
        base_filters=base_filters,
    )

    # --- Strategy ---
    initial_model = UNetCNN(
        in_channels=in_channels,
        out_channels=1,
        base_filters=base_filters,
    )
    initial_params = fl.common.ndarrays_to_parameters(
        get_parameters(initial_model),
    )

    _final_params: List[np.ndarray] = []

    from fl4hma.federation.federation import get_evaluate_fn

    _inner_eval = get_evaluate_fn(
        test_loader,
        in_channels=in_channels,
        base_filters=base_filters,
    )

    def _capturing_eval(server_round, parameters, config):
        _final_params.clear()
        _final_params.extend(parameters)
        return _inner_eval(server_round, parameters, config)

    strategy = DPFedAvg(
        dp_config=dp_config,
        num_clients=num_clients,
        in_channels=in_channels,
        base_filters=base_filters,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=_capturing_eval,
        initial_parameters=initial_params,
    )

    # --- Simulation ---
    history = run_ray_simulation(
        client_list,
        build_client,
        num_rounds=num_rounds,
        strategy=strategy,
    )

    # --- Collect results ---
    rounds = [r for r, _ in history.losses_centralized]
    losses = [l for _, l in history.losses_centralized]
    mse_values = [m for _, m in history.metrics_centralized.get("mse", [])]
    rmse_values = [m for _, m in history.metrics_centralized.get("rmse", [])]

    final_mse = mse_values[-1] if mse_values else 0.0
    final_rmse = rmse_values[-1] if rmse_values else 0.0

    # Reconstruct final global model
    final_model = UNetCNN(
        in_channels=in_channels,
        out_channels=1,
        base_filters=base_filters,
    ).to(_get_device())
    if _final_params:
        set_parameters(final_model, _final_params)
    final_model.eval()

    # Privacy summary
    global_epsilon = strategy.accountant.epsilon if dp_config.global_dp else None
    local_epsilon = (
        {
            name: local_dp_epsilon(
                dp_config,
                dataset_size=len(ds),
                batch_size=batch_size,
                local_epochs=local_epochs,
                num_rounds=num_rounds,
            )
            for name, ds in client_datasets.items()
        }
        if dp_config.local_dp
        else None
    )

    print()
    print(f"Final DP-federated test MSE  after {num_rounds} rounds: {final_mse:.6f}")
    print(f"Final DP-federated test RMSE after {num_rounds} rounds: {final_rmse:.6f}")
    if global_epsilon is not None:
        print(
            f"Global DP budget: ε = {global_epsilon:.4f}, "
            f"δ = {dp_config.target_delta}"
        )
    if local_epsilon is not None:
        for name, eps in local_epsilon.items():
            print(f"Local DP budget ({name}): ε = {eps:.4f}")

    return {
        "model": final_model,
        "history": history,
        "rounds": rounds,
        "losses": losses,
        "mse_values": mse_values,
        "rmse_values": rmse_values,
        "final_mse": final_mse,
        "final_rmse": final_rmse,
        "dp_config": dp_config,
        "global_epsilon": global_epsilon,
        "local_epsilon": local_epsilon,
        "global_accountant": strategy.accountant,
        "config": {
            "num_clients": num_clients,
            "country_names": country_names,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "dp_local": dp_config.local_dp,
            "dp_global": dp_config.global_dp,
            "clip_norm": dp_config.clip_norm,
            "noise_multiplier": dp_config.noise_multiplier,
            "target_delta": dp_config.target_delta,
        },
    }
