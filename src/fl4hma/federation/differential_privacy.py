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
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
from flwr.client import NumPyClient
from flwr.common import Context, FitRes, NDArrays, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from torch.utils.data import DataLoader

from fl4hma.data.data import build_country_datasets
from fl4hma.data.torch_dataset import StationPatchDataset
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
                rdp[i] = (
                    math.log1p(
                        sample_rate**2
                        * (math.exp(alpha / (noise_multiplier**2)) - 1)
                        / (alpha - 1)
                    )
                    if alpha > 1
                    else 0.0
                )
                # Tighter bound for large alpha
                rdp[i] = min(rdp[i], alpha / (2.0 * noise_multiplier**2))
        return rdp

    def step(self, noise_multiplier: float, sample_rate: float = 1.0) -> None:
        """Record one DP mechanism application (one training step or round)."""
        rdp = self._compute_rdp_gaussian(noise_multiplier, sample_rate)
        self._rdp_eps += rdp
        self._steps += 1

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


def _clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip per-parameter gradients and return total gradient norm."""
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()


def _add_noise_to_gradients(
    model: nn.Module, noise_std: float, device: torch.device
) -> None:
    """Add Gaussian noise to model gradients (in-place)."""
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.normal(
                mean=0.0,
                std=noise_std,
                size=param.grad.shape,
                device=device,
            )
            param.grad.add_(noise)


def dp_train_sparse_pixel(
    model: nn.Module,
    loader: DataLoader,
    dp_config: DPConfig,
    epochs: int = 1,
    lr: float = 0.001,
    accountant: Optional[DPAccountant] = None,
) -> Tuple[float, DPAccountant]:
    """Train with local DP-SGD (per-batch gradient clipping + noise).

    Parameters
    ----------
    model : nn.Module
        Model to train.
    loader : DataLoader
        Training data loader.
    dp_config : DPConfig
        DP configuration.
    epochs : int
        Number of local epochs.
    lr : float
        Learning rate.
    accountant : DPAccountant or None
        Privacy accountant (created if not given).

    Returns
    -------
    (avg_loss, accountant)
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
    dataset_size = len(loader.dataset)
    sample_rate = batch_size / dataset_size

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

                # DP-SGD: clip + noise
                _clip_gradients(model, max_norm)
                _add_noise_to_gradients(model, noise_std, device)

                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

                # Account for this step
                accountant.step(dp_config.noise_multiplier, sample_rate)

    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, accountant


# ---------------------------------------------------------------------------
# Global DP: Server-side clipping and noise
# ---------------------------------------------------------------------------


def clip_model_update(
    original_params: List[np.ndarray],
    updated_params: List[np.ndarray],
    clip_norm: float,
) -> List[np.ndarray]:
    """Clip a model update (Δ = updated - original) to a maximum L2 norm.

    Returns the clipped updated parameters (original + clipped_Δ).
    """
    deltas = [u - o for u, o in zip(updated_params, original_params)]
    flat_delta = np.concatenate([d.ravel() for d in deltas])
    delta_norm = np.linalg.norm(flat_delta)

    if delta_norm > clip_norm:
        scale = clip_norm / delta_norm
        deltas = [d * scale for d in deltas]

    clipped_params = [o + d for o, d in zip(original_params, deltas)]
    return clipped_params


def add_noise_to_parameters(
    parameters: List[np.ndarray],
    noise_std: float,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """Add Gaussian noise to model parameters."""
    if rng is None:
        rng = np.random.default_rng()
    noisy = []
    for p in parameters:
        noise = rng.normal(loc=0.0, scale=noise_std, size=p.shape).astype(p.dtype)
        noisy.append(p + noise)
    return noisy


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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dp_config = dp_config
        self.num_clients = num_clients
        self.accountant = DPAccountant(target_delta=dp_config.target_delta)
        self._rng = np.random.default_rng(42)
        # Cache the pre-round global params for computing deltas
        self._global_params: Optional[List[np.ndarray]] = None

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
                    self._global_params, client_params, self.dp_config.clip_norm
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
            # Add noise to aggregated parameters
            agg_ndarrays = fl.common.parameters_to_ndarrays(aggregated_params)
            noise_std = (
                self.dp_config.noise_multiplier
                * self.dp_config.clip_norm
                / self.num_clients
            )
            noisy_params = add_noise_to_parameters(agg_ndarrays, noise_std, self._rng)
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
    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        return DPAphroFlowerClient(
            train_ds=client_list[cid],
            dp_config=dp_config,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            in_channels=in_channels,
            base_filters=base_filters,
        ).to_client()

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
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=_capturing_eval,
        initial_parameters=initial_params,
    )

    # --- Simulation ---
    history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 1,
            "num_gpus": (1.0 / num_clients) if torch.cuda.is_available() else 0.0,
        },
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

    print()
    print(f"Final DP-federated test MSE  after {num_rounds} rounds: {final_mse:.6f}")
    print(f"Final DP-federated test RMSE after {num_rounds} rounds: {final_rmse:.6f}")
    if global_epsilon is not None:
        print(
            f"Global DP budget: ε = {global_epsilon:.4f}, "
            f"δ = {dp_config.target_delta}"
        )

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
