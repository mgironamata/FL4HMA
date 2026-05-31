"""Federated learning client, strategy, and simulation orchestration."""

from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import xarray as xr
from flwr.client import NumPyClient
from flwr.common import Context
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from torch.utils.data import DataLoader

from fl4hma.data.data import build_country_datasets
from fl4hma.data.torch_dataset import StationPatchDataset
from fl4hma.models.unet import UNetCNN
from fl4hma.training.training import (
    _get_device,
    evaluate_sparse_pixel,
    get_parameters,
    set_parameters,
    train_sparse_pixel,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AphroFlowerClient(NumPyClient):
    """Flower client for sparse pixel regression on APHRODITE data."""

    def __init__(
        self,
        train_ds: StationPatchDataset,
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
        self.device = _get_device()
        self.model = UNetCNN(
            in_channels=in_channels,
            out_channels=1,
            base_filters=base_filters,
        ).to(self.device)
        self.num_examples = len(train_ds)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        loss = train_sparse_pixel(
            self.model,
            self.train_loader,
            epochs=self.local_epochs,
            lr=self.lr,
        )
        return get_parameters(self.model), self.num_examples, {"train_loss": loss}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        metrics = evaluate_sparse_pixel(self.model, self.train_loader)
        return metrics["loss"], self.num_examples, {"mse": metrics["mse"]}


def get_evaluate_fn(
    test_loader: DataLoader,
    in_channels: int = 3,
    base_filters: int = 32,
):
    """Return a server-side evaluation function for centralised test set."""

    def evaluate_fn(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        device = _get_device()
        model = UNetCNN(
            in_channels=in_channels,
            out_channels=1,
            base_filters=base_filters,
        ).to(device)
        set_parameters(model, parameters)
        metrics = evaluate_sparse_pixel(model, test_loader)
        print(
            f"  [Server] Round {server_round}: "
            f"loss={metrics['loss']:.4f}, mse={metrics['mse']:.6f}, "
            f"rmse={metrics['rmse']:.6f}"
        )
        return metrics["loss"], {"mse": metrics["mse"], "rmse": metrics["rmse"]}

    return evaluate_fn


def run_centralised(
    da_train: xr.DataArray,
    da_test: xr.DataArray,
    input_mask_path: str,
    output_mask_path: str,
    num_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 0.001,
    in_channels: int = 3,
    base_filters: int = 32,
    patch_size: int = 32,
    stride: int = 32,
) -> Dict:
    """Train a single model on all data (centralised baseline).

    Returns dict with ``model``, ``train_losses``, ``test_metrics``.
    """
    device = _get_device()

    train_ds = StationPatchDataset(
        da_train,
        input_mask_path=input_mask_path,
        output_mask_path=output_mask_path,
        patch_size=patch_size,
        stride=stride,
    )
    test_ds = StationPatchDataset(
        da_test,
        input_mask_path=input_mask_path,
        output_mask_path=output_mask_path,
        patch_size=patch_size,
        stride=stride,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = UNetCNN(
        in_channels=in_channels,
        out_channels=1,
        base_filters=base_filters,
    ).to(device)

    print("=" * 60)
    print("Centralised Training")
    print("=" * 60)
    print(f"  Train samples : {len(train_ds)}")
    print(f"  Test samples  : {len(test_ds)}")
    print(f"  Epochs        : {num_epochs}")
    print(f"  Device        : {device}")
    print()

    train_losses = []
    test_metrics_history = []

    for epoch in range(num_epochs):
        loss = train_sparse_pixel(model, train_loader, epochs=1, lr=lr)
        test_met = evaluate_sparse_pixel(model, test_loader)
        train_losses.append(loss)
        test_metrics_history.append(test_met)
        print(
            f"  Epoch {epoch + 1}/{num_epochs}: "
            f"train_loss={loss:.4f}, test_mse={test_met['mse']:.6f}, "
            f"test_rmse={test_met['rmse']:.6f}"
        )

    final_test = evaluate_sparse_pixel(model, test_loader)
    print(f"\nFinal centralised test MSE : {final_test['mse']:.6f}")
    print(f"Final centralised test RMSE: {final_test['rmse']:.6f}")

    return {
        "model": model,
        "train_losses": train_losses,
        "test_metrics_history": test_metrics_history,
        "final_test_metrics": final_test,
    }


def run_federated(
    da_train: xr.DataArray,
    da_test: xr.DataArray,
    country_masks: Dict[str, str],
    output_mask_path: str,
    centralised_mask_path: str,
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
    """Run Flower FedAvg simulation with per-country clients.

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
    test_input_mask_path : str or None
        If given, use this mask for server-side test evaluation instead of
        *centralised_mask_path*.  Useful when the test-time input mask
        should differ from the training union mask.
    num_rounds : int
        Number of FL communication rounds.
    local_epochs : int
        Client-local training epochs per round.

    Returns
    -------
    dict with ``model``, ``history``, ``rounds``, ``losses``, ``mse_values``, ``config``.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    num_clients = len(country_masks)
    country_names = list(country_masks.keys())

    print("=" * 64)
    print("Federated Learning – Sparse Pixel APHRODITE (Flower)")
    print("=" * 64)
    print(f"  Clients       : {num_clients} ({', '.join(country_names)})")
    print(f"  Rounds        : {num_rounds}")
    print(f"  Local epochs  : {local_epochs}")
    print(f"  Device        : {DEVICE}")
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

    # --- Test dataset (use explicit test mask or fall back to centralised) ---
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
        return AphroFlowerClient(
            train_ds=client_list[cid],
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

    # Container to capture the final global parameters
    _final_params: List[np.ndarray] = []

    _inner_eval = get_evaluate_fn(
        test_loader,
        in_channels=in_channels,
        base_filters=base_filters,
    )

    def _capturing_eval(server_round, parameters, config):
        _final_params.clear()
        _final_params.extend(parameters)
        return _inner_eval(server_round, parameters, config)

    strategy = FedAvg(
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
    print()
    print(f"Final federated test MSE  after {num_rounds} rounds: {final_mse:.6f}")
    print(f"Final federated test RMSE after {num_rounds} rounds: {final_rmse:.6f}")

    # Reconstruct final global model from captured parameters
    final_model = UNetCNN(
        in_channels=in_channels,
        out_channels=1,
        base_filters=base_filters,
    ).to(_get_device())
    if _final_params:
        set_parameters(final_model, _final_params)
    final_model.eval()

    return {
        "model": final_model,
        "history": history,
        "rounds": rounds,
        "losses": losses,
        "mse_values": mse_values,
        "rmse_values": rmse_values,
        "final_mse": final_mse,
        "final_rmse": final_rmse,
        "config": {
            "num_clients": num_clients,
            "country_names": country_names,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
        },
    }
