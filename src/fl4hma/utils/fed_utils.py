import json
import os
import urllib.request
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.client import NumPyClient
from flwr.common import Context
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from matplotlib.path import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import xarray as xr

from fl4hma.models.unet import UNetCNN, sparse_pixel_loss
from fl4hma.data.dataset_v2 import StationPatchDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)  
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True) 


def load_aphro_data(
    train_path: str,
    test_path: str,
    variable: str = "precip",
    lon_slice: Tuple[float, float] = (60, 105),
    lat_slice: Tuple[float, float] = (20, 40),
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Load APHRODITE data for any variable and return (da_train, da_test).

    Parameters
    ----------
    variable : str
        NetCDF variable name, e.g. ``"precip"`` or ``"tave"``.
    """
    ds_train = xr.open_dataset(train_path).sel(
        lon=slice(*lon_slice), lat=slice(*lat_slice),
    )
    ds_test = xr.open_dataset(test_path).sel(
        lon=slice(*lon_slice), lat=slice(*lat_slice),
    )

    # Add lat/lon coordinate channels
    for ds in (ds_train, ds_test):
        ds["lats"] = (
            ("time", "lat", "lon"),
            np.tile(ds.lat.values, (ds.time.size, ds.lon.size, 1)).transpose(0, 2, 1),
        )
        ds["lons"] = (
            ("time", "lat", "lon"),
            np.tile(ds.lon.values, (ds.time.size, ds.lat.size, 1)),
        )

    da_train = ds_train[[variable, "lats", "lons"]].to_array().fillna(0)
    da_test = ds_test[[variable, "lats", "lons"]].to_array().fillna(0)
    return da_train, da_test

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

def generate_random_split_masks(
    centralised_mask: np.ndarray,
    n_clients: int,
    out_dir: str,
    seed: int = 42,
    prefix: str = "random",
) -> Dict[str, str]:
    """Randomly partition station pixels into *n_clients* disjoint masks.

    Parameters
    ----------
    centralised_mask : 2-D bool/int array
    n_clients : number of disjoint groups
    out_dir : output directory
    seed : random seed for reproducibility
    prefix : filename prefix

    Returns
    -------
    {client_name: path_to_mask.npy}  e.g. {"client_0": "...", ...}
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    positions = np.argwhere(centralised_mask.astype(bool))
    n_stations = len(positions)
    perm = rng.permutation(n_stations)
    splits = np.array_split(perm, n_clients)

    mask_paths: Dict[str, str] = {}
    for i, indices in enumerate(splits):
        m = np.zeros_like(centralised_mask, dtype=int)
        rows, cols = positions[indices, 0], positions[indices, 1]
        m[rows, cols] = 1
        name = f"client_{i}"
        path = os.path.join(out_dir, f"{prefix}_{name}_mask.npy")
        np.save(path, m)
        mask_paths[name] = path

    # Verify
    union = np.zeros_like(centralised_mask, dtype=int)
    for p in mask_paths.values():
        union += np.load(p)
    assert np.array_equal(union, centralised_mask.astype(int)), \
        "Random split union does not equal centralised mask"

    return mask_paths


# ---------------------------------------------------------------------------
# Country name → Natural Earth "name" field mapping
# ---------------------------------------------------------------------------
_COUNTRY_NAME_MAP = {
    "afghanistan": "Afghanistan",
    "china": "China",
    "india": "India",
    "kazakhstan": "Kazakhstan",
    "kyrgyzstan": "Kyrgyzstan",
    "nepal": "Nepal",
    "pakistan": "Pakistan",
    "tajikistan": "Tajikistan",
    "uzbekistan": "Uzbekistan",
}

_NE_GEOJSON_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_50m_admin_0_countries.geojson"
)


def _download_natural_earth(cache_path: str) -> dict:
    """Download Natural Earth 50m countries GeoJSON (cached on disk)."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    print(f"  Downloading Natural Earth GeoJSON → {cache_path}")
    urllib.request.urlretrieve(_NE_GEOJSON_URL, cache_path)
    with open(cache_path) as f:
        return json.load(f)


def _polygon_coords_to_paths(geometry: dict) -> List[Path]:
    """Convert a GeoJSON Polygon/MultiPolygon to matplotlib Paths."""
    paths = []
    if geometry["type"] == "Polygon":
        paths.append(Path(geometry["coordinates"][0]))
    elif geometry["type"] == "MultiPolygon":
        for poly in geometry["coordinates"]:
            paths.append(Path(poly[0]))
    return paths


def generate_country_boundary_masks(
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    countries: List[str],
    out_dir: str,
    land_mask: Optional[np.ndarray] = None,
    geojson_cache: str = "data/ne_50m_admin_0_countries.geojson",
    name_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Rasterise country polygons onto a lat/lon grid.

    Uses Natural Earth 50m boundaries (downloaded once and cached).
    Point-in-polygon is done via ``matplotlib.path.Path``.

    Parameters
    ----------
    lat_vals, lon_vals : 1-D arrays of grid coordinates.
    countries : list of lowercase country names (keys in *name_map*).
    out_dir : directory where ``<country>_boundary_mask.npy`` is saved.
    land_mask : optional 2-D array; boundary masks are intersected with it.
    geojson_cache : path to cache the downloaded GeoJSON file.
    name_map : {lowercase_name: Natural_Earth_name}. Defaults to HMA set.

    Returns
    -------
    {country_name: path_to_boundary_mask.npy}
    """
    if name_map is None:
        name_map = _COUNTRY_NAME_MAP

    os.makedirs(out_dir, exist_ok=True)

    # Download / load Natural Earth
    geojson = _download_natural_earth(geojson_cache)

    # Build lookup: NE name → geometry
    ne_geom = {}
    for feat in geojson["features"]:
        ne_geom[feat["properties"]["NAME"]] = feat["geometry"]

    # Build grid points (lon, lat) — shape (N, 2) for Path.contains_points
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    mask_paths: Dict[str, str] = {}
    for country in countries:
        path_file = os.path.join(out_dir, f"{country}_boundary_mask.npy")
        if os.path.exists(path_file):
            print(f"    {country:15s}: cached ({int(np.load(path_file).sum()):5d} pixels)")
            mask_paths[country] = path_file
            continue

        ne_name = name_map[country]
        if ne_name not in ne_geom:
            raise ValueError(
                f"Country '{ne_name}' not found in Natural Earth data. "
                f"Available: {sorted(ne_geom.keys())}"
            )

        paths = _polygon_coords_to_paths(ne_geom[ne_name])
        inside = np.zeros(grid_points.shape[0], dtype=bool)
        for p in paths:
            inside |= p.contains_points(grid_points)
        mask = inside.reshape(lon_grid.shape).astype(int)

        if land_mask is not None:
            mask = mask & land_mask.astype(int)

        np.save(path_file, mask)
        print(f"    {country:15s}: {int(mask.sum()):5d} pixels")
        mask_paths[country] = path_file

    return mask_paths


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
                            pred[b][:, mask_b], sparse_tgt[b][:, mask_b],
                        )
                        total_mse += mse_b.item()
                n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_mse = total_mse / max(1, n_batches)
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
            train_ds, batch_size=batch_size, shuffle=True,
        )
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = _get_device()
        self.model = UNetCNN(
            in_channels=in_channels, out_channels=1, base_filters=base_filters,
        ).to(self.device)
        self.num_examples = len(train_ds)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        loss = train_sparse_pixel(
            self.model, self.train_loader,
            epochs=self.local_epochs, lr=self.lr,
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
            in_channels=in_channels, out_channels=1, base_filters=base_filters,
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
        in_channels=in_channels, out_channels=1, base_filters=base_filters,
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
    dict with ``history``, ``rounds``, ``losses``, ``mse_values``, ``config``.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    num_clients = len(country_masks)
    country_names = list(country_masks.keys())

    print("=" * 64)
    print("Federated Learning – Sparse Pixel APHRODITE Precip (Flower)")
    print("=" * 64)
    print(f"  Clients       : {num_clients} ({', '.join(country_names)})")
    print(f"  Rounds        : {num_rounds}")
    print(f"  Local epochs  : {local_epochs}")
    print(f"  Device        : {DEVICE}")
    print()

    # --- Per-country training datasets ---
    client_datasets = build_country_datasets(
        da_train, country_masks, output_mask_path,
        patch_size=patch_size, stride=stride,
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
        in_channels=in_channels, out_channels=1, base_filters=base_filters,
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
    mse_values = [
        m for _, m in history.metrics_centralized.get("mse", [])
    ]
    rmse_values = [
        m for _, m in history.metrics_centralized.get("rmse", [])
    ]

    final_mse = mse_values[-1] if mse_values else 0.0
    final_rmse = rmse_values[-1] if rmse_values else 0.0
    print()
    print(f"Final federated test MSE  after {num_rounds} rounds: {final_mse:.6f}")
    print(f"Final federated test RMSE after {num_rounds} rounds: {final_rmse:.6f}")

    # Reconstruct final global model from captured parameters
    final_model = UNetCNN(
        in_channels=in_channels, out_channels=1, base_filters=base_filters,
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


