"""
Federated Learning prototype for sparse CIFAR-10 classification using Flower.

This demonstrates:
1. Partitioning CIFAR-10 across multiple simulated clients (non-IID optional)
2. Each client has sparse labels (only a fraction of its data is labeled)
3. Federated Averaging (FedAvg) aggregation strategy
4. Single-machine simulation via Flower's simulation engine

Builds on the existing sparse_cifar10_example.py pattern.
"""

import sys
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.client import NumPyClient
from flwr.common import Context
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Re-use model & dataset helpers from sparse_cifar10_example
# ---------------------------------------------------------------------------

class SparseCIFAR10Dataset(Dataset):
    """CIFAR-10 dataset with sparse labeling for a single client partition."""

    def __init__(self, subset: Subset, sparsity: float = 0.1, seed: int = 42):
        self.subset = subset
        self.sparsity = sparsity
        rng = np.random.RandomState(seed)
        self.sparse_mask = rng.random(len(subset)) < sparsity

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool]:
        image, label = self.subset[idx]
        is_labeled = bool(self.sparse_mask[idx])
        sparse_label = label if is_labeled else -1
        return image, sparse_label, is_labeled


class VanillaCNN(nn.Module):
    """Simple CNN for CIFAR-10 classification (same architecture as the example)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_device():
    """Detect device dynamically — works inside Ray workers where CUDA may be masked."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Load parameters into a model."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train_on_client(
    model: nn.Module,
    loader: DataLoader,
    epochs: int = 1,
    lr: float = 0.001,
) -> float:
    """Local training loop used by each FL client."""
    device = _get_device()
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    for _ in range(epochs):
        for data, targets, is_labeled in loader:
            data, targets = data.to(device), targets.to(device)
            is_labeled = is_labeled.to(device)
            mask = is_labeled & (targets != -1)
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            loss = criterion(model(data[mask]), targets[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


def evaluate(
    model: nn.Module, loader: DataLoader
) -> Tuple[float, int, int]:
    """Evaluate model accuracy (used on full test set)."""
    device = _get_device()
    model.to(device)
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, targets, _ in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss_sum += criterion(outputs, targets).item() * targets.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
    return loss_sum / total, correct, total


# ---------------------------------------------------------------------------
# Data partitioning
# ---------------------------------------------------------------------------

def partition_data(
    num_clients: int,
    sparsity: float = 0.1,
    data_dir: str = "./data",
    iid: bool = True,
    alpha: float = 0.5,
) -> Tuple[List[SparseCIFAR10Dataset], SparseCIFAR10Dataset]:
    """Split CIFAR-10 training set across *num_clients* simulated clients.

    Parameters
    ----------
    num_clients : int
        Number of FL clients.
    sparsity : float
        Fraction of labels each client can see.
    data_dir : str
        Path to download / cache CIFAR-10.
    iid : bool
        If True, split uniformly at random.  If False, use a Dirichlet
        distribution controlled by *alpha* to create non-IID splits.
    alpha : float
        Concentration parameter for the Dirichlet non-IID split.
        Lower values = more heterogeneous.

    Returns
    -------
    client_datasets : list[SparseCIFAR10Dataset]
    test_dataset : SparseCIFAR10Dataset  (fully labelled)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform,
    )
    full_test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform,
    )

    n = len(full_train)

    if iid:
        indices = np.random.permutation(n)
        splits = np.array_split(indices, num_clients)
    else:
        # Non-IID Dirichlet partitioning
        targets = np.array(full_train.targets)
        num_classes = 10
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            class_idx = np.where(targets == c)[0]
            np.random.shuffle(class_idx)
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(class_idx)).astype(int)
            # fix rounding
            proportions[-1] = len(class_idx) - proportions[:-1].sum()
            start = 0
            for client_id in range(num_clients):
                end = start + proportions[client_id]
                client_indices[client_id].extend(class_idx[start:end].tolist())
                start = end
        splits = [np.array(idx) for idx in client_indices]

    client_datasets = []
    for cid, idx in enumerate(splits):
        subset = Subset(full_train, idx.tolist())
        ds = SparseCIFAR10Dataset(subset, sparsity=sparsity, seed=42 + cid)
        client_datasets.append(ds)

    # Test set – fully labeled
    test_subset = Subset(full_test, list(range(len(full_test))))
    test_dataset = SparseCIFAR10Dataset(test_subset, sparsity=1.0, seed=0)

    return client_datasets, test_dataset


# ---------------------------------------------------------------------------
# Flower NumPyClient
# ---------------------------------------------------------------------------

class CifarFlowerClient(NumPyClient):
    """Flower client wrapping local sparse CIFAR-10 training."""

    def __init__(
        self,
        train_ds: SparseCIFAR10Dataset,
        local_epochs: int = 1,
        batch_size: int = 32,
        lr: float = 0.001,
    ):
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
        )
        self.local_epochs = local_epochs
        self.lr = lr
        self.device = _get_device()
        self.model = VanillaCNN().to(self.device)
        self.num_examples = len(train_ds)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        loss = train_on_client(
            self.model, self.train_loader,
            epochs=self.local_epochs, lr=self.lr,
        )
        return get_parameters(self.model), self.num_examples, {"train_loss": loss}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.to(self.device)
        loss, correct, total = evaluate(self.model, self.train_loader)
        return loss, total, {"accuracy": correct / total}


# ---------------------------------------------------------------------------
# Server-side evaluation callback
# ---------------------------------------------------------------------------

def get_evaluate_fn(test_dataset: SparseCIFAR10Dataset, batch_size: int = 64):
    """Return a server-side evaluation function (centralised test set)."""
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def evaluate_fn(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model = VanillaCNN().to(_get_device())
        set_parameters(model, parameters)
        loss, correct, total = evaluate(model, test_loader)
        accuracy = correct / total
        print(
            f"  [Server] Round {server_round}: "
            f"test_loss={loss:.4f}, test_acc={accuracy:.4f}"
        )
        return loss, {"accuracy": accuracy}

    return evaluate_fn


# ---------------------------------------------------------------------------
# Main entry point – run FL simulation
# ---------------------------------------------------------------------------

def run_federated(
    num_clients: int = 3,
    num_rounds: int = 5,
    local_epochs: int = 2,
    sparsity: float = 0.1,
    batch_size: int = 32,
    lr: float = 0.001,
    iid: bool = True,
    alpha: float = 0.5,
    fraction_fit: float = 1.0,
    data_dir: str = "./data",
) -> Dict:
    """Run a single-machine FL simulation and return history.

    Parameters
    ----------
    num_clients : int
        Number of simulated FL clients.
    num_rounds : int
        Number of FL communication rounds.
    local_epochs : int
        Epochs of local training per round per client.
    sparsity : float
        Fraction of labels available on each client.
    batch_size : int
        Batch size for local training.
    lr : float
        Learning rate.
    iid : bool
        IID vs non-IID data split.
    alpha : float
        Dirichlet alpha for non-IID (only used when iid=False).
    fraction_fit : float
        Fraction of clients sampled per round.
    data_dir : str
        CIFAR-10 download directory.

    Returns
    -------
    dict with keys ``history``, ``final_accuracy``, ``config``.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 64)
    print("Federated Learning – Sparse CIFAR-10 (Flower Simulation)")
    print("=" * 64)
    print(f"  Clients       : {num_clients}")
    print(f"  Rounds        : {num_rounds}")
    print(f"  Local epochs  : {local_epochs}")
    print(f"  Sparsity      : {sparsity*100:.0f}% labeled")
    print(f"  IID           : {iid}")
    if not iid:
        print(f"  Dirichlet α   : {alpha}")
    print(f"  Device        : {DEVICE}")
    print()

    # ---- data ----------------------------------------------------------
    client_datasets, test_dataset = partition_data(
        num_clients=num_clients,
        sparsity=sparsity,
        data_dir=data_dir,
        iid=iid,
        alpha=alpha,
    )
    for i, ds in enumerate(client_datasets):
        labeled = int(ds.sparse_mask.sum())
        print(f"  Client {i}: {len(ds)} samples, {labeled} labeled")
    print()

    # ---- client factory ------------------------------------------------
    def client_fn(context: Context):
        cid = int(context.node_config["partition-id"])
        return CifarFlowerClient(
            train_ds=client_datasets[cid],
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
        ).to_client()

    # ---- strategy ------------------------------------------------------
    initial_model = VanillaCNN()
    initial_params = fl.common.ndarrays_to_parameters(get_parameters(initial_model))

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,  # disable client-side eval to speed things up
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_fn=get_evaluate_fn(test_dataset, batch_size=64),
        initial_parameters=initial_params,
    )

    # ---- simulation ----------------------------------------------------
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

    # ---- collect results -----------------------------------------------
    # history.losses_centralized is List[(round, loss)]
    # history.metrics_centralized["accuracy"] is List[(round, acc)]
    rounds = [r for r, _ in history.losses_centralized]
    losses = [l for _, l in history.losses_centralized]
    accs = [a for _, a in history.metrics_centralized.get("accuracy", [])]

    final_acc = accs[-1] if accs else 0.0
    print()
    print(f"Final test accuracy after {num_rounds} rounds: {final_acc:.4f}")

    return {
        "history": history,
        "rounds": rounds,
        "losses": losses,
        "accuracies": accs,
        "final_accuracy": final_acc,
        "config": {
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "local_epochs": local_epochs,
            "sparsity": sparsity,
            "iid": iid,
            "alpha": alpha,
        },
    }


if __name__ == "__main__":
    results = run_federated(
        num_clients=3,
        num_rounds=5,
        local_epochs=2,
        sparsity=0.1,
        batch_size=32,
        iid=True,
        data_dir="./data",
    )
