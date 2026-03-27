"""
Run the federated sparse CIFAR-10 example.

Usage:
    python run_federated_example.py

Requirements:
    pip install -e .[examples]
    pip install "flwr[simulation]>=1.5"
"""

import sys
import os

# Ensure src/ and repo root are on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))

try:
    from examples.federated_sparse_cifar10 import run_federated
except ImportError as exc:
    print(f"Import error: {exc}")
    print("Make sure you have installed the required packages:")
    print("  pip install torch torchvision 'flwr[simulation]>=1.5'")
    sys.exit(1)


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
    print(f"\nDone. Final accuracy: {results['final_accuracy']:.4f}")
