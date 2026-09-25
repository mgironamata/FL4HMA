"""Grid-search hyperparameter tuning for the federated (FedAvg) UNet model.

Usage
-----
    python run_hp_tuning_federated.py configs/hp_tuning_federated_tave.yaml

The script generates the full Cartesian product of the search_space defined in
the YAML config, runs a fresh federated simulation (per-country clients) for
each combination, records the final test MSE reported by the server-side
evaluation, and writes a CSV of results plus a JSON summary of the best trial.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import sys
import time
from typing import Any, Dict, List

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fl4hma.data.data_loading import load_aphro_data
from fl4hma.federation.federation import run_federated


def load_config(path: str) -> dict:
    """Load a YAML tuning config from disk."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_grid(search_space: dict) -> List[Dict[str, Any]]:
    """Return the full Cartesian product of ``search_space`` as trial dicts.

    ``patch_size_stride_pairs`` is treated as a coupled axis: each element is a
    ``[patch_size, stride]`` pair (stride must be ``<= patch_size``). If the key
    is absent, a single ``[32, 32]`` pair is assumed.
    """
    search_space = copy.deepcopy(search_space)
    patch_stride_pairs: List[List[int]] = search_space.pop(
        "patch_size_stride_pairs", [[32, 32]]
    )

    axes_keys = list(search_space.keys())
    axes_vals = [search_space[k] for k in axes_keys]

    grid: List[Dict[str, Any]] = []
    for patch_size, stride in patch_stride_pairs:
        for combo in itertools.product(*axes_vals):
            trial = dict(zip(axes_keys, combo))
            trial["patch_size"] = patch_size
            trial["stride"] = stride
            grid.append(trial)

    return grid


CSV_FIELDS = [
    "trial",
    "lr",
    "local_epochs",
    "num_rounds",
    "batch_size",
    "base_filters",
    "patch_size",
    "stride",
    "val_mse",
    "val_rmse",
    "val_loss",
    "elapsed_s",
    "status",
]


def _init_csv(csv_path: str) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()


def _append_row(csv_path: str, row: dict) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writerow(row)


def main(config_path: str) -> None:
    """Run the federated grid search defined by the given YAML config."""
    cfg = load_config(config_path)

    experiment_name = cfg.get("experiment_name", "hp_tuning_federated")
    out_cfg = cfg["output"]
    results_dir = out_cfg["results_dir"]
    csv_path = out_cfg["csv_file"]
    os.makedirs(results_dir, exist_ok=True)

    # Load data once — reused across all trials
    data_cfg = cfg["data"]
    print("Loading data …")
    da_train, da_test = load_aphro_data(
        train_path=data_cfg["train_path"],
        test_path=data_cfg["test_path"],
        variable=data_cfg["variable"],
        lon_slice=tuple(data_cfg["lon_slice"]),
        lat_slice=tuple(data_cfg["lat_slice"]),
        elevation_path=data_cfg.get("elevation_path"),
    )
    print(f"  train={da_train.shape}  test={da_test.shape}\n")

    mask_cfg = cfg["masks"]
    country_masks: Dict[str, str] = mask_cfg["country_masks"]
    output_mask_path = mask_cfg["output_mask_path"]
    centralised_mask_path = mask_cfg["centralised_mask_path"]
    test_input_mask_path = mask_cfg.get("test_input_mask_path")
    in_channels = cfg["model"]["in_channels"]

    grid = build_grid(cfg["search_space"])
    n_trials = len(grid)

    print(f"{'=' * 70}")
    print(f"Experiment : {experiment_name}")
    print(f"Clients    : {len(country_masks)} ({', '.join(country_masks.keys())})")
    print(f"Total trials: {n_trials}")
    print(f"CSV output : {csv_path}")
    print(f"{'=' * 70}\n")

    _init_csv(csv_path)

    best_mse = float("inf")
    best_trial: Dict[str, Any] = {}
    best_idx = -1

    for trial_idx, hp in enumerate(grid, start=1):
        lr = hp["lr"]
        local_epochs = int(hp["local_epochs"])
        num_rounds = int(hp["num_rounds"])
        batch_size = int(hp["batch_size"])
        base_filters = int(hp["base_filters"])
        patch_size = int(hp["patch_size"])
        stride = int(hp["stride"])

        print(f"\n[Trial {trial_idx}/{n_trials}]")
        print(
            f"  lr={lr}, local_epochs={local_epochs}, num_rounds={num_rounds}, "
            f"batch_size={batch_size}, base_filters={base_filters}, "
            f"patch_size={patch_size}, stride={stride}"
        )

        row: Dict[str, Any] = {
            "trial": trial_idx,
            "lr": lr,
            "local_epochs": local_epochs,
            "num_rounds": num_rounds,
            "batch_size": batch_size,
            "base_filters": base_filters,
            "patch_size": patch_size,
            "stride": stride,
        }

        t0 = time.time()
        try:
            results = run_federated(
                da_train=da_train,
                da_test=da_test,
                country_masks=country_masks,
                output_mask_path=output_mask_path,
                centralised_mask_path=centralised_mask_path,
                test_input_mask_path=test_input_mask_path,
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                in_channels=in_channels,
                base_filters=base_filters,
                patch_size=patch_size,
                stride=stride,
            )
            elapsed = time.time() - t0
            val_mse = results["final_mse"]
            val_rmse = results["final_rmse"]
            val_loss = results["losses"][-1] if results["losses"] else float("nan")

            row.update(
                {
                    "val_mse": val_mse,
                    "val_rmse": val_rmse,
                    "val_loss": val_loss,
                    "elapsed_s": round(elapsed, 1),
                    "status": "ok",
                }
            )

            print(
                f"  -> val_mse={val_mse:.6f}  val_rmse={val_rmse:.6f}  "
                f"elapsed={elapsed:.0f}s"
            )

            if val_mse < best_mse:
                best_mse = val_mse
                best_trial = dict(hp)
                best_idx = trial_idx
                print(f"  ** New best (trial {trial_idx}): val_mse={best_mse:.6f}")

        except Exception as exc:
            elapsed = time.time() - t0
            row.update(
                {
                    "val_mse": float("nan"),
                    "val_rmse": float("nan"),
                    "val_loss": float("nan"),
                    "elapsed_s": round(elapsed, 1),
                    "status": f"error: {exc}",
                }
            )
            print(f"  -> FAILED: {exc}")

        _append_row(csv_path, row)

    print(f"\n{'=' * 70}")
    print(f"Grid search complete.  {n_trials} trials.")
    if best_idx >= 0:
        print(f"Best trial : {best_idx}  (val_mse={best_mse:.6f})")
        print("Best hyperparameters:")
        for k, v in best_trial.items():
            print(f"  {k}: {v}")
    print(f"Full results saved to: {csv_path}")

    best_json_path = os.path.join(results_dir, "best_config.json")
    with open(best_json_path, "w") as f:
        json.dump(
            {
                "best_trial": best_idx,
                "val_mse": best_mse,
                "hyperparameters": best_trial,
            },
            f,
            indent=2,
        )
    print(f"Best config saved to : {best_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid-search hyperparameter tuning for the federated UNet."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/hp_tuning_federated_tave.yaml",
        help=(
            "Path to the YAML tuning config "
            "(default: configs/hp_tuning_federated_tave.yaml)"
        ),
    )
    args = parser.parse_args()
    main(args.config)
