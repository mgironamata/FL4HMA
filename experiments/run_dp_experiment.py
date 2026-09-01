"""Run a single DP experiment from a YAML config file.

Usage:
    python run_dp_experiment.py configs/dp_baseline_no_dp.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch

from fl4hma.data.data_loading import load_aphro_data
from fl4hma.federation.differential_privacy import DPConfig, run_federated_dp
from fl4hma.federation.federation import run_federated
from fl4hma.training.training import get_parameters


def load_config(config_path: str) -> dict:
    """Load a YAML experiment config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(cfg: dict) -> dict:
    """Execute a single experiment defined by a config dict."""
    print(f"{'=' * 70}")
    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Description: {cfg['description']}")
    print(f"{'=' * 70}\n")

    # Load data
    data_cfg = cfg["data"]
    da_train, da_test = load_aphro_data(
        train_path=data_cfg["train_path"],
        test_path=data_cfg["test_path"],
        variable=data_cfg["variable"],
        lon_slice=tuple(data_cfg["lon_slice"]),
        lat_slice=tuple(data_cfg["lat_slice"]),
    )
    print(f"Data loaded: train={da_train.shape}, test={da_test.shape}")

    # Masks
    mask_cfg = cfg["masks"]
    country_masks = mask_cfg["country_masks"]
    output_mask_path = mask_cfg["output_mask_path"]
    centralised_mask_path = mask_cfg["centralised_mask_path"]

    # Training params
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    # DP config
    dp_cfg = cfg["dp"]

    # Output directory
    out_cfg = cfg["output"]
    results_dir = out_cfg["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    # Run
    if dp_cfg["enabled"]:
        dp_config = DPConfig(
            clip_norm=dp_cfg["clip_norm"],
            noise_multiplier=dp_cfg["noise_multiplier"],
            target_delta=dp_cfg["target_delta"],
            local_dp=dp_cfg["local_dp"],
            global_dp=dp_cfg["global_dp"],
        )
        results = run_federated_dp(
            da_train=da_train,
            da_test=da_test,
            country_masks=country_masks,
            output_mask_path=output_mask_path,
            centralised_mask_path=centralised_mask_path,
            dp_config=dp_config,
            num_rounds=train_cfg["num_rounds"],
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            lr=train_cfg["lr"],
            in_channels=model_cfg["in_channels"],
            base_filters=model_cfg["base_filters"],
            patch_size=train_cfg["patch_size"],
            stride=train_cfg["stride"],
        )
    else:
        results = run_federated(
            da_train=da_train,
            da_test=da_test,
            country_masks=country_masks,
            output_mask_path=output_mask_path,
            centralised_mask_path=centralised_mask_path,
            num_rounds=train_cfg["num_rounds"],
            local_epochs=train_cfg["local_epochs"],
            batch_size=train_cfg["batch_size"],
            lr=train_cfg["lr"],
            in_channels=model_cfg["in_channels"],
            base_filters=model_cfg["base_filters"],
            patch_size=train_cfg["patch_size"],
            stride=train_cfg["stride"],
        )

    # Save results
    summary = {
        "experiment_name": cfg["experiment_name"],
        "description": cfg["description"],
        "final_mse": results["final_mse"],
        "final_rmse": results["final_rmse"],
        "rounds": results["rounds"],
        "losses": results["losses"],
        "mse_values": results["mse_values"],
        "rmse_values": results.get("rmse_values", []),
        "config": results["config"],
    }

    if dp_cfg["enabled"]:
        summary["dp"] = {
            "local_dp": dp_cfg["local_dp"],
            "global_dp": dp_cfg["global_dp"],
            "clip_norm": dp_cfg["clip_norm"],
            "noise_multiplier": dp_cfg["noise_multiplier"],
            "target_delta": dp_cfg["target_delta"],
            "global_epsilon": results.get("global_epsilon"),
        }

    # Save JSON summary
    summary_path = os.path.join(results_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to: {summary_path}")

    # Save model weights
    if out_cfg.get("save_model", False):
        model_path = os.path.join(results_dir, "model.pt")
        torch.save(results["model"].state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    # Save per-round losses as numpy for easy plotting
    np.savez(
        os.path.join(results_dir, "history.npz"),
        rounds=np.array(results["rounds"]),
        losses=np.array(results["losses"]),
        mse_values=np.array(results["mse_values"]),
        rmse_values=np.array(results.get("rmse_values", [])),
    )

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run a DP experiment from config")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
