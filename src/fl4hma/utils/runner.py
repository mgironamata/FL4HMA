"""Experiment runner for centralised vs federated comparison."""

from typing import Dict

import numpy as np
import xarray as xr

from fl4hma.data.mask_manager import MaskManager
from fl4hma.federation.federation import run_centralised, run_federated
from fl4hma.training.training import evaluate_model_with_mask
from fl4hma.utils.config import ExperimentConfig


class ExperimentRunner:
    """Orchestrates the full experiment suite from the notebook.

    Runs:
    - Centralised baseline
    - Experiment 1: FL with random IID split
    - Experiment 2: FL with country (non-IID) split
    - Experiment 3: Per-country centralised models + cross-country heatmap
    """

    def __init__(self, config: ExperimentConfig, masks: MaskManager):
        self.config = config
        self.masks = masks
        self.results: Dict = {}

    def run_all(
        self,
        da_train: xr.DataArray,
        da_test: xr.DataArray,
    ) -> Dict:
        """Run all experiments and return the combined results dict.

        Parameters
        ----------
        da_train : xr.DataArray
            Training data array with dims (variable, time, lat, lon).
        da_test : xr.DataArray
            Test data array with dims (variable, time, lat, lon).

        Returns
        -------
        dict
            Results containing all models and evaluation metrics.
        """
        self.results = {}
        self._run_centralised_baseline(da_train, da_test)
        self._run_exp1_random_split(da_train, da_test)
        self._run_exp2_country_split(da_train, da_test)
        self._run_exp3_per_country(da_train, da_test)

        print("\n" + "=" * 70)
        print("ALL EXPERIMENTS COMPLETE")
        print("=" * 70)
        return self.results

    def _run_centralised_baseline(
        self, da_train: xr.DataArray, da_test: xr.DataArray
    ) -> None:
        """Train and evaluate the centralised baseline model."""
        cfg = self.config
        masks = self.masks

        print("\n" + "=" * 70)
        print("CENTRALISED BASELINE")
        print("=" * 70)

        cl_res = run_centralised(
            da_train=da_train,
            da_test=da_test,
            input_mask_path=masks.centralised_mask_path,
            output_mask_path=masks.output_mask_path,
            num_epochs=cfg.num_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            in_channels=cfg.in_channels,
            base_filters=cfg.base_filters,
            patch_size=cfg.patch_size,
            stride=cfg.stride,
        )
        self.results["cl_model"] = cl_res["model"]

        # Evaluate with centralised mask
        self.results["cl_cent_eval"] = self._eval(
            cl_res["model"],
            da_test,
            masks.centralised_mask_path,
            masks.output_mask_path,
        )

    def _run_exp1_random_split(
        self, da_train: xr.DataArray, da_test: xr.DataArray
    ) -> None:
        """Experiment 1: FL with random IID split."""
        cfg = self.config
        masks = self.masks

        print("\n" + "=" * 70)
        print("EXPERIMENT 1: FL WITH RANDOM SPLIT")
        print("=" * 70)

        fl_res = run_federated(
            da_train=da_train,
            da_test=da_test,
            country_masks=masks.random_masks,
            output_mask_path=masks.output_mask_path,
            centralised_mask_path=masks.centralised_mask_path,
            test_input_mask_path=masks.centralised_mask_path,
            num_rounds=cfg.num_rounds,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            in_channels=cfg.in_channels,
            base_filters=cfg.base_filters,
            patch_size=cfg.patch_size,
            stride=cfg.stride,
        )
        fl_model = fl_res["model"]
        self.results["fl_random_model"] = fl_model

        # Global eval
        self.results["fl_cent_eval"] = self._eval(
            fl_model,
            da_test,
            masks.centralised_mask_path,
            masks.output_mask_path,
        )

        # Per-client eval
        cl_model = self.results["cl_model"]
        exp1_per_client = {}
        for name, path in masks.random_masks.items():
            cl_m = self._eval(cl_model, da_test, path, masks.output_mask_path)
            fl_m = self._eval(fl_model, da_test, path, masks.output_mask_path)
            exp1_per_client[name] = {"cl": cl_m, "fl": fl_m}
        self.results["exp1_per_client"] = exp1_per_client

    def _run_exp2_country_split(
        self, da_train: xr.DataArray, da_test: xr.DataArray
    ) -> None:
        """Experiment 2: FL with country (non-IID) split."""
        cfg = self.config
        masks = self.masks

        print("\n" + "=" * 70)
        print("EXPERIMENT 2: FL WITH COUNTRY SPLIT")
        print("=" * 70)

        fl_res = run_federated(
            da_train=da_train,
            da_test=da_test,
            country_masks=masks.country_masks,
            output_mask_path=masks.output_mask_path,
            centralised_mask_path=masks.centralised_mask_path,
            test_input_mask_path=masks.centralised_mask_path,
            num_rounds=cfg.num_rounds,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            in_channels=cfg.in_channels,
            base_filters=cfg.base_filters,
            patch_size=cfg.patch_size,
            stride=cfg.stride,
        )
        fl_model = fl_res["model"]
        self.results["fl_country_model"] = fl_model

        # Global eval
        self.results["fl_country_cent_eval"] = self._eval(
            fl_model,
            da_test,
            masks.centralised_mask_path,
            masks.output_mask_path,
        )

        # Per-country eval — station masks
        cl_model = self.results["cl_model"]
        exp2_per_country = {}
        for name, path in masks.country_masks.items():
            cl_m = self._eval(cl_model, da_test, path, masks.output_mask_path)
            fl_m = self._eval(fl_model, da_test, path, masks.output_mask_path)
            exp2_per_country[name] = {"cl": cl_m, "fl": fl_m}
        self.results["exp2_per_country"] = exp2_per_country

        # Per-country eval — boundary masks
        exp2_per_country_boundary = {}
        for name, bnd_path in masks.boundary_masks.items():
            cl_m = self._eval(cl_model, da_test, masks.centralised_mask_path, bnd_path)
            fl_m = self._eval(fl_model, da_test, masks.centralised_mask_path, bnd_path)
            exp2_per_country_boundary[name] = {"cl": cl_m, "fl": fl_m}
        self.results["exp2_per_country_boundary"] = exp2_per_country_boundary

    def _run_exp3_per_country(
        self, da_train: xr.DataArray, da_test: xr.DataArray
    ) -> None:
        """Experiment 3: Per-country centralised models + cross-country heatmap."""
        cfg = self.config
        masks = self.masks

        print("\n" + "=" * 70)
        print("EXPERIMENT 3: PER-COUNTRY CL MODELS")
        print("=" * 70)

        fl_country_model = self.results["fl_country_model"]

        # Train per-country centralised models
        cl_country_models = {}
        for name, mask_path in masks.country_masks.items():
            print(f"\n--- Training CL model for: {name.upper()} ---")
            res = run_centralised(
                da_train=da_train,
                da_test=da_test,
                input_mask_path=mask_path,
                output_mask_path=masks.output_mask_path,
                num_epochs=cfg.num_epochs,
                batch_size=cfg.batch_size,
                lr=cfg.lr,
                in_channels=cfg.in_channels,
                base_filters=cfg.base_filters,
                patch_size=cfg.patch_size,
                stride=cfg.stride,
            )
            cl_country_models[name] = res["model"]
        self.results["cl_country_models"] = cl_country_models

        # 3A: Within-country — station masks
        exp3a = {}
        for name, mask_path in masks.country_masks.items():
            cl_m = self._eval(
                cl_country_models[name], da_test, mask_path, masks.output_mask_path
            )
            fl_m = self._eval(
                fl_country_model, da_test, mask_path, masks.output_mask_path
            )
            exp3a[name] = {"cl": cl_m, "fl": fl_m}
        self.results["exp3a"] = exp3a

        # 3A-boundary: Within-country — full boundary masks
        exp3a_boundary = {}
        for name, bnd_path in masks.boundary_masks.items():
            cl_m = self._eval(
                cl_country_models[name], da_test, masks.country_masks[name], bnd_path
            )
            fl_m = self._eval(
                fl_country_model, da_test, masks.centralised_mask_path, bnd_path
            )
            exp3a_boundary[name] = {"cl": cl_m, "fl": fl_m}
        self.results["exp3a_boundary"] = exp3a_boundary

        # 3B: Cross-country heatmap — station masks
        countries = list(masks.country_masks.keys())
        nc = len(countries)
        heatmap = np.zeros((nc + 1, nc))
        for i, tc in enumerate(countries):
            for j, ec in enumerate(countries):
                met = self._eval(
                    cl_country_models[tc],
                    da_test,
                    masks.country_masks[ec],
                    masks.output_mask_path,
                )
                heatmap[i, j] = met["rmse"]
        for j, ec in enumerate(countries):
            met = self._eval(
                fl_country_model,
                da_test,
                masks.country_masks[ec],
                masks.output_mask_path,
            )
            heatmap[nc, j] = met["rmse"]
        self.results["heatmap"] = heatmap

        # 3B-boundary: Cross-country heatmap — full boundary masks
        heatmap_bnd = np.zeros((nc + 1, nc))
        for i, tc in enumerate(countries):
            for j, ec in enumerate(countries):
                met = self._eval(
                    cl_country_models[tc],
                    da_test,
                    masks.country_masks[tc],
                    masks.boundary_masks[ec],
                )
                heatmap_bnd[i, j] = met["rmse"]
        for j, ec in enumerate(countries):
            met = self._eval(
                fl_country_model,
                da_test,
                masks.centralised_mask_path,
                masks.boundary_masks[ec],
            )
            heatmap_bnd[nc, j] = met["rmse"]
        self.results["heatmap_boundary"] = heatmap_bnd
        self.results["countries"] = countries

    def _eval(self, model, da_test, input_mask_path, output_mask_path):
        """Shorthand for evaluate_model_with_mask with config defaults."""
        cfg = self.config
        return evaluate_model_with_mask(
            model,
            da_test,
            input_mask_path,
            output_mask_path,
            batch_size=cfg.batch_size,
            patch_size=cfg.patch_size,
            stride=cfg.stride,
        )
