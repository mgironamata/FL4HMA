"""Mask management for federated learning experiments."""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr

from fl4hma.data.station_masks import (
    generate_country_boundary_masks,
    generate_random_split_masks,
)
from fl4hma.utils.config import ExperimentConfig


class MaskManager:
    """Loads and generates all masks needed for the experiment suite.

    Produces 2-D ``(lat, lon)`` masks compatible with
    :class:`~fl4hma.data.dataset.StationPatchDataset`.  All masks are saved as
    integer ``0/1`` arrays so they convert cleanly to ``torch.bool`` in the
    dataset's ``__getitem__``.

    Handles:
    - Centralised station mask (pre-existing)
    - Output (land/sea) mask (derived from data)
    - Per-country station masks (pre-existing)
    - Random split masks (generated from centralised mask)
    - Country boundary masks (rasterised from Natural Earth polygons)
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.centralised_mask_path: Optional[str] = None
        self.centralised_mask_arr: Optional[np.ndarray] = None
        self.output_mask_path: Optional[str] = None
        self.country_masks: Dict[str, str] = {}
        self.random_masks: Dict[str, str] = {}
        self.boundary_masks: Dict[str, str] = {}
        self.n_stations: int = 0
        # Spatial grid metadata (populated after load)
        self.lat_vals: Optional[np.ndarray] = None
        self.lon_vals: Optional[np.ndarray] = None

    @property
    def spatial_shape(self) -> Optional[Tuple[int, int]]:
        """(lat_len, lon_len) of all masks, or None before load()."""
        if self.lat_vals is not None and self.lon_vals is not None:
            return (len(self.lat_vals), len(self.lon_vals))
        return None

    def load(
        self,
        train_path: str,
        variable: str,
        local_mask_dir: str,
    ) -> "MaskManager":
        """Load all masks for a given variable/dataset.

        Parameters
        ----------
        train_path : str
            Path to the training NetCDF file (used to derive land mask).
        variable : str
            NetCDF variable name (e.g. "precip", "tave").
        local_mask_dir : str
            Directory to store generated masks.

        Returns
        -------
        self
            For method chaining.
        """
        cfg = self.config
        lon_sl = slice(*cfg.lon_slice)
        lat_sl = slice(*cfg.lat_slice)
        os.makedirs(local_mask_dir, exist_ok=True)

        # --- Determine spatial grid from the data (consistent with load_aphro_data) ---
        ds_ref = xr.open_dataset(train_path).sel(lon=lon_sl, lat=lat_sl)
        self.lat_vals = ds_ref.lat.values
        self.lon_vals = ds_ref.lon.values
        expected_shape = (len(self.lat_vals), len(self.lon_vals))

        # --- Centralised mask (pre-existing) ---
        self.centralised_mask_path = os.path.join(
            cfg.stat_mask_dir, "centralised_mask.npy"
        )
        self.centralised_mask_arr = np.load(self.centralised_mask_path)
        self._validate_shape(
            self.centralised_mask_arr, expected_shape, "centralised_mask"
        )
        self.n_stations = int(self.centralised_mask_arr.sum())
        print(
            f"  Loaded centralised_mask: density={self.centralised_mask_arr.mean()*100:.2f}%, "
            f"stations={self.n_stations}"
        )

        # --- Output (land/sea) mask ---
        self.output_mask_path = os.path.join(local_mask_dir, "out_mask.npy")
        if not os.path.exists(self.output_mask_path):
            out_mask_arr = np.where(np.isnan(ds_ref[variable][0].values), 0, 1).astype(
                np.int8
            )
            np.save(self.output_mask_path, out_mask_arr)
            print(
                f"  Generated out_mask: shape={out_mask_arr.shape}, "
                f"density={out_mask_arr.mean()*100:.1f}%"
            )
        else:
            out_mask_arr = np.load(self.output_mask_path)
            self._validate_shape(out_mask_arr, expected_shape, "out_mask")
            print(f"  Loaded out_mask: density={out_mask_arr.mean()*100:.1f}%")

        ds_ref.close()

        # --- Country station masks (pre-existing) ---
        self.country_masks = {}
        for name in cfg.countries:
            path = os.path.join(cfg.stat_mask_dir, f"{name}_mask.npy")
            self.country_masks[name] = path
            m = np.load(path)
            self._validate_shape(m, expected_shape, f"{name}_mask")
            print(f"    {name:15s}: {int(m.sum()):4d} stations")

        # --- Random split masks ---
        random_dir = os.path.join(local_mask_dir, "random_split")
        self.random_masks = generate_random_split_masks(
            self.centralised_mask_arr,
            cfg.n_clients,
            out_dir=random_dir,
            seed=42,
            prefix="random",
        )
        print(f"  Random split masks ({cfg.n_clients} clients):")
        for name, path in self.random_masks.items():
            m = np.load(path)
            print(f"    {name:15s}: {int(m.sum()):4d} stations")

        # --- Country boundary masks ---
        boundary_dir = os.path.join(local_mask_dir, "boundary")
        print("  Country boundary masks:")
        self.boundary_masks = generate_country_boundary_masks(
            lat_vals=self.lat_vals,
            lon_vals=self.lon_vals,
            countries=cfg.countries,
            out_dir=boundary_dir,
            land_mask=out_mask_arr,
            geojson_path=cfg.geojson_path,
        )

        return self

    @staticmethod
    def _validate_shape(
        mask: np.ndarray,
        expected: Tuple[int, int],
        name: str,
    ) -> None:
        """Raise if a 2-D mask doesn't match the expected spatial grid."""
        if mask.ndim == 2 and mask.shape != expected:
            raise ValueError(
                f"Mask '{name}' has shape {mask.shape} but the data grid "
                f"requires {expected}. Ensure masks match the lon/lat slice "
                f"used by StationPatchDataset."
            )
