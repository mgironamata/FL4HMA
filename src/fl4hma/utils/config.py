"""Experiment configuration dataclass."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch


@dataclass
class ExperimentConfig:
    """Hyperparameters and paths for a federated learning experiment suite.

    Attributes
    ----------
    batch_size : int
        Mini-batch size for training and evaluation.
    num_epochs : int
        Number of centralised training epochs.
    num_rounds : int
        Number of FL communication rounds.
    local_epochs : int
        Client-local training epochs per FL round.
    lr : float
        Learning rate.
    in_channels : int
        Number of input channels (e.g. variable + lat + lon = 3).
    base_filters : int
        Base filter count for UNet.
    patch_size : int
        Spatial patch size for dataset extraction.
    stride : int
        Stride for patch extraction.
    n_clients : int
        Number of random-split FL clients.
    countries : list of str
        Country names for non-IID partitioning.
    stat_mask_dir : str
        Directory containing pre-existing station masks.
    device : torch.device
        Compute device (auto-detected if not specified).
    """

    batch_size: int = 16
    num_epochs: int = 5
    num_rounds: int = 5
    local_epochs: int = 1
    lr: float = 0.001
    in_channels: int = 3
    base_filters: int = 32
    patch_size: int = 32
    stride: int = 32
    n_clients: int = 5
    countries: List[str] = field(
        default_factory=lambda: [
            "afghanistan",
            "china",
            "india",
            "nepal",
            "pakistan",
            "tajikistan",
            "uzbekistan",
        ]
    )
    lon_slice: Tuple[float, float] = (60.0, 105.0)
    lat_slice: Tuple[float, float] = (20.0, 40.0)
    stat_mask_dir: str = "station_data/masks/stat"
    geojson_path: str = "data/country_masks/geoBoundariesCGAZ_ADM0.geojson"
    mask_start_year: int = 1998
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
