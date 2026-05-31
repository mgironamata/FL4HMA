"""Backward-compatible re-exports.

This module previously contained all federated learning utilities.
It has been split into focused modules:

- ``fl4hma.utils.data``       — data loading (load_aphro_data, build_country_datasets)
- ``fl4hma.utils.masks``      — mask generation (generate_random_split_masks, generate_country_boundary_masks)
- ``fl4hma.utils.training``   — train/eval loops (train_sparse_pixel, evaluate_sparse_pixel, etc.)
- ``fl4hma.utils.federation`` — FL client, strategy, simulation (AphroFlowerClient, run_federated, etc.)
"""

# Re-export everything so existing imports still work.
from typing import Tuple

from fl4hma.data.data import build_country_datasets, load_aphro_data  # noqa: F401
from fl4hma.data.masks2 import (  # noqa: F401
    COUNTRY_NAME_MAP,
    generate_country_boundary_masks,
    generate_random_split_masks,
)
from fl4hma.federation.federation import (  # noqa: F401
    DEVICE,
    AphroFlowerClient,
    get_evaluate_fn,
    run_centralised,
    run_federated,
)
from fl4hma.training.training import (  # noqa: F401
    _get_device,
    evaluate_model_with_mask,
    evaluate_sparse_pixel,
    get_parameters,
    set_parameters,
    train_sparse_pixel,
)
