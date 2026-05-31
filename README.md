# Federated learning for High Mountain Asia (FL4HMA)

Federated learning experiments to recreate APHRODITE precipitation and temperature fields over High Mountain Asia. The package allows user to train a infilling model from stations locations in centralised and federated learning frameworks.

The federated model configurations include:
* randomised station mask
* per-country stations masks
* per-country evaluation
* different federated averaging methods
* local and global differential privacy


## Installation

### Development Installation

We recommend creating a fresh conda environment for the project:

```bash
# Create new environment with Python 3.9 (or 3.8, 3.10, 3.11)
conda create -n fl4hma python=3.9

# Activate the environment
conda activate fl4hma
```

Clone the repository and install in development mode:

```bash
git clone https://github.com/mgironamata/FL4HMA.git
cd FL4HMA
pip install -e .
```

This installs the package in "editable" mode, so changes to the source code are immediately available without reinstalling.

To add optional dependencies, follow the below instructions instead. 

For running examples:
```bash
pip install -e ".[examples]"  # Adds torch, torchvision, matplotlib
```

For development tools:
```bash
pip install -e ".[dev]"       # Adds pytest, black, flake8
```

## Structure

```
FL4HMA/
├── src/fl4hma/             # Main package (src layout)
│   ├── core/
│   │   ├── config.py       # ExperimentConfig dataclass
│   │   └── runner.py       # ExperimentRunner (full experiment suite)
│   ├── data/
│   │   ├── create.py       # APHRODITE NetCDF → xarray loader
│   │   ├── data.py         # load_aphro_data, build_country_datasets
│   │   ├── masks.py        # MaskManager with shape validation
│   │   ├── masks2.py       # Mask generation (random split, country boundaries)
│   │   └── torch_dataset.py  # StationPatchDataset (sparse pixel patches)
│   ├── federation/
│   │   ├── federation.py   # AphroFlowerClient, run_federated, run_centralised
│   │   └── differential_privacy.py  # DPConfig, DPFedAvg, run_federated_dp
│   ├── models/
│   │   └── unet.py         # UNetCNN for sparse pixel regression
│   └── training/
│       └── training.py     # train_sparse_pixel, evaluate_sparse_pixel
├── examples/
│   ├── federated_dp_demo.ipynb          # Differential privacy demo
│   ├── federated_sparse_cifar10_demo.ipynb
│   ├── cifar10_sparse_demo.ipynb
│   └── sparse_pixel_example.py
├── tests/
├── data/                   # Station masks, country boundaries, elevation
└── pyproject.toml
```

## Usage

### Quick Start

```python
from fl4hma.core.config import ExperimentConfig
from fl4hma.data.data import load_aphro_data
from fl4hma.federation.federation import run_federated

# Load APHRODITE temperature data
da_train, da_test = load_aphro_data(
    "path/to/train.nc", "path/to/test.nc", variable="tave"
)

# Run federated learning with country-based client splits
results = run_federated(
    da_train=da_train,
    da_test=da_test,
    country_masks={"nepal": "masks/nepal.npy", "india": "masks/india.npy"},
    output_mask_path="masks/out_mask.npy",
    centralised_mask_path="masks/centralised.npy",
    num_rounds=10,
)
```

### Federated Learning with Differential Privacy

```python
from fl4hma.federation.differential_privacy import DPConfig, run_federated_dp

dp_config = DPConfig(
    clip_norm=1.0,
    noise_multiplier=1.0,
    target_delta=1e-5,
    local_dp=True,   # DP-SGD at each client
    global_dp=True,  # Server-side noise on aggregated model
)

dp_results = run_federated_dp(
    da_train=da_train,
    da_test=da_test,
    country_masks=country_masks,
    output_mask_path="masks/out_mask.npy",
    centralised_mask_path="masks/centralised.npy",
    dp_config=dp_config,
    num_rounds=10,
)
print(f"Final MSE: {dp_results['final_mse']:.4f}, ε = {dp_results['global_epsilon']:.2f}")
```

### Full Experiment Suite

```python
from fl4hma.utils.config import ExperimentConfig
from fl4hma.utils.runner import ExperimentRunner

cfg = ExperimentConfig(
    countries=["nepal", "india", "china"],
    num_rounds=10,
    n_clients=5,
)
runner = ExperimentRunner(cfg)
runner.run_all()
```

## Examples

| Notebook | Description |
|----------|-------------|
| `examples/federated_dp_demo.ipynb` | Differential privacy in FL: local DP-SGD, global DP, privacy accounting, utility trade-offs |
| `examples/federated_sparse_cifar10_demo.ipynb` | Federated sparse pixel regression on CIFAR-10 |
| `examples/cifar10_sparse_demo.ipynb` | Single-client sparse pixel training demo |