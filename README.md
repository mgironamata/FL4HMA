# FL4HMA
Federated Learning for High Mountain Asia

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
pip install -e .[examples]  # Adds torch, torchvision, matplotlib
```

For development tools:
```bash
pip install -e .[dev]       # Adds pytest, black, flake8
```

## Structure

```
FL4HMA/
├── src/
│   └── fl4hma/    # Main package (src layout)
│       ├── core/          # Core federated learning functionality  
│       ├── data/          # Data handling and preprocessing
│       ├── models/        # Machine learning models
│       └── utils/         # Utility functions
├── tests/         # Tests
├── examples/      # Examples
└── pyproject.toml # Package configuration
```

This project uses the **src layout**, which provides better isolation between source code and the installed package, preventing common development pitfalls.

## Examples

### CIFAR-10 Sparse Labeling Example

A minimal example demonstrating sparse labeling with CNNs on CIFAR-10:

```bash
# Install with example dependencies
pip install -e .[examples]

# Run the example
python run_sparse_example.py
```

This example shows:
- Loading CIFAR-10 with artificial sparse labeling (10% labeled data)
- Training a vanilla CNN with sparse supervision
- Visualizing labeled vs unlabeled samples
- Evaluating model performance

Perfect for demonstrating federated learning scenarios where different clients have different amounts of labeled data.

## Usage

```python
import fl4hma
```