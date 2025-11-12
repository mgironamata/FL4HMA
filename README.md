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

### CIFAR-10 Sparse Pixel Example

A realistic example demonstrating **image-to-image learning with sparse pixel supervision**:

```bash
# Make sure you're in the fl4hma environment
conda activate fl4hma

# Install with example dependencies
pip install -e .[examples]

# Run the Python example
python run_sparse_example.py

# Or run the interactive Jupyter notebook
jupyter notebook examples/cifar10_sparse_demo.ipynb
```

This example shows:
- **Sparse input pixels**: Images with missing/masked pixels (simulating clouds, sensor failures)
- **Sparse output labels**: Only some pixels have target values (simulating limited ground truth)
- **U-Net CNN architecture**: For pixel-to-pixel prediction and reconstruction
- **Realistic federated learning scenario**: Different clients with varying data availability

**Perfect for applications like:**
- Satellite imagery with cloud cover
- Climate data with sparse measurements  
- Remote sensing with incomplete observations
- Federated learning with heterogeneous data quality

## Usage

```python
import fl4hma
```