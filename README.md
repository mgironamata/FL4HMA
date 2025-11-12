# FL4HMA
Federated Learning for High Mountain Asia

## Installation

### Development Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/mgironamata/FL4HMA.git
cd FL4HMA
pip install -e .
```

This installs the package in "editable" mode, so changes to the source code are immediately available without reinstalling.

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


## Usage

```python
import fl4hma
```