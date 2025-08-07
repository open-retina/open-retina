---
title: Installation
---

# Installation

`openretina` can be installed in several ways depending on your needs. Choose the installation method that best fits your use case.

## Prerequisites

`openretina` requires:

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (recommended for training)

## Method 1: Install from PyPI (Recommended for Users)

The simplest way to install `openretina` is through pip:

```bash
pip install openretina
```

This will install the latest stable release with all required dependencies.

## Method 2: Install from Source (Recommended for Developers)

For development or to access the example notebooks, clone the repository:

```bash
git clone git@github.com:open-retina/open-retina.git
cd open-retina
pip install -e .
```

The `-e` flag installs the package in "editable" mode, so changes to the source code are immediately reflected.

## Method 3: Install with Development Dependencies

If you want to contribute to `openretina`, install with development dependencies:

```bash
git clone git@github.com:open-retina/open-retina.git
cd open-retina
pip install -e ".[dev]"
```

This will install additional packages needed for testing, linting, and documentation.

## Method 4: Install with Extra Dependencies for Model Development

If you plan to develop new models or work with specific datasets:

```bash
git clone git@github.com:open-retina/open-retina.git
cd open-retina
pip install -e ".[devmodels]"
```

## Verifying Your Installation

To verify that `openretina` was installed correctly, run:

```python
import openretina
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load a small pre-trained model
from openretina.models import load_core_readout_from_remote
model = load_core_readout_from_remote("hoefling_2024_base_low_res", device)
print("Installation successful!")
```

## Common Issues

### CUDA Compatibility

If you encounter CUDA-related errors, ensure that:

1. Your PyTorch installation matches your CUDA version
2. You have the appropriate NVIDIA drivers installed

### Package Conflicts

If you encounter dependency conflicts, consider using a virtual environment:

```bash
python -m venv openretina_env
source openretina_env/bin/activate  # On Windows, use: openretina_env\Scripts\activate
pip install openretina
```

## Next Steps

Once you've successfully installed `openretina`, check out the [Quick Start Guide](./quickstart.md) to begin using the package. 
