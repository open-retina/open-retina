---
title: Installation
---

# Installation

`openretina` can be installed in several ways depending on your needs. Choose the installation method that best fits your use case. In either case, we recommend a dedicated virtual environment to avoid dependency conflicts.

## Prerequisites

`openretina` requires:

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (recommended for training)

## Method 1: Install from PyPI (Recommended for Users)

If you only want to use the pre-trained models and intend to use `openretina` "as-is" for analysis and
exploration rather than extending it, the simplest way is to install the package directly from the Python Package Index (PyPI) by running the following command:

```bash
pip install openretina
```

This will install the latest stable release with all required dependencies.

## Method 2: Install from Source (Recommended for Developers)

If you would like to train your own models, modify configurations, and follow the accompanying
Jupyter notebooks, we recommend cloning our GitHub repository and installing from source:

```bash
git clone git@github.com:open-retina/open-retina.git
cd open-retina
pip install -e .
```

The `-e` flag installs the package in "editable" mode, so changes to the source code are immediately reflected.

If you want to contribute code to `openretina`, you can additionally install its development dependencies to gain access to the packages we use for testing, linting, and generating this documentation:

```bash
pip install -e ".[dev]"
```

To run hyperparameter tuning or to test models that are currently in development and not officially supported, use the following installation options:
```bash
# for hyperparameter tuning
pip install -e ".[optuna]"
# for using dev models
pip install -e ".[devmodels]"
```

## Verifying Your Installation

Regardless of which installation method you choose, you can verify that `openretina` was installed correctly by downloading a model and running a forward pass as follows:

```python
import openretina
import torch

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

from openretina.models import load_core_readout_from_remote
# Load a small pre-trained model
model = load_core_readout_from_remote("hoefling_2024_base_low_res", device)
# Run forward pass
responses = model.forward(torch.rand(model.stimulus_shape(time_steps=50)))
```

This first confirms the device you will use for computation, either your CPU or a CUDA GPU. It then downloads a pre-trained model and predicts the activity of the modeled neurons based on a random stimulus.


## Next Steps

Once you've successfully installed `openretina`, check out the [Quick Start Guide](./quickstart.md) to begin using the package. 


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
