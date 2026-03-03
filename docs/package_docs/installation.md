---
title: Installation
---

# Installation

`openretina` can be installed in several ways depending on your needs. We recommend using a dedicated virtual environment to avoid dependency conflicts.

## Prerequisites

`openretina` requires:

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (recommended for training)

## Recommended paths

### Method 1: Install from PyPI (quick user path)

If you only want to use the pre-trained models and intend to use 
`openretina` "as-is" for analysis and
exploration rather than extending it, the simplest way is to 
install the package directly from the Python Package Index (PyPI) 
by running the following command:

```bash
pip install openretina
```

This will install the latest public stable release with all required dependencies.

### Method 2: Install from source with uv (recommended for development)

If you would like to train your own models, modify configurations, and follow the accompanying
Jupyter notebooks, we recommend cloning our GitHub repository and installing from source:

```bash
git clone git@github.com:open-retina/open-retina.git
cd open-retina
uv sync
```

`uv` is the default workflow in this repository and matches the provided `Makefile` commands (`uv run ...`).

If you additionally want to contribute models or datasets, edit the documentation, or implement bug fixes and improvements, install the development dependencies via:

```bash
uv sync --extra dev
```

### Method 3: Install from source with pip (alternative)

```bash
git clone git@github.com:open-retina/open-retina.git
cd open-retina
pip install -e .
```
The `-e` flag installs the package in "editable" mode, so changes to the source code are immediately reflected.

Optional extras:

```bash
# development
pip install -e ".[dev]"

# hyperparameter tuning support
pip install -e ".[optuna]"

# development-only models
pip install -e ".[devmodels]"
```

## Choosing the right PyTorch backend

Different systems may require different PyTorch wheels. Use the official PyTorch selector to generate commands for your platform:

- [PyTorch install selector](https://pytorch.org/get-started/locally/)
- [PyTorch MPS notes (Apple Silicon)](https://pytorch.org/docs/stable/notes/mps.html)

For `uv` users, a practical approach is:

1. Install/sync OpenRetina environment (`uv sync ...`).
2. Run a quick import/device check (below). If you are on Linux and on the latest CUDA backends, this can already work out of the box. If it did not:
3. Install the PyTorch build matching your backend and driver/runtime, [following the guide provided by uv](https://docs.astral.sh/uv/guides/integration/pytorch/).
4. Re-run a quick import/device check (below).

## Verifying Your Installation

Run this diagnostic snippet:

```python
import torch

print("torch:", torch.__version__)
print("cuda_runtime:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("mps_available:", hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
```

Then verify OpenRetina by loading a pre-trained model and running a forward pass:

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

## Next Steps

Once you've successfully installed `openretina`, check out the [Quick Start Guide](./quickstart.md) to begin using the package.

## Common Issues

### CUDA Compatibility

If you encounter CUDA-related errors, ensure that:

1. Your PyTorch installation matches your CUDA version
2. You have the appropriate NVIDIA drivers installed
3. You used the install command from the PyTorch selector for your exact platform/backend

### Package Conflicts

If you encounter dependency conflicts, consider using a virtual environment:

```bash
python -m venv openretina_env
source openretina_env/bin/activate  # On Windows, use: openretina_env\Scripts\activate
pip install openretina
```
