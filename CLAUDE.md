# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (development):**
```bash
uv sync --extra dev
```

**Run tests:**
```bash
make test-all                         # types + codestyle + formatting + unittests
make test-unittests                   # pytest only
uv run pytest tests/path/to/test.py::test_name  # single test
uv run pytest tests/ -k "keyword"    # filter by name
make test-notebooks                   # nbmake on notebooks/
```

**Lint and format:**
```bash
make fix-all          # fix formatting and codestyle
make fix-formatting   # ruff format
make fix-codestyle    # ruff check --fix
make test-formatting  # check only (no changes)
make test-codestyle   # check only (no changes)
make test-types       # mypy type checking
```

**Training (CLI):**
```bash
uv run openretina train                                   # uses default Hydra config
uv run openretina train model.in_shape=[2,150,72,64]      # override config keys
uv run openretina eval                                     # evaluate a trained model
```

## Architecture

The project implements retinal neural encoding models using a **shared core + multi-session readout** pattern backed by PyTorch Lightning and Hydra configuration.

### Core abstraction: Core-Readout

All main models follow `BaseCoreReadout(LightningModule)` in `models/core_readout.py`:
- **Core** (`modules/core/`): A single shared convolutional feature extractor across all recording sessions (e.g., `Core3d` for spatiotemporal CNNs, GRU-based variants).
- **Readout** (`modules/readout/`): Per-session decoders mapping core features to individual neuron responses. Implementations include Gaussian readouts, factorized readouts, and spatial contrast readouts. All implement the abstract `Readout` interface with `initialize()`, `regularizer()`, and `initialize_bias()`.

Model variants in `models/`:
- `core_readout.py` — base multi-session encoder
- `spatial_contrast.py` — spatial contrast sensitivity model
- `linear_nonlinear.py` — LNP (Linear-Nonlinear-Poisson) baseline
- `sparse_autoencoder.py` — autoencoder variants

### Data

`data_io/base_dataloader.py` defines `MovieDataSet` and `MultiSessionDataLoader`. Each published dataset has its own submodule under `data_io/` (e.g., `hoefling_2024/`, `karamanlis_2024/`, `maheswaranathan_2023/`).

**Stimulus shape convention:** `(channels, time, height, width)` — e.g., `[2, 150, 72, 64]` means 2 color channels, 150 frames, 72×64 pixels.

**Response shape:** `(n_frames, n_neurons)`, or a dict with `"avg"` (trial-averaged) and `"by_trial"` keys.

### Configuration

Hydra YAML configs live in `configs/` with one file per model/paper. The training CLI automatically composes defaults. Override any key from the command line. The environment variable `$OPENRETINA_CACHE_DIRECTORY` controls where models and datasets are cached (defaults to `~/openretina_cache`).

### Interpretability tools

`insilico/` contains analysis tools:
- `stimulus_optimization/` — MEI (most exciting input) computation
- `vector_field_analysis/` — gradient-based trajectory analysis
- `tuning_analyses/` — neuron property analysis

### Pre-trained models

Models are hosted on HuggingFace (`open-retina/open-retina`). Load them with:
```python
from openretina.models.core_readout import load_core_readout_from_remote
model = load_core_readout_from_remote("hoefling_2024_low_res", device="cpu")
```

## Conventions

- **Type annotations:** Use `jaxtyping` for tensor shapes, e.g., `Float[torch.Tensor, "batch time neurons"]`.
- **Line length:** 120 characters.
- **Excluded from linting/type-checking:** `openretina/legacy/` and `tests/paper_openretina_2025/`.
- **Causality:** Models must be causal; verified by `is_model_causal()` in the test suite.
- **Python version:** 3.10+.
