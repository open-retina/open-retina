# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (development):**
```bash
uv sync --extra dev
uv sync --extra optuna   # additionally, for hyperparameter sweeps
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

**Hyperparameter search (Optuna sweeper):**
```bash
uv run openretina train --config-name <name>_hparams_search --multirun
```
Without `--multirun` Hydra ignores the `hydra.sweeper` block entirely and runs a single job — useful
for ablations, silent if unintended. `objective_target` names the metric read from
`trainer.validate(..., ckpt_path="best")`; `objective_direction` is honoured only by the
`openretina train` entry point, which turns a failed trial into `-inf` instead of aborting the sweep.
To share one study across jobs, set `hydra.sweeper.storage` to a SQLite URL and reuse `study_name`.

## Architecture

The project implements retinal neural encoding models using a **shared core + multi-session readout** pattern backed by PyTorch Lightning and Hydra configuration.

### Core abstraction: Core-Readout

All main models follow `BaseCoreReadout(LightningModule)` in `models/core_readout.py`:
- **Core** (`modules/core/`): A single shared convolutional feature extractor across all recording sessions (e.g., `Core3d` for spatiotemporal CNNs, GRU-based variants).
- **Readout** (`modules/readout/`): Per-session decoders mapping core features to individual neuron responses. Implementations include Gaussian readouts, factorized readouts, and spatial contrast readouts. All implement the abstract `Readout` interface with `initialize()`, `regularizer()`, and `initialize_bias()`.
- **Shifter** (`modules/shifters/`, optional): Per-session MLP mapping a behavioral variable (pupil center) to a 2D shift added to each neuron's readout RF center, correcting for eye movements. Defaults to `None`, in which case the model behaves exactly as before it existed. Only engages when the batch also carries `pupil_center` and a `data_key`.

Model variants in `models/`:
- `core_readout.py` — base multi-session encoder
- `spatial_contrast.py` — spatial contrast sensitivity model
- `linear_nonlinear.py` — LNP (Linear-Nonlinear-Poisson) baseline
- `sparse_autoencoder.py` — autoencoder variants

### Data

`data_io/base_dataloader.py` defines `MovieDataSet` and `MultiSessionDataLoader`. Each published dataset has its own submodule under `data_io/` (e.g., `hoefling_2024/`, `karamanlis_2024/`, `maheswaranathan_2023/`, `qiu_2026/`).

**Stimulus shape convention:** `(channels, time, height, width)` — e.g., `[2, 150, 72, 64]` means 2 color channels, 150 frames, 72×64 pixels. Channels are not necessarily colors: `qiu_2026` folds behavioral traces in as extra input channels, so its `[3, 300, 36, 64]` is 1 video + 2 behavior channels.

**Response shape:** `(n_frames, n_neurons)`, or a dict with `"avg"` (trial-averaged) and `"by_trial"` keys.

### Configuration

Hydra YAML configs live in `configs/` with one file per model/paper. The training CLI automatically composes defaults. Override any key from the command line. The environment variable `$OPENRETINA_CACHE_DIRECTORY` controls where models and datasets are cached (defaults to `~/openretina_cache`).

**Dataloaders are built via `_partial_`, deliberately.** Both CLIs do:
```python
build_dataloaders = hydra.utils.instantiate(cfg.dataloader, _partial_=True)
dataloaders = build_dataloaders(**dataloader_kwargs)   # plain Python call
```
Do not collapse this into `instantiate(cfg.dataloader, **dataloader_kwargs)`. Data passed *through*
`instantiate` is OmegaConf-rebuilt, so the builder receives copies and cannot free the caller's
movies — worth ~28 GB of RAM on `qiu_2026`, with no error raised anywhere. See
`hydra_partial_instantiate.md`. The same copying makes `release_movies: true` silently inert on the
`instantiate(...)` path, which is why notebooks keep their movies dict.

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
