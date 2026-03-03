# How to

This page is the fastest way to find the right OpenRetina entry point for your goal.

## Choose your workflow

- **Install OpenRetina and verify your environment**: start with the [Installation guide](../installation.md).
- **Run a pre-trained model for inference**: use the [Quick Start guide](../quickstart.md).
- **Train in a notebook first**: begin with [`notebooks/openretina_demo.ipynb`](https://github.com/open-retina/open-retina/blob/main/notebooks/openretina_demo.ipynb), then check [Training overview](../training/index.md).
- **Train from the CLI with configs**: go to the [Command Line guide](../command_line.md) and the [Unified Training Script guide](../training/unified_training.md).
- **Add a new dataset**: follow [Data I/O](../data_io.md), [`docs/data_io_flow.md`](../../data_io_flow.md), and [`notebooks/new_datasets_guide.ipynb`](https://github.com/open-retina/open-retina/blob/main/notebooks/new_datasets_guide.ipynb).
- **Contribute a new model**: start from [Core-Readout models](../training/core_readout.md) and [Contributing](../contributing.md).
- **Evaluate an existing or trained model**: use [Command Line](../command_line.md#evaluate-a-model) and [Training and Evaluation](../training/training_and_evaluation.md).

## Hydra and Lightning: which path should I pick?

OpenRetina supports two main training workflows:

**Hydra + Lightning (recommended for most users)**:
- Best for reproducible experiments and collaborative work.
- Uses composable YAML configs (`data_io`, `dataloader`, `model`, `trainer`, etc.).
- Launches with one command (`openretina train ...`).
**Manual Python + Lightning (advanced/prototyping)**:
- Best when you are quickly iterating on custom code.
- You build loaders and `Trainer(...)` directly in Python.
- You manage reproducibility and config tracking yourself.

For a practical comparison and examples, see [Unified Training Script](../training/unified_training.md#hydra-lightning-vs-manual-lightning-without-hydra).

## Common starting points

**I have my own HDF5-like data and want to train quickly**:
- Start from config: [`configs/hdf5_core_readout.yaml`](https://github.com/open-retina/open-retina/blob/main/configs/hdf5_core_readout.yaml)
**I want a known-good multi-session training config**:
- Start from config: [`configs/hoefling_2024_core_readout_low_res.yaml`](https://github.com/open-retina/open-retina/blob/main/configs/hoefling_2024_core_readout_low_res.yaml)
**I want to create a new outer config for my own dataset**:
- Start from template: [`configs/template_outer_config.yaml`](https://github.com/open-retina/open-retina/blob/main/configs/template_outer_config.yaml)

For a detailed "which config to copy" walkthrough, see [Config templates](./config_templates.md).

## Additional examples

**Mix and match model and dataset configs**:
- See [Mix and match models and datasets](./mix_and_match_models_and_datasets.md) for a concrete cross-dataset example and compatibility caveats.
