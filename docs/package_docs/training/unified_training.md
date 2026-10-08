# Unified Training Script

`openretina` ships a single training entry point that works across datasets and model variants. The CLI command `openretina train` (backed by `openretina.cli.train.py`) combines [Hydra](https://hydra.cc/){:target="\_blank"} for configuration management and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/){:target="\_blank"} for training, checkpointing, and logging. With the right configuration files in place, you can launch a full experiment from the terminal with one command.

If you need a refresher on how data is prepared, start with the [Data I/O guide](../data_io.md). To see how trained models are evaluated, visit the [training overview](./index.md) and the [core + readout walkthrough](./core_readout.md).

## Why Hydra + Lightning?

OpenRetina uses Hydra and Lightning together for a practical reason:

- **Hydra** keeps experiment definitions modular (`data_io`, `dataloader`, `model`, `trainer`, `logger`), so you can reuse components instead of duplicating YAML files.
- **Lightning** handles the training lifecycle (fit/test loops, checkpointing, multi-device setup, precision settings, logging hooks).
- **Together**, they make runs easier to reproduce and easier to hand over to collaborators.

## Hydra + Lightning vs manual Lightning (without Hydra)

Choose the workflow based on your task:

- **Use Hydra + Lightning (recommended default)** when you need reproducible runs, config overrides from CLI, or frequent dataset/model swaps.
- **Use manual Lightning** when you are prototyping quickly in Python and want to avoid config composition overhead.

In both cases, the model and dataloader interfaces are the same; Hydra mainly changes how those objects are constructed.

## Configuration Layout

Hydra looks for configuration files under the `configs/` directory. Each subfolder groups related settings so you can mix and match them per experiment:

- `data_io/`: dataset loading functions and dataset-specific metadata.
- `dataloader/`: clip extraction and batching behavior.
- `model/`: reusable model definitions (for example `base_core_readout.yaml`).
- `trainer/`: defaults for the Lightning trainer (epochs, devices, checkpointing, etc.).
- `logger/`: logging backends such as CSV, TensorBoard, or Weights & Biases.
- `training_callbacks/`: optional callbacks like early stopping or learning-rate schedulers.
- `quality_checks/`: extra validation steps after loading data.
- root-level YAML files such as `hoefling_2024_core_readout_high_res.yaml`: "outer configs" that compose the pieces above into a concrete experiment.

```text
configs/
├── data_io/
│   ├── hoefling_2024.yaml
│   └── maheswaranathan_2023.yaml
├── dataloader/
├── hydra/
├── logger/
├── model/
│   └── base_core_readout.yaml
├── quality_checks/
├── trainer/
├── training_callbacks/
├── hoefling_2024_core_readout_high_res.yaml
└── maheswaranathan_2023_core_readout.yaml
```

Hydra merges the selected files so you can override defaults without copying entire configs.

In practice, the common source of confusion is why `data_io` and `dataloader` are separate. The short answer:

- `data_io` decides **what data to load** and dataset semantics.
- `dataloader` decides **how that data is chunked/batched** for training.

## Running the Script

1. **Add dataset configs**: create YAML files in `data_io/` and `dataloader/` that describe where stimuli and responses live and how they should be batched.
2. **Create an outer config**: place a YAML file in `configs/` that sets high-level settings (dataset choice, model, trainer, logging). This file references the nested configs created above.
3. **Launch training**: call `openretina train --config-name <outer_config>` from the project root (or any location if the package is installed). Hydra resolves the configuration graph and Lightning starts training.

Example run with a custom experiment name:

```bash
openretina train --config-name your_outer_config_name \
  exp_name=my_first_openretina_run
```

### Example: HDF5 Tutorial Dataset

If you followed the HDF5 walkthrough in the [data guide](../data_io.md#hdf5-based-data-structure), you can train the bundled core + readout model on that synthetic data with:

```bash
openretina train --config-path configs --config-name hdf5_core_readout \
  paths.data_dir="test_data" \
  data_io.test_names="[random_noise_2, random_noise_3]" \
  data_io.color_channels=3 \
  data_io.video_height=16 \
  data_io.video_width=8
```

Any value in the configuration tree can be overridden from the command line in the same fashion, making it easy to sweep hyperparameters or swap modules without editing files.

## Why Use the Unified Script?

- **Consistency**: experiments share the same entry point, making it easy to reproduce results.
- **Modularity**: mix and match datasets, models, and trainers by reusing small YAML files.
- **Automation-ready**: CLI overrides fit naturally with grid search tools and schedulers.
- **Documentation and support**: workflows align with the rest of the training docs and the [config template guide](../how_to/config_templates.md).

## Related pages

- [How to...](../how_to/index.md)
- [Which config should I copy?](../how_to/config_templates.md)
- [Manual training and evaluation](./training_and_evaluation.md)
- [Data IO flow and multi-test support](../data_io_flow.md)

For deeper coverage of Hydra itself, refer to the [official documentation](https://hydra.cc/docs/intro/).
