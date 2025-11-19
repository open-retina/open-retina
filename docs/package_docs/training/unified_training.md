# Unified Training Script

`openretina` ships a single training entry point that works across datasets and model variants. The CLI command `openretina train` (backed by `openretina.cli.train.py`) combines [Hydra](https://hydra.cc/){:target="\_blank"} for configuration management and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/){:target="\_blank"} for training, checkpointing, and logging. With the right configuration files in place, you can launch a full experiment from the terminal with one command.

If you need a refresher on how data is prepared, start with the [Data I/O guide](../data_io/index.md). To see how trained models are evaluated, visit the [training overview](./index.md) and the [core + readout walkthrough](./core_readout.md).

## Configuration Layout

Hydra looks for configuration files under the `configs/` directory. Each subfolder groups related settings so you can mix and match them per experiment:

* `data_io/` and `dataloader/`: dataset-specific paths, preprocessing steps, and batching parameters. Add your own YAML files here when introducing a new dataset.
* `model/`: reusable model definitions (for example `base_core_readout.yaml`).
* `trainer/`: defaults for the Lightning trainer (epochs, devices, checkpointing, etc.).
* `logger/`: logging backends such as CSV, TensorBoard, or Weights & Biases.
* `training_callbacks/`: optional callbacks like early stopping or learning-rate schedulers.
* `quality_checks/`: extra validation steps after loading data.
* root-level YAML files such as `hoefling_2024_core_readout_high_res.yaml`: “outer configs” that compose the pieces above into a concrete experiment.

```
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

If you followed the HDF5 walkthrough in the [data guide](../data_io/index.md#hdf5-based-data-structure), you can train the bundled core + readout model on that synthetic data with:

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

* **Consistency**: experiments share the same entry point, making it easy to reproduce results.
* **Modularity**: mix and match datasets, models, and trainers by reusing small YAML files.
* **Automation-ready**: CLI overrides fit naturally with grid search tools and schedulers.
* **Documentation and support**: workflows align with the rest of the training docs and the [Hydra tutorial in the training guide](../tutorials/training.md#configuring-training-with-hydra).

For deeper coverage of Hydra itself, refer to the [official documentation](https://hydra.cc/docs/intro/).
