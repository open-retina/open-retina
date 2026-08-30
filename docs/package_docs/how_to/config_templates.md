# Which config should I copy?

If you are new to the OpenRetina config stack, start from one of these templates and modify only a few fields first.

## Recommended templates by use case

### 1) Quick local run with synthetic or local HDF5 data

- **Template**: [`configs/hdf5_core_readout.yaml`](https://github.com/open-retina/open-retina/blob/main/configs/hdf5_core_readout.yaml)
- **Use when**: you want a runnable baseline without implementing a custom dataset loader.
- **Typical command**:

```bash
openretina train --config-path configs --config-name hdf5_core_readout \
  paths.data_dir="test_data" \
  data_io.test_names="[random_noise_2, random_noise_3]" \
  data_io.color_channels=3 \
  data_io.video_height=16 \
  data_io.video_width=8
```

### 2) Built-in dataset training reference

- **Template**: [`configs/hoefling_2024_core_readout_low_res.yaml`](https://github.com/open-retina/open-retina/blob/main/configs/hoefling_2024_core_readout_low_res.yaml)
- **Use when**: you want a stable, real-data baseline in the existing framework.
- **Typical command**:

```bash
openretina train --config-path configs --config-name hoefling_2024_core_readout_low_res
```

### 3) New dataset onboarding (write your own outer config)

- **Template**: [`configs/template_outer_config.yaml`](https://github.com/open-retina/open-retina/blob/main/configs/template_outer_config.yaml)
- **Use when**: you are adding a new dataset and need to define new `data_io` and `dataloader` configs.
- **Tip**: first create `configs/data_io/<your_dataset>.yaml` and `configs/dataloader/<your_dataset>.yaml`, then reference them in the outer config defaults.

## Why are there multiple config files?

OpenRetina uses Hydra composition so each config layer has one responsibility:

- `data_io/*`: how to load stimuli/responses and dataset-specific metadata.
- `dataloader/*`: batching and clip construction (`batch_size`, `clip_length`, etc.).
- `model/*`: reusable model architecture defaults.
- top-level `configs/*.yaml` ("outer configs"): select and combine the pieces above for one experiment.

This separation is what lets you swap datasets, model definitions, and trainer settings without copying a single monolithic YAML file.

## Minimal override examples

These command-line overrides are often enough to start:

```bash
# Change experiment name
openretina train --config-name hoefling_2024_core_readout_low_res exp_name=my_run

# Change batch size from dataloader config
openretina train --config-name hoefling_2024_core_readout_low_res dataloader.batch_size=32

# Override trainer epochs
openretina train --config-name hoefling_2024_core_readout_low_res trainer.max_epochs=20
```

For a deeper explanation of how Hydra and Lightning interact in OpenRetina, see [Unified Training Script](../training/unified_training.md).
