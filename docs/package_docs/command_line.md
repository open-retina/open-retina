# Command Line Interface

After installing `openretina`, you can inspect all available subcommands with:

```bash
openretina --help
```

## Create synthetic test data

```bash
# Show help
openretina create-data --help

# Create synthetic data under ./test_data
openretina create-data ./test_data --num-colors 3 --num-stimuli 4 --num-sessions 2
```

And use this artificial data to train a new model from scratch  using the provided configs (make sure you have also cloned the  Github repository for this). When working with your own dataset, be sure to adjust the number of colour channels and the video dimensions accordingly and specify the names of the stimuli to be used for testing.

## Train a model

### Train with local HDF5-style data

```bash
openretina train --config-path configs --config-name hdf5_core_readout \
  paths.data_dir="test_data" \
  data_io.test_names="[random_noise_2, random_noise_3]" \
  data_io.color_channels=3 \
  data_io.video_height=16 \
  data_io.video_width=8
```

### Train with a built-in dataset config

```bash
openretina train --config-path configs --config-name hoefling_2024_core_readout_low_res
```

## Evaluate a model

Use `openretina eval` to run the evaluation pipeline for one split (`test` by default):

```bash
openretina eval --config-path configs --config-name karamanlis_2024_eval \
  evaluation.model_path=karamanlis_2024_base_mouse
```

Where the model path can either be the model tag of one of the models stored in the `openretina` huggingface, or a path to a local checkpoint you have trained.
Similarly, the config path and name can be set to the local configs you have used for training.

You can also evaluate on a different split by overriding:

```bash
openretina eval --config-path configs --config-name karamanlis_2024_eval \
  evaluation.model_path=karamanlis_2024_base_mouse \
  evaluation.data_split=validation
```

For split semantics and multi-test dataloader behavior, see [Data IO flow and multi-test support](./data_io_flow.md).

## Visualize model neurons

The model path can be either a local checkpoint path or a Hugging Face model identifier:

```bash
# Show visualization options
openretina visualize --help

# Download and visualize a pretrained model
openretina visualize --model-path hoefling_2024_base_low_res --save-folder visualizations

# Visualize original Hoefling et al. (2024) ensemble model
openretina visualize --is-hoefling-ensemble-model --model-id 0 --save-folder vis_ensemble_0
```
