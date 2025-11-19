After installing openretina, you can explore it's powerful command line interface:
```bash
# Show all avalaible commands
openretina --help
```

Using this interface, you can create artificial data:
```bash
# Print help message
openretina create-data --help
# Create data in the folder `./test_data`
openretina create-data ./test_data --num-colors 3 --num-stimuli 4 --num-sessions 2
```

And use this artificial data to train a new model from scratch using the provided configs (make sure you have also cloned the Github repository for this). When working with your own dataset, be sure to adjust the number of colour channels and the video dimensions accordingly and specify the names of the stimuli to be used for testing.
```bash
openretina train --config-path configs --config-name hdf5_core_readout \
    paths.data_dir="test_data" data_io.test_names="[random_noise_2, random_noise_3]" \
    data_io.color_channels=3 data_io.video_height=16 data_io.video_width=8
```

If you prefer to train a model using datasets that are already provided in openretina use a command like the following:
```bash
openretina train --config-path configs --config-name hoefling_2024_core_readout_low_res
```

Or visualize existing pretrained models, the model path can be either a path on your local drive or a name of a model that we uploaded to huggingface:
```bash
# Explore the options of the visualise tool
openretina visualize --help
# Download and visualise a pretrained model
openretina visualize --model-path hoefling_2024_base_low_res --save-folder visualizations
# We also support the original ensemble model of the paper Hoefling, Elife, 2024.
openretina visualize --is-hoefling-ensemble-model --model-id 0 --save-folder vis_ensemble_0
```
