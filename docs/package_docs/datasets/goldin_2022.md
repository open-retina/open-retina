# Goldin et al., 2022

## Dataset at a glance

| Species | Recording technique | Stimulus type(s) | Stim channels | Total cells (quality filtered) | Original recording fs | Original stim fs | Test repeats | Train repeats | Saccade-like correction | Pixel size on retina [µm] |
| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: |
| Mouse | multielectrode array (MEA) | Flashed natural images | 1 (greyscale) | 40 | 3.33 Hz (one image every 300 ms) | 3.33 Hz | 30 | 1 | None | 28 |
| Axolotl | multielectrode array (MEA) | Flashed natural images | 1 (greyscale) | 48 | 3.33 Hz (one image every 300 ms) | 3.33 Hz | 20 | 1 | None | 28 |

## Dataset description
Response to flashed images from the Van Hateren dataset.
All images are shown for 300 ms and preceded by 300 ms of grey (~3000 images).
Some images are repeated and form the test set (30 images).
The `stimulus` section of the dataset contains only the images themselves.
The reported firing rates are summed over the entire image presentation (300 ms).

## Unique characteristics
These data were collected to train and test two-layer CNN models of RGCs on flashed natural images.
The models were then used to predict local spike-triggered averages (LSTAs) of the model cells.
The dataset supports modelling responses to flashed natural images in two different species.


## Citation information

Paper: https://doi.org/10.1038/s41467-022-33242-8
Data: [Zenodo record 6868362](https://zenodo.org/records/6868362)


## HuggingFace mirror

OpenRetina provides a mirror of the dataset on Hugging Face:
https://huggingface.co/datasets/open-retina/open-retina/tree/main/marre_lab/goldin_2022
