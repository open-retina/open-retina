# Goldin et al., 2022

## Dataset at a glance

| Species | Recording technique | Stimulus type(s) | Stim channels | Total cells (quality filtered) | Original recording fs | Original stim fs | Test repeats | Train repeats | Saccade-like correction | Pixel size on retina [um] |
|---------|---------------------|------------------|---------------|--------------------------------|----------------------|------------------|--------------|---------------|-----------------------------------------------------------|---------------------------|
| Mouse     | multi-electrode array(MAEs) | Flashed Natural Images          | 1 (greyscale)              | 40           | (3.33Hz) One image every 300ms                            | 3.33Hz                  | 30              | 1           | None                                                       | 28
| Axoltl     | multi-electrode array(MAEs) | Flashed Natural Images          | 1 (greyscale)              | 48           | (3.33Hz) One image every 300ms                            | 3.33Hz                  | 20              | 1           | None                                                       | 28                       |

## Dataset description
Response to flashed images from the Van Hateren dataset (http://bethgelab.org/datasets/vanhateren/.).
All images are shown for 300ms and precedeed by 300ms of grey (~3000 images).  
Some images are repeated and form the test set (30 images).
The 'stimulus' section of the dataset contained only the image themselves.
The reported firing rate are sumemd over the presentation of the entire image (300 ms).

## Unique characteristics
This data was collected to train and test 2 layers CNN model of RGCs on flashed natural images. 
The model were then used to predict Local Spike Trigger Average (LSTA) of the model cells.
It's a good dataset to build model of response to flashed natural images in two different species.


## Citation information

Paper: https://doi.org/10.1038/s41467-022-33242-8
Data: https://zenodo.org/record/6868362#.YtgeLoxBxH4 [Unformatted]


## HuggingFace mirror

OpenRetina provides a mirror of the dataset on huggingface:
https://huggingface.co/datasets/open-retina/open-retina/tree/main/marre_lab/goldin_2022