# Palmer et al., 2024

## Dataset at a glance

| Species | Recording technique | Stimulus type(s) | Stim channels | Total cells (quality filtered) | Original recording fs | Original stim fs | Test repeats | Train repeats | Saccade-like correction | Pixel size on retina [µm] |
| --- | --- | --- | --- | ---: | --- | --- | --- | ---: | --- | --- |
| Salamander | multielectrode array (MEA) | Natural videos | 1 (greyscale) | 93 | 60 Hz | 60 Hz (30 Hz for movie 1, tree) | 83, 80, 84, 91, 85 | 1 |  |  |

## Dataset description
Voltage traces from the retinal ganglion cell layer of a larval tiger salamander retina were recorded following the methods outlined in O. Marre et al., *Mapping a complete neural population in the retina*, J. Neurosci. 32, 14859–14873 (2012). In brief, the retina was isolated in darkness and pressed against a 252-channel multielectrode array. Voltage recordings were taken during presentation of natural movies and white-noise stimuli, then spike-sorted using an automated clustering algorithm with manual curation. This technique captured an overlapping population of 93 cells that tiled the recorded region of visual space. Spike times were binned at 16.667 ms for the reported analyses.

For training, a binary white-noise checkerboard was presented at 30 frames per second (fps) for 30 minutes before and after the natural-scene stimuli. For testing, five 20-second natural movies were presented in pseudorandom order at least 80 times each. The movies labelled tree, water, grasses, fish, and self-motion were repeated 83, 80, 84, 91, and 85 times, respectively. All natural scenes except the tree stimulus were displayed at 60 fps. The tree stimulus was updated at 30 fps, with each frame repeated twice to match the 60 fps rate of the other movies.

## Unique characteristics
This is an early dataset of retinal responses to natural movies and provides recordings from salamander rather than mouse retina.


## Citation information

Data: [doi.org/10.5061/dryad.4qrfj6qm8](https://doi.org/10.5061/dryad.4qrfj6qm8)


## HuggingFace mirror

OpenRetina provides a mirror of the dataset on Hugging Face:
https://huggingface.co/datasets/open-retina/open-retina/tree/main/marre_lab/palmer_2024
