# Palmer et al., 2024

## Dataset at a glance

| Species | Recording technique | Stimulus type(s) | Stim channels | Total cells (quality filtered) | Original recording fs | Original stim fs | Test repeats | Train repeats | Saccade-like correction | Pixel size on retina [um] |
|---------|---------------------|------------------|---------------|--------------------------------|----------------------|------------------|--------------|---------------|-----------------------------------------------------------|---------------------------|
| Salamander     | multi-electrode array(MAEs) | Natural Videos          | 1 (greyscale)              | 93           | 60Hz                          | 60Hz (30Hz for movie 1 tree)                 | 83, 80, 84, 91, 85              | 1           | ???                                                       | ??? (different in train and test in this dataset)

## Dataset description
Neural data Voltage traces from the output, retinal ganglion cell layer of a larval tiger salamander retina were recorded following the methods outlined in O. Marre et al., Mapping a complete neural population in the retina. J. Neurosci. 32, 14859–14873 (2012). In brief, the retina was isolated in darkness and pressed against a 252-channel multielectrode array. Voltage recordings were taken during stimulus presentation of both natural movies and white noise stimuli and spike-sorted using an automated clustering algorithm that was hand-curated after initial template clustering and fits. This technique captured a highly overlapping neural population of 93 cells that fully tiled the recorded region of visual space. Spike times were binned at 16.667ms for all analyses presented.

Visual stimuli : (TRAIN) A white noise checkerboard stimulus (with binary white and black squares) was played at 30 frames per second (fps) for 30 minutes prior to and after the natural scene stimuli. (TEST) Five different natural movies lasting 20 seconds were played in a pseudorandom order, and each was displayed a minimum of 80 times. The movies labeled tree, water, grasses, fish, and self-motion were repeated 83, 80, 84, 91, and 85 times, respectively. All natural scenes except for the tree stimulus were displayed at 60fps. The tree stimulus was updated at a rate of 30fps with each frame repeated twice to match the 60fps frame rate of the other movies. 

## Unique characteristics
One of the first dataset with retinal responses to natural images.
Non mouse (salamander). 


## Citation information

Data: https://datadryad.org/dataset/doi:10.5061/dryad.4qrfj6qm8#methods [Unformatted]


## HuggingFace mirror

OpenRetina provides a mirror of the dataset on huggingface:
https://huggingface.co/datasets/open-retina/open-retina/tree/main/marre_lab/palmer_2024