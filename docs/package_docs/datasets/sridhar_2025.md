# Sridhar et al., 2025

## Dataset at a glance

| Retinal piece | Species  | Recording technique           | Stimulus type(s)                      | Stim channels        | Total cells (not quality filtered) | Original recording fs | Original stim fs | Test repeats   | Train repeats | Saccade-like correction | Pixel size on retina [um] |
|---------------|----------|-------------------------------|---------------------------------------|----------------------|------------------------------------|-----------------------|------------------|----------------|---------------|-----------------------------------------------------------|---------------------------|
| Retina 01     | marmoset | multi-electrode arrays (MAEs) | white noise (WN) & natural movie (NM) | 1 for both WN and NM | 370                                | 85                    | 85               | WN: 11  NM: 10 | WN: 11 NM: 21 |                                                           |                           |
| Retina 02     | marmoset | multi-electrode arrays (MAEs) | white noise (WN) & natural movie (NM) | 1 for both WN and NM | 181                                | 85                    | 85               | WN: 10  NM: 11 | WN: 10 NM: 11 | ...                                                       | ...                       |
| Retina 03     | marmoset | multi-electrode arrays (MAEs) | white noise (WN) & natural movie (NM) | 1 for both WN and NM | 495                                | 85                    | 85               | WN: 10  NM: 20 | WN: 10 NM: 20 | ...                                                       | ...                       |
| Retina 04     | marmoset | multi-electrode arrays (MAEs) | white noise (WN) & natural movie (NM) | 1 for both WN and NM | 110                                | 85                    | 85               | WN: 10  NM: 20 | WN: 10 NM: 20 | ...                                                       | ...                       |

## Dataset description

Spike train data from marmoset retinal ganglion cells under visual stimulation: The dataset contains spike times of different types of ganglion cells, recorded extracellularly with multielectrode arrays from **four isolated retinas of marmoset monkeys**.

The visual stimuli shown are spatio-temporal white noise and naturalistic stimuli. Each of these stimuli is stored as a separate dataset having its own dataloader within open-retina.

### White noise dataset description

#### Dataset structure
```angular2html

```
### Natural movie dataset description


## Unique characteristics
Why was this dataset collected?

How is it different from other datasets?

What features of it make it nice for modelling? For modelling which aspects?


## Citation information

Original collecting paper

Original source for dataset

## HuggingFace mirror
