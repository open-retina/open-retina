# Karamanlis et al., 2024

This dataset contains mouse and marmoset retinal ganglion cell responses to natural scenes.

## Dataset at a glance

| Species | Recording technique | Stimulus type(s) | Stim channels | Total cells (quality filtered) | Original recording fs | Original stim fs | Test repeats | Train repeats | Saccade-like correction | Pixel size on retina [µm] |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- | ---: |
| Mouse & Marmoset | MEA | Jittered natural scenes + WN | Greyscale | ≈1,800 (mouse), 42 (marmoset) |  |  | 18–55 |  |  | 7.5 |

The values filled into the previously empty cells are from Table 1 of the [OpenRetina preprint](https://doi.org/10.1101/2025.03.07.642012). WN denotes white noise.

## Resources

- **Paper:** [Nonlinear receptive fields evoke redundant retinal coding of natural scenes](https://doi.org/10.1038/s41586-024-08212-3)
- **Original dataset:** [doi.org/10.12751/g-node.ejk8kx](https://doi.org/10.12751/g-node.ejk8kx)
- **OpenRetina mirror:** [Hugging Face](https://huggingface.co/datasets/open-retina/open-retina/tree/main/gollisch_lab/karamanlis_2024)

## OpenRetina support

See the [Karamanlis dataset API reference](../../api_reference/data_io/karamanlis_2024.md) for the available stimulus and response loaders.
