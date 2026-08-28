# Hoefling et al., 2024

This dataset contains mouse retinal ganglion cell responses to natural visual stimuli. OpenRetina provides loaders for the natural-movie responses and associated artificial-stimulus recordings.

## Dataset at a glance

| Species | Recording technique | Stimulus type(s) | Stim channels | Total cells (quality filtered) | Original recording fs | Original stim fs | Test repeats | Train repeats | Saccade-like correction | Pixel size on retina [µm] |
| --- | --- | --- | --- | ---: | --- | --- | ---: | --- | --- | ---: |
| mouse | 2P imaging | Natural movie | 2 (UV / Green) | ≈3,000 |  |  | 3 |  |  | 12.5 |

Previously empty values for the cell count, test repeats, and pixel size are from Table 1 of the [OpenRetina preprint](https://doi.org/10.1101/2025.03.07.642012).

## Resources

- **Paper:** [A chromatic feature detector in the retina signals visual context changes](https://doi.org/10.7554/eLife.86860)
- **Original dataset:** [gin.g-node.org/eulerlab/rgc-natstim](https://gin.g-node.org/eulerlab/rgc-natstim)
- **OpenRetina mirror:** [Hugging Face](https://huggingface.co/datasets/open-retina/open-retina/tree/main/euler_lab/hoefling_2024)

## OpenRetina support

See the [Höfling dataset API reference](../../api_reference/data_io/hoefling_2024.md) for the available dataloaders, stimulus utilities, response processing, and constants.
