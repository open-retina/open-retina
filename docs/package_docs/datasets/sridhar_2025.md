# Sridhar et al., 2025

## Dataset at a glance

| Retinal piece | Species  | Recording technique           | Stimulus type(s)                      | Stim channels        | Total cells (not quality filtered) | Original recording fs | Original stim fs | Test repeats   | Train repeats | Saccade-like correction | Pixel size on retina [um]                       |
|---------------|----------|-------------------------------|---------------------------------------|----------------------|------------------------------------|-----------------------|------------------|----------------|---------------|-----------------------------------------------------------|-------------------------------------------------|
| Retina 01     | marmoset | multi-electrode arrays (MAEs) | white noise (WN) & natural movie (NM) | 1 for both WN and NM | 370                                | 85                    | 85               | WN: 11  NM: 10 | WN: 11 NM: 21 |                                                           | WN: 30  NM: 7.5 (downsampled to 30 used for training) |
| Retina 02     | marmoset | multi-electrode arrays (MAEs) | white noise (WN) & natural movie (NM) | 1 for both WN and NM | 181                                | 85                    | 85               | WN: 10  NM: 11 | WN: 10 NM: 11 | ...                                                       | WN: 30  NM: 7.5 (downsampled to 30 used for training)                                                |
| Retina 03     | marmoset | multi-electrode arrays (MAEs) | white noise (WN) & natural movie (NM) | 1 for both WN and NM | 495                                | 85                    | 85               | WN: 10  NM: 20 | WN: 10 NM: 20 | ...                                                       | WN: 30  NM: 7.5 (downsampled to 30 used for training)                                                |
| Retina 04     | marmoset | multi-electrode arrays (MAEs) | white noise (WN) & natural movie (NM) | 1 for both WN and NM | 110                                | 85                    | 85               | WN: 10  NM: 20 | WN: 10 NM: 20 | ...                                                       | WN: 30  NM: 7.5 (downsampled to 30 used for training)                                                |

## Dataset description

Spike train data from marmoset retinal ganglion cells under visual stimulation: The dataset contains spike times of 
different types of ganglion cells, recorded extracellularly with multielectrode arrays from 
**four isolated retinas of marmoset monkeys**.

The visual stimuli shown are spatio-temporal white noise and naturalistic stimuli. 
Each of these stimuli is stored as a separate dataset having its own dataset class within open-retina.

For the natural movie dat is the `MarmosetMovieDataset` class in `openretina/data_io/sridhar_2025/dataloaders.py`.  
For white noise dataloader is the `NoiseDataset` class in `openretina/data_io/sridhar_2025/dataloaders.py`.

### Natural movie dataset description
The naturalistic movie stimulus was derived from the film [Tears of Steel](https://mango.blender.org/about/) by 
Blender Studios. It was processed to grayscale and projected at a resolution of 1600x1200 pixels. 
Fro training models, it is by default cropped to 800x600 pixels and downsample to 200x150, matching 
the white-noise dataset resolution. 
Jittering was introduced to the movie frames to mimic the statistics of active vision in marmosets, including saccades, 
fixations, and fixational eye movements.

#### Dataset structure
This dataset is organized into five main directories containing different types of data:
* **`fixations`**: Contains text files with eye fixation data, including original and 4x downsampled 
versions for different seeds. The `complete_fixations_seed_2022and2023_s4_fixed.txt` file is used as the default for 
model training.
* **`responses`**: Stores `.pkl` (pickle) files containing recorded cell responses to fixation movies.
* **`stas`**: Holds `.npy` (numpy) files for Spike-Triggered Averages (STAs), with files indexed 
* by both the retinal piece dataset index (e.g., 01, 02) and individual cell index (e.g., cell_0 to cell_370).
* **`stimuli_padded`**: Contains the original padded stimulus images as `.npy` files.
* **`stimuli_padded_4`**: Contains the 4x downsampled version of the padded stimulus images as `.npy` files, corresponding to the fixations in files like `complete_fixations_seed_2022and2023_s4_fixed.txt`.

```angular2html

nm_marmoset_data
├── fixations
│   ├── complete_fixations_seed_2022and2023_s4_fixed.txt (Default for complete model training, 4x downsampled)
│   ├── complete_fixations_seed_2022.txt (Original fixation file for seed 2022)
│   ├── complete_fixations_seed_2023.txt (Original fixation file for seed 2023)
│   ├── complete_fixations_seed_2022_s4.txt (Seed 2022 fixations for 4x downsampled stimulus)
│   └── complete_fixations_seed_2023_s4.txt (Seed 2023 fixations for 4x downsampled stimulus)
├── responses
│   ├── cell_responses_01_fixation_movie.pkl
│   ├── cell_responses_02_fixation_movie.pkl
│   ├── cell_responses_03_fixation_movie.pkl
│   └── cell_responses_04_fixation_movie.pkl
├── stas
│   ├── cell_data_01_WN_stas_cell_0.npy
│   ├── cell_data_01_WN_stas_cell_1.npy
│   ├── ...
│   └── cell_data_01_WN_stas_cell_370.npy
│   ├── cell_data_02_WN_stas_cell_0.npy
│   └── ...
├── stimuli_padded
│   ├── 00000_img_10088.npy
│   ├── 00001_img_10089.npy
│   ├── ...
│   └── 14543_img_14758.npy
└── stimuli_padded_4
    ├── 00000_img_10088.npy
    ├── 00001_img_10089.npy
    ├── ...
    └── 14543_img_14758.npy
```
### White noise dataset description

The white noise stimulus consists of a sequence of frames of black and white squares (100% Michelson contrast) arranged in a uniform checkerboard pattern.

Spatial layout: 150x200 stimulus squares, with each square spanning 30 µm x 30 µm on the retinal surface.

#### Dataset structure
```
wn_marmoset_data
├── responses
│   ├── cell_responses_01_wn.pkl
│   ├── cell_responses_02_wn.pkl
│   ├── cell_responses_03_wn.pkl
│   └── cell_responses_04_wn.pkl
├── stas
│   ├── cell_data_01_WN_stas_cell_0.npy
│   ├── cell_data_01_WN_stas_cell_1.npy
│   ├── ...
│   └── cell_data_01_WN_stas_cell_370.npy
│   ├── cell_data_02_WN_stas_cell_0.npy
│   └── ...
├── non_repeating_stimuli_1 # white noise stimuli shown to Retina 01, 02 and 03
│   ├── trial_000
│       └── all_images.npy
│   ├── ...
│   └── trial_024.npy
|       └── all_images.npy
├── non_repeating_stimuli_2 # white noise stimuli shown to Retina 04
│   ├── trial_000
│       └── all_images.npy
│   ├── ...
│   └── trial_024.npy
|       └── all_images.npy
├── repeating_stimuli_1  # white noise stimuli shown to Retina 01, 02 and 03
│       └── all_images.npy
└── repeating_stimuli_2
        └── all_images.npy # white noise stimuli shown to Retina 04
```

## Unique characteristics

These datasets were collected to investigate how spatial contrast (variations in light intensity) inside receptive fields 
affects the encoding of visual stimuli by retinal ganglion cells. It specifically aims to address the limitations of
linear receptive field models by providing data suitable for testing the Spatial Contrast (SC) model under both 
artificial and naturalistic conditions.


### How is it different from other datasets?

**Primate Model:** The data is recorded from the marmoset (Callithrix jacchus) retina, offering insights into primate 
visual processing that differs from common salamander or mouse models.

**Fixation Simulation:** The natural movie stimuli in this 
dataset include simulated eye movements (saccades, fixations) modeled based on marmoset viewing statistics.



### What features of it make it nice for modelling?


* **Prediction of responses to dynamic natural stimuli with active vision statistics**
    The naturalistic stimuli in this dataset are overlaid with simulated eye movements (saccades, fixations, and drift) 
    specific to the marmoset visual system.
    This allows modelers to test how well architectures capture responses to the dynamic spatiotemporal statistics. 
    The high temporal resolution (85 Hz) and fine spatial resolution (7.5 µm/pixel in the case of natural movie data) enable the modeling of precise 
    spike timing and fine spatial integration features.


* **Benchmarking model generalization and adaptation**
    The dataset’s structure containing paired recordings of the *same* cells responding to both white noise and natural 
    movies, makes it a great dataset for testing **out-of-domain model generalization**.


* **Existing Benchmarks:** Extensive benchmarking has already been performed on this data, systematically comparing 
     Linear-Nonlinear (LN) models with various regularizers against Convolutional Neural Networks (CNNs) 
     of varying depths [(Vystrčilová et al. 2025)](https://www.biorxiv.org/content/10.1101/2024.03.06.583740v2.article-metrics) including their generalization capabilities.


* **Open-Retina Integration:** These specific benchmarking tasks (e.g., training on White Noise vs. Natural Movies) and 
baseline models (LN vs. CNN) from [(Vystrčilová et al. 2025)](https://www.biorxiv.org/content/10.1101/2024.03.06.583740v2.article-metrics)
 are integrated into `open-retina`, providing a simple way to benchmark new architectures against 
established state-of-the-art performance.


* **Studying the primate visual system**
    Most open retinal datasets come from salamanders or mice. This dataset offers rare access
to **primate retinal ganglion cells**, which are phylogenetically closer to humans. 
It includes reliably classified primate-specific cell types, including **OFF Midget, OFF Parasol, and ON Parasol cells**.
This allows modelers to investigate distinct functional roles.


## Citation information

[Original collecting paper](https://www.biorxiv.org/content/10.1101/2024.03.05.583449v2):
Sridhar, S., Vystrčilová, M., Khani, M. H., Karamanlis, D., Schreyer, H. M., Ramakrishna, V., Krüppel, S., Zapp, 
S. J., Mietsch, M., Ecker, A. S., & Gollisch, T. Modeling spatial contrast sensitivity in responses of primate 
retinal ganglion cells to natural movies. biorxiv, 2025

[Original source for dataset](https://doi.gin.g-node.org/10.12751/g-node.3dfiti/): Sridhar, S., Gollisch, T. Dataset - Marmoset retinal ganglion cell responses to naturalistic movies and 
spatiotemporal white noise. (2025) DOI: https://doi.gin.g-node.org/10.12751/g-node.3dfiti/

## HuggingFace mirror

[Natural movie dataset](https://huggingface.co/datasets/open-retina/nm_marmoset_data)  
[White noise dataset](https://huggingface.co/datasets/open-retina/wn_marmoset_data)

## Manuscripts using data from these datasets: 
* Sridhar, S., Vystrčilová, M., Khani, M. H., Karamanlis, D., Schreyer, H. M., Ramakrishna, V., Krüppel, S., Zapp, 
S. J., Mietsch, M., Ecker, A. S., & Gollisch, T. Modeling spatial contrast sensitivity in responses of primate 
retinal ganglion cells to natural movies. biorxiv, 2025. [URL](https://www.biorxiv.org/content/10.1101/2024.03.05.583449v2)
* Vystrčilová, Sridhar, S., Burg, M. F., M., Khani, M. H., Karamanlis, D., Schreyer, H. M., Ramakrishna, V., Krüppel, S., Zapp, 
S. J., Mietsch, Gollisch, T. & Ecker, A. S. A systematic comparison of predictive models on the retina. biorxiv, 2025. 
[URL](https://www.biorxiv.org/content/10.1101/2024.03.06.583740v2)
*  Vystrčilová M., Sridhar, S., Burg, M. F., M., Khani, M. H., Karamanlis, D., Schreyer, H. M., Ramakrishna, V., Krüppel, S., Zapp, 
S. J., Mietsch, Gollisch, T. & Ecker, A. S. Spatial Adaptation of Primate Retinal Ganglion Cells between Artificial and Natural Stimuli. 
biorxiv, 2025.
[URL](https://www.biorxiv.org/content/10.1101/2025.04.09.647910v2)
* Vystrčilová M., Sridhar, S., Burg, M. F., Gollisch, T. & Ecker, A. S. Interpreting convolutional neural networks to study
wide-field amacrine cell inhibition in the retina. NeurIPS 2025 UniReps workshop.
[URL](https://openreview.net/pdf?id=e7HeJezGbp)
* Nellen, N. S., Turishcheva, P., Vystrčilová M., Sridhar S., Gollisch T., Tolias A. S., Ecker A. S. Learning to cluster neuronal function.
NeurIPS 2025. [URL](https://openreview.net/pdf?id=Eufm2Jmjod)