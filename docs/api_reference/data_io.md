---
title: Data I/O API Reference
---

# Data I/O API Reference

The data_io package provides tools for loading, preprocessing, and managing retinal datasets. It includes base classes for creating custom dataloaders and implementations for specific published datasets.

## Overview

The data_io module is organized into:

- **[Base Data Classes](data_io/base.md)** — Core data containers (`MoviesTrainTestSplit`, `ResponsesTrainTestSplit`)
- **[Base Dataloader](data_io/base_dataloader.md)** — `MovieDataSet`, `MovieSampler`, and dataloader factories
- **[Artificial Stimuli](data_io/artificial_stimuli.md)** — Synthetic stimulus generation (chirp, moving bar)
- **[Cyclers](data_io/cyclers.md)** — Multi-dataloader cycling utilities

### Dataset Implementations

Each published dataset has its own subpackage with stimuli loading, response loading, and constants:

- **[Hoefling et al. 2024](data_io/hoefling_2024.md)** — Mouse retinal ganglion cell responses to natural stimuli. From [A chromatic feature detector in the retina signals visual context changes](https://doi.org/10.7554/eLife.86860) (eLife).
- **[Karamanlis et al. 2024](data_io/karamanlis_2024.md)** — Mouse and marmoset retinal ganglion cell responses to natural stimuli. From [Nonlinear receptive fields evoke redundant retinal coding of natural scenes](https://doi.org/10.1038/s41586-024-08212-3) (Nature).
- **[Goldin et al. 2022](data_io/goldin_2022.md)** — Mouse retinal ganglion cell responses under varying visual contexts. From [Context-dependent selectivity to natural images in the retina](https://doi.org/10.1038/s41467-022-33242-8) (Nature Communications).
- **[Maheswaranathan et al. 2023](data_io/maheswaranathan_2023.md)** — Primate and mouse retinal ganglion cell responses to natural scenes. From [Interpreting the retinal neural code for natural scenes: From computations to neurons](https://doi.org/10.1016/j.neuron.2023.06.007) (Neuron).
- **[Sridhar et al. 2025](data_io/sridhar_2025.md)** — Marmoset retinal ganglion cell responses to naturalistic movies and white noise. From [Modeling spatial contrast sensitivity in responses of primate retinal ganglion cells to natural movies](https://www.biorxiv.org/content/10.1101/2024.03.05.583449v1) (bioRxiv).
