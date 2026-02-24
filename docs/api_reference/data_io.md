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

- **[Hoefling et al. 2024](data_io/hoefling_2024.md)** — Mouse RGC responses to natural stimuli (eLife)
- **[Karamanlis et al. 2024](data_io/karamanlis_2024.md)** — Mouse + marmoset nonlinear receptive fields (Nature)
- **[Goldin et al. 2022](data_io/goldin_2022.md)** — Context-dependent selectivity (Nature Communications)
- **[Maheswaranathan et al. 2023](data_io/maheswaranathan_2023.md)** — Natural scene interpretation (Neuron)
- **[Sridhar et al. 2025](data_io/sridhar_2025.md)** — Marmoset spatial contrast sensitivity (bioRxiv)
