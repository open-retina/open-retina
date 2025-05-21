---
title: Architecture Overview
---

# OpenRetina Architecture

OpenRetina is designed with modularity in mind, allowing researchers to mix and match components while ensuring consistent interfaces between them.

## Core Components

The package is organized into the following main modules:

```
openretina/
├── modules/         # Building blocks for neural network models
├── models/          # Complete models ready for use
├── data_io/         # Data loading and processing
├── insilico/        # In-silico analysis tools
├── optimization/    # Model training and optimization utilities
└── utils/           # Helper functions
```

### Modules

The `modules` package contains the fundamental building blocks for constructing retina models:

- **Core modules** (`modules/core/`): Convolutional feature extractors that learn spatio-temporal filters
- **Readout modules** (`modules/readout/`): Components that map extracted features to neural responses
- **Layers** (`modules/layers/`): Custom PyTorch layers for retina modeling
- **Regularizers** (`modules/layers/regularizers.py`): Functions to enforce constraints during training

### Models

The `models` package provides complete, ready-to-use models:

- **Core-readout models** (`models/core_readout.py`): End-to-end models with convolutional cores and readout mechanisms
- **Linear-nonlinear models** (`models/linear_nonlinear.py`): Classical LN cascade models
- **Sparse autoencoders** (`models/sparse_autoencoder.py`): For learning sparse representations

### Data I/O

The `data_io` package handles loading and preprocessing retina datasets:

- **Base classes** (`data_io/base.py`, `data_io/base_dataloader.py`): Abstract interfaces for data handling
- **Dataset-specific loaders**: Modules for loading data from published studies (e.g., `data_io/hoefling_2024/`)
- **Artificial stimuli** (`data_io/artificial_stimuli.py`): Generation of synthetic visual stimuli

### In-silico Experiments

The `insilico` package contains tools for analyzing trained models:

- **Stimulus optimization** (`insilico/stimulus_optimization/`): Methods to find optimal stimuli for modeled neurons
- **Tuning analyses** (`insilico/tuning_analyses/`): Tools to characterize neural response properties

## Code Design Philosophy

OpenRetina follows these core design principles:

1. **Modularity**: Components can be used independently or combined in various ways
2. **PyTorch-based**: All models are built using PyTorch for efficient GPU utilization
3. **Training with Lightning**: PyTorch Lightning is used for training loops and multi-GPU support
4. **Configuration with Hydra**: Hydra manages complex configurations for experiments
5. **Reproducibility**: Pre-trained models are versioned and easily downloadable

## Typical Workflow

A typical workflow using OpenRetina involves:

1. **Loading data**: Using the appropriate data_io module
2. **Setting up a model**: Either using a pre-trained model or configuring a new one
3. **Training**: If using a custom dataset, training the model with optimization utilities
4. **Analysis**: Running in-silico experiments to understand the model's behavior

## Extending OpenRetina

OpenRetina can be extended by:

1. **Adding new core modules**: Implementing new feature extractors
2. **Creating new readout mechanisms**: For different neural response patterns
3. **Supporting new datasets**: Adding data loaders for your own data
4. **Implementing new analyses**: Adding custom in-silico experimental methods 
