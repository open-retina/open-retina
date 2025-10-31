---
title: API Reference
---

# API Reference

This section provides detailed documentation for all modules, classes, and functions in OpenRetina.

## Core Components

- [Models](./models.md): Pre-trained and customizable retina models
- [Modules](./modules.md): Building blocks for constructing models
- [Data I/O](./data_io.md): Data loading and processing utilities
- [In-silico](./insilico.md): Tools for analyzing models with virtual experiments
- [Utilities](./utils.md): Helper functions and visualization tools

## API Documentation by Module

The OpenRetina package is organized in a modular structure where each module serves a specific purpose in the retina modeling workflow.

### Models

The `models` module provides complete model implementations:

- [`openretina.models.core_readout`](./models/core_readout.md): End-to-end convolutional models with spatial readouts
- [`openretina.models.linear_nonlinear`](./models/linear_nonlinear.md): Classical linear-nonlinear cascade models
- [`openretina.models.sparse_autoencoder`](./models/sparse_autoencoder.md): Models for learning sparse representations

### Modules

The `modules` package contains building blocks for model construction. See the [modules overview](./modules.md) for complete documentation.

### Data I/O

The `data_io` package handles data loading and preprocessing:

- [`openretina.data_io.base_dataloader`](./data_io/base_dataloader.md): Abstract interfaces for data handling
- [`openretina.data_io.hoefling_2024`](./data_io/hoefling_2024.md): Dataloaders for Höfling et al. 2024 dataset
- [`openretina.data_io.artificial_stimuli`](./data_io/artificial_stimuli.md): Utilities for generating artificial stimuli
- [`openretina.data_io.cyclers`](./data_io/cyclers.md): Utilities for cycling through datasets

See the [data I/O overview](./data_io.md) for complete documentation.

### In-silico

The `insilico` package provides tools for analyzing models:

- [`openretina.insilico.stimulus_optimization`](./insilico/stimulus_optimization.md): Tools for finding optimal stimuli

See the [in-silico overview](./insilico.md) for complete documentation. 
