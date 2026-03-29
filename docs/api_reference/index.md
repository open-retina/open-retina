# API Reference

This is the API reference for OpenRetina. It documents all public classes, functions, and modules available in the package.

**How to use this reference:**

- **Browse by module**: Navigate through the sections below to find specific components
- **Search**: Use the search function to find specific classes or functions
- **Link to source**: Most entries link directly to the source code

If you're new to OpenRetina, start with the [Quick Start](../package_docs/quickstart.md) or [Tutorials](../package_docs/tutorials/pretrained_models.md) before diving into the API reference.

## Modules

### [Models](./models.md)

Complete model architectures for retinal response prediction:

- [`BaseCoreReadout`](./models.md#basecorereadout) — Base class for all core-readout models
- [`UnifiedCoreReadout`](./models.md#unifiedcorereadout) — Hydra-configurable model (recommended)
- [`load_core_readout_from_remote`](./models.md#loading-pre-trained-models) — Load pre-trained models
- [Linear-Nonlinear Models](./models/linear_nonlinear.md) — LNP cascade models
- [Sparse Autoencoder](./models/sparse_autoencoder.md) — Sparse representation models

### [Modules](./modules.md)

Building blocks for constructing models:

- [Core Modules](./modules/core.md) — Convolutional feature extractors (`SimpleCoreWrapper`, `ConvGRUCore`)
- [Readout Modules](./modules/readout.md) — Spatial readouts (`PointGaussianReadout`, `GaussianMaskReadout`, multi-session wrappers)
- [Layers](./modules/layers.md) — Convolutions, regularizers, scaling, GRU cells
- [Loss Functions](./modules/losses.md) — Poisson, correlation, and MSE losses

### [Data I/O](./data_io.md)

Data loading and preprocessing:

- [Base Data Classes](./data_io/base.md) — `MoviesTrainTestSplit`, `ResponsesTrainTestSplit`
- [Base Dataloader](./data_io/base_dataloader.md) — `MovieDataSet`, `MovieSampler`, dataloader factories
- Dataset implementations: [Hoefling 2024](./data_io/hoefling_2024.md), [Karamanlis 2024](./data_io/karamanlis_2024.md), [Goldin 2022](./data_io/goldin_2022.md), [Maheswaranathan 2023](./data_io/maheswaranathan_2023.md), [Sridhar 2025](./data_io/sridhar_2025.md)
- [Artificial Stimuli](./data_io/artificial_stimuli.md), [Cyclers](./data_io/cyclers.md)

### [In-silico](./insilico.md)

Tools for computational experiments with trained models:

- [Stimulus Optimization](./insilico/stimulus_optimization.md) — MEIs, discriminatory stimuli, regularizers
- [Vector Field Analysis](./insilico/vector_field_analysis.md) — PCA-based response analysis
- [Tuning Analyses](./insilico/tuning_analyses.md) — Gradient-based response characterization

### [Evaluation](./eval.md)

Metrics and oracle computations for model evaluation:

- Correlation, Poisson loss, MSE, FEVe, variance analysis
- Oracle correlations (jackknife, global mean)

### [CLI](./cli.md)

Command-line entry points:

- `train_model` — Full training pipeline
- `evaluate_model` — Full evaluation pipeline

### [Utilities](./utils.md)

Helper functions for file handling, visualization, model management, and data processing.
