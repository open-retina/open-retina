---
title: Models API Reference
---

# Models API Reference

Complete neural network architectures for retinal response prediction. All models follow
the **Core + Readout** pattern: a shared feature extraction core paired with per-session readouts.

## Loading Pre-trained Models

::: openretina.models.core_readout.load_core_readout_from_remote
    options:
        show_root_heading: true

::: openretina.models.core_readout.load_core_readout_model
    options:
        show_root_heading: true

## BaseCoreReadout

::: openretina.models.core_readout.BaseCoreReadout
    options:
        show_root_heading: true
        members:
            - __init__
            - forward
            - training_step
            - validation_step
            - test_step
            - configure_optimizers
            - save_weight_visualizations
            - compute_readout_input_shape
            - stimulus_shape
            - update_model_data_info

## UnifiedCoreReadout

::: openretina.models.core_readout.UnifiedCoreReadout
    options:
        show_root_heading: true
        members:
            - __init__
            - configure_optimizers

## ExampleCoreReadout

::: openretina.models.core_readout.ExampleCoreReadout
    options:
        show_root_heading: true

## Sub-modules

- [Core-Readout Models](models/core_readout.md) — Full module reference
- [Linear-Nonlinear Models](models/linear_nonlinear.md) — LNP cascade models
- [Sparse Autoencoder](models/sparse_autoencoder.md) — Sparse representation models

