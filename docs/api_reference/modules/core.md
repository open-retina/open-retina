---
title: Core Modules
---

# Core Modules

Feature extraction modules that process spatio-temporal visual input into learned representations.

## SimpleCoreWrapper

The primary convolutional core used in most models. Stacks spatio-temporal separable Conv3D layers with
configurable regularization (Laplace, group sparsity, temporal smoothness).

::: openretina.modules.core.base_core.SimpleCoreWrapper
    options:
        show_root_heading: true
        members:
            - __init__
            - forward
            - regularizer
            - save_weight_visualizations

## DummyCore

::: openretina.modules.core.base_core.DummyCore
    options:
        show_root_heading: true

## ConvGRUCore

A recurrent core using convolutional GRU cells for temporal processing.

::: openretina.modules.core.gru_core.ConvGRUCore
    options:
        show_root_heading: true
        members:
            - __init__
            - forward
            - regularizer
