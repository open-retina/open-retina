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

## ResidualAttentionCore

The stock first layer followed by inverted residual depthwise blocks and causal temporal self-attention
(per spatial position, sliding window, learned per-lag bias). The residual blocks follow the DwiseNeuro core of the
[Sensorium 2023 winning solution](https://github.com/lRomul/sensorium). Train it with
`configs/hoefling_2024_core_readout_low_res_residual_attention.yaml`, which uses long training chunks and weight EMA
(`openretina.callbacks.ema.EMACallback`).

::: openretina.modules.core.residual_attention_core.ResidualAttentionCore
    options:
        show_root_heading: true
        members:
            - __init__
            - forward
            - regularizer

::: openretina.modules.core.residual_attention_core.CausalTemporalAttention
    options:
        show_root_heading: true
