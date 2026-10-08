---
title: Layers
---

# Custom Layers

Specialized neural network layers used as building blocks in core and readout modules.

## Convolution Layers

### STSeparableBatchConv3d

Spatio-temporal separable 3D convolution with batch-indexed temporal kernels.

::: openretina.modules.layers.convolutions.STSeparableBatchConv3d
    options:
        show_root_heading: true
        members:
            - __init__
            - forward

::: openretina.modules.layers.convolutions.TorchFullConv3D
    options:
        show_root_heading: true

::: openretina.modules.layers.convolutions.TorchSTSeparableConv3D
    options:
        show_root_heading: true

::: openretina.modules.layers.convolutions.compute_temporal_kernel
    options:
        show_root_heading: true

## Regularizers

::: openretina.modules.layers.regularizers.Laplace
    options:
        show_root_heading: true

::: openretina.modules.layers.regularizers.LaplaceL2norm
    options:
        show_root_heading: true

::: openretina.modules.layers.regularizers.FlatLaplaceL23dnorm
    options:
        show_root_heading: true

::: openretina.modules.layers.regularizers.GaussianLaplaceL2
    options:
        show_root_heading: true

## Scaling Layers

::: openretina.modules.layers.scaling.Bias3DLayer
    options:
        show_root_heading: true

::: openretina.modules.layers.scaling.Scale2DLayer
    options:
        show_root_heading: true

::: openretina.modules.layers.scaling.FiLM
    options:
        show_root_heading: true

## GRU Layers

::: openretina.modules.layers.gru.ConvGRUCell
    options:
        show_root_heading: true

::: openretina.modules.layers.gru.GRU_Module
    options:
        show_root_heading: true

## Ensemble

::: openretina.modules.layers.ensemble.EnsembleModel
    options:
        show_root_heading: true

## Reducers

::: openretina.modules.layers.reducers.WeightedChannelSumLayer
    options:
        show_root_heading: true
