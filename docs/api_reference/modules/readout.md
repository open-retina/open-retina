---
title: Readout Modules
---

# Readout Modules

Modules that map core feature representations to individual neuron response predictions.

## Base Readout

::: openretina.modules.readout.base.Readout
    options:
        show_root_heading: true
        members:
            - initialize
            - regularizer
            - apply_reduction
            - initialize_bias

## PointGaussianReadout

Spatial readout using learned Gaussian-sampled grid positions.

::: openretina.modules.readout.gaussian.PointGaussianReadout
    options:
        show_root_heading: true
        members:
            - __init__
            - forward
            - sample_grid
            - regularizer

## GaussianMaskReadout

Readout using a factorized Gaussian mask over spatial feature maps.

::: openretina.modules.readout.factorized_gaussian.GaussianMaskReadout
    options:
        show_root_heading: true
        members:
            - __init__
            - forward
            - regularizer

## FactorizedReadout

::: openretina.modules.readout.factorized.FactorizedReadout
    options:
        show_root_heading: true
        members:
            - __init__
            - forward
            - regularizer

## LNPReadout

::: openretina.modules.readout.linear_nonlinear_poison.LNPReadout
    options:
        show_root_heading: true
        members:
            - __init__
            - forward
            - regularizer

## Multi-Session Readouts

Wrappers that manage one readout instance per recording session.

::: openretina.modules.readout.multi_readout.MultiReadoutBase
    options:
        show_root_heading: true
        members:
            - __init__
            - forward
            - add_sessions
            - regularizer

::: openretina.modules.readout.multi_readout.MultiGaussianMaskReadout
    options:
        show_root_heading: true
        show_bases: true

::: openretina.modules.readout.multi_readout.MultiFactorizedReadout
    options:
        show_root_heading: true
        show_bases: true

::: openretina.modules.readout.multi_readout.MultiSampledGaussianReadout
    options:
        show_root_heading: true
        show_bases: true

::: openretina.modules.readout.multi_readout.MultipleLNPReadout
    options:
        show_root_heading: true
        show_bases: true
