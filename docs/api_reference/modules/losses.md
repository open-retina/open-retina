---
title: Loss Functions
---

# Loss Functions

Loss functions for training retinal models. All losses handle temporal lag between predictions and targets
automatically (the model may produce fewer time steps than the target due to temporal convolutions).

## Poisson Losses

::: openretina.modules.losses.poisson.PoissonLoss3d
    options:
        show_root_heading: true

::: openretina.modules.losses.poisson.L1PoissonLoss3d
    options:
        show_root_heading: true

::: openretina.modules.losses.poisson.CelltypePoissonLoss3d
    options:
        show_root_heading: true

## Correlation Losses

::: openretina.modules.losses.correlation.CorrelationLoss3d
    options:
        show_root_heading: true

::: openretina.modules.losses.correlation.CelltypeCorrelationLoss3d
    options:
        show_root_heading: true

::: openretina.modules.losses.correlation.ScaledCorrelationLoss3d
    options:
        show_root_heading: true

## MSE Loss

::: openretina.modules.losses.mse.MSE3d
    options:
        show_root_heading: true
