---
title: Modules API Reference
---

# Modules API Reference

The modules package contains the building blocks for constructing retinal models in OpenRetina. These modular components can be combined to create custom neural network architectures.

## Overview

The modules are organized into four categories:

- **[Core Modules](modules/core.md)** — Convolutional feature extractors that process spatio-temporal visual input
- **[Readout Modules](modules/readout.md)** — Components that map core features to individual neuron responses
- **[Layers](modules/layers.md)** — Custom neural network layers (convolutions, regularizers, scaling, GRU)
- **[Loss Functions](modules/losses.md)** — Loss functions specialized for neural response prediction (Poisson, correlation, MSE)
