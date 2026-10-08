---
title: core_readout
---

# Core-Readout Models

The [Models overview](../models.md) documents the public core-readout model classes and the helpers for loading pretrained checkpoints.

The components used to assemble these models are documented separately:

- [Core modules](../modules/core.md) extract shared spatiotemporal features.
- [Readout modules](../modules/readout.md) map those features to neuron responses for each recording session.
- [Custom layers](../modules/layers.md) provide convolutions, regularisers, scaling, and recurrent building blocks.
- [Loss functions](../modules/losses.md) provide the training and evaluation objectives.
