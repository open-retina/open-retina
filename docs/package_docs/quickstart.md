---
title: Quick Start
---

TODO needs to be made the paper quickstart

# Quick Start Guide

This guide will help you get started with `openretina` quickly by:
1. Installing the package
2. Loading a pre-trained model
3. Running inference with the model
4. Visualizing the results

## Installation

Install `openretina` using pip:

```bash
pip install openretina
```

For development or to access example notebooks, clone the repository:

```bash
git clone git@github.com:open-retina/open-retina.git
cd open-retina
pip install -e .
```

## Loading a Pre-trained Model

`openretina` provides several pre-trained models. Here's how to load one:

```python
import torch
from openretina.models import load_core_readout_from_remote

# Load a pre-trained model (will download if not already cached)
model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")
```

## Running Inference

Generate some random input and run a forward pass:

```python
# Get the appropriate input shape for this model
batch_size = 1
time_steps = 50
input_shape = model.stimulus_shape(time_steps=time_steps, num_batches=batch_size)

# Create random input (in a real scenario, this would be your visual stimulus)
random_stimulus = torch.rand(input_shape)

# Run inference
responses = model.forward(random_stimulus)
print(f"Model predicted responses shape: {responses.shape}")
```

## Visualizing Model Components

Visualize the spatial and temporal components of the model's filters:

```python
import matplotlib.pyplot as plt
from openretina.utils.plotting import plot_stimulus_composition

# Extract the first convolutional layer weights
first_layer = model.core.features[0].conv.weight_spatial

# Plot the weights of the first few filters
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat[:5]):
    # Plot spatial filter (first channel, first time step)
    if i < first_layer.shape[0]:
        spatial_filter = first_layer[i, 0, 0].detach().cpu().numpy()
        im = ax.imshow(spatial_filter, cmap='RdBu_r')
        ax.set_title(f"Filter {i+1}")
        ax.axis('off')
plt.colorbar(im, ax=axes.ravel().tolist())
plt.tight_layout()
plt.show()
```

## Using In-silico Analysis Tools

Find the most exciting stimulus for a specific neuron:

```python
from openretina.insilico.stimulus_optimization import find_most_exciting_stimulus

# Select a neuron index to analyze
neuron_idx = 0

# Find the most exciting stimulus for this neuron
optimized_stimulus = find_most_exciting_stimulus(
    model, 
    neuron_idx=neuron_idx,
    num_steps=100,
    learning_rate=0.1
)

# Visualize the optimized stimulus
plt.figure(figsize=(10, 6))
plot_stimulus_composition(optimized_stimulus.squeeze())
plt.title(f"Most Exciting Stimulus for Neuron {neuron_idx}")
plt.show()
```

## Next Steps

Now that you're familiar with the basics, check out:

- [Loading different datasets](./data_io/index.md)
- [Training your own models](./tutorials/training.md)
- [Advanced in-silico experiments](./insilico/index.md) 
