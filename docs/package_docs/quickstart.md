---
title: Quick Start
---
After installing `openretina` (see [installation](./installation.md)), we will now showcase a few features of the library by loading a pre-trained model, using it to predict neural activity, and visualizing the model's internal weights.

## Loading a Pre-trained Model

`openretina` provides several pre-trained models. Here's how to load one:

```python
import torch
from openretina.models import load_core_readout_from_remote

# Load a pre-trained model (will download if not already cached)
model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")
```

## Running Inference

We can now use this model by first creating a random stimulus and predicting the activity of each modeled neuron in response.

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

The models in openretina use a Core+Readout architecture.
The Core, which is shared across all sessions and cells, processes the input stimulus and extracts a rich feature representation of the visual input.
The Readout, which is specific to each experimental session, then maps the Core's output feature space to individual neurons' firing-rate predictions for a given stimulus.

We can visualize these components individually, for example by looking at a particular weight in the first layer of the Core.
```python
# Visualise a specific channel weight for the first convolutional layer
conv_layer_figure = model.core.plot_weight_visualization(layer=0, in_channel=1, out_channel=0)
conv_layer_figure.show()
```

And by visualizing the weights of the readout layer that predict the activity of an individual neuron based on the features of the Core:

```python
# Visualise the readout weights of a neuron for a particular readout session
session_key = model.readout.readout_keys()[0]
readout_figure = model.readout[session_key].plot_weight_for_neuron(5)
readout_figure.show()
```

# Next Steps

Now that you're familiar with the basics, you can continue to learn about:

- [Command Line Interface](./command_line.md)
- [Architecture](./architecture.md)
- [Loading different datasets](./data_io/index.md)
- [Training your own models](./tutorials/training.md)
- [Advanced in-silico experiments](./insilico/index.md) 
