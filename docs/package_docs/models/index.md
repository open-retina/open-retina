---
title: Models
---

# Models

OpenRetina provides various neural network architectures for modeling retinal responses. These range from classical linear-nonlinear cascades to deep convolutional neural networks with spatial readouts.

## Available Model Types

### Core-Readout Models

The primary model architecture in OpenRetina is the core-readout model structure, which consists of:

1. A **core** module that processes visual input through convolutional layers to extract relevant features
2. A **readout** module that maps these features to neural responses

This architecture is inspired by state-of-the-art deep learning approaches to neural system identification.

```python
from openretina.models import load_core_readout_from_remote

# Load a pre-trained core-readout model
model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")
```

Learn more about [Core-Readout Models](./core_readout.md).

### Single-cell models

TODO write this section


## Pre-trained Models

OpenRetina includes several pre-trained models from published studies:

TODO extend table...

| Model Name                    | Paper                | Description                                  | Input Dimensions |
| ----------------------------- | -------------------- | -------------------------------------------- | ---------------- |
| `hoefling_2024_base_low_res`  | Höfling et al., 2024 | Base model trained on mouse retina responses | (2, T, 16, 18)   |
| `hoefling_2024_base_high_res` | Höfling et al., 2024 | Higher resolution version                    | (2, T, 32, 36)   |

Where `T` is the number of time steps (typically 30-100).

## Creating Custom Models

To create a custom model, you can compose core and readout modules:

```python
import torch
from openretina.modules.core.base_core import Core
from openretina.modules.readout.base import Readout
from openretina.models.core_readout import CoreReadout

# Define a custom core module
class MyCustomCore(Core):
    def __init__(self, input_channels=2, hidden_channels=32):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(input_channels, hidden_channels, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.conv2 = torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
        
    def stimulus_shape(self, time_steps, num_batches=1):
        return (num_batches, 2, time_steps, 16, 18)

# Define a custom readout module
class MyCustomReadout(Readout):
    def __init__(self, input_shape, num_neurons=150):
        super().__init__()
        self.num_neurons = num_neurons
        _, channels, _, height, width = input_shape
        self.flatten = torch.nn.Flatten(start_dim=2)
        flattened_size = channels * height * width
        self.linear = torch.nn.Linear(flattened_size, num_neurons)
        
    def forward(self, x):
        # x has shape (batch, channels, time, height, width)
        batch, _, time = x.shape[:3]
        x = x.permute(0, 2, 1, 3, 4)  # (batch, time, channels, height, width)
        x = x.reshape(batch * time, x.shape[2], x.shape[3], x.shape[4])
        x = self.flatten(x)
        x = self.linear(x)
        x = x.reshape(batch, time, self.num_neurons)
        return x

# Create the core and readout modules
core = MyCustomCore(input_channels=2, hidden_channels=32)
readout = MyCustomReadout(input_shape=(1, 32, 1, 16, 18), num_neurons=150)

# Combine them into a CoreReadout model
custom_model = CoreReadout(core=core, readout=readout)
```

## Performance Considerations

Models vary in terms of:

- **Computational requirements**: Larger models require more memory and computation
- **Generalization**: Some models may generalize better to new stimuli
- **Biological realism**: Models differ in how well they capture biological constraints

Choose a model based on your specific research needs and computational resources. 
