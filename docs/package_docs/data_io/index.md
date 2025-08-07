---
title: Data Input/Output
---

## TODO check:

This page was partly written by AI. needs proofreading.

`openretina` provides tools for loading, processing, and generating visual stimuli and neural responses for retina modeling.

## Supported Datasets

`openretina` includes loaders for several published datasets:

### Höfling et al., 2024

This dataset contains two-photon calcium imaging responses from mouse retinal ganglion cells to visual stimuli.

```python
# First, import the necessary modules
from openretina.data_io.hoefling_2024.dataloaders import natmov_dataloaders_v2
from openretina.data_io.hoefling_2024.responses import get_all_responses
from openretina.data_io.hoefling_2024.stimuli import get_all_movies

# Load responses and movies
responses = get_all_responses()
movies = get_all_movies()

# Create dataloaders with validation clips 0 and 1
dataloaders = natmov_dataloaders_v2(
    neuron_data_dictionary=responses,
    movies_dictionary=movies,
    validation_clip_indices=[0, 1],
    batch_size=32
)

# Access specific splits
train_loaders = dataloaders["train"]  # Dictionary mapping session IDs to training dataloaders
validation_loaders = dataloaders["validation"]
test_loaders = dataloaders["test"]
```

Learn more about the [Höfling et al. dataset](./hoefling_2024.md).

### Karamanlis et al., 2024

This dataset contains calcium imaging responses from mouse retinal ganglion cells.

```python
from openretina.data_io.karamanlis_2024 import get_karamanlis_dataloaders

# Load the dataset
dataloaders = get_karamanlis_dataloaders(
    batch_size=32,
    download=True
)
```

### Maheswaranathan et al., 2023

This dataset contains electrophysiology recordings from primate retinal ganglion cells.

```python
from openretina.data_io.maheswaranathan_2023 import get_maheswaranathan_dataloaders

# Load the dataset
dataloaders = get_maheswaranathan_dataloaders(
    batch_size=32,
    download=True
)
```

## Base Classes

`openretina` provides abstract base classes for creating custom data loaders:

```python
from openretina.data_io.base_dataloader import BaseDataLoader
from pytorch_lightning import LightningDataModule

class MyCustomDataset(BaseDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Your implementation here
        
    def get_item(self, idx):
        # Return stimulus and response for a given index
        pass
```

## Stimulus Generation

You can generate artificial stimuli for model training and testing:

```python
from openretina.data_io.artificial_stimuli import (
    generate_gratings,
    generate_noise_stimulus,
    generate_moving_bar
)

# Generate a drifting grating stimulus
grating = generate_gratings(
    shape=(2, 30, 16, 18),  # (channels, time, height, width)
    temporal_frequency=2.0,  # Hz
    spatial_frequency=0.1,   # cycles per pixel
    orientation=45           # degrees
)

# Generate white noise stimulus
noise = generate_noise_stimulus(
    shape=(2, 30, 16, 18),
    distribution="gaussian"
)

# Generate a moving bar stimulus
bar = generate_moving_bar(
    shape=(2, 30, 16, 18),
    width=2,                # pixels
    speed=1,                # pixels per frame
    direction=0             # degrees
)
```

## Data Cyclers

For training models, `openretina` uses data cyclers that efficiently batch and preprocess data:

```python
from torch.utils.data import DataLoader
from openretina.data_io.cyclers import LongCycler, ShortCycler

# Example dataloaders
session_dataloaders = {
    "session1": DataLoader(...),
    "session2": DataLoader(...),
    "session3": DataLoader(...)
}

# Create a LongCycler that cycles through sessions until all data is exhausted
long_cycler = LongCycler(
    loaders=session_dataloaders,
    shuffle=True
)

# Create a ShortCycler that cycles through each session exactly once
short_cycler = ShortCycler(
    loaders=session_dataloaders
)

# Get the next batch from the cycler
for session_key, batch in long_cycler:
    # Process the batch
    inputs, targets = batch
    print(f"Processing batch from {session_key}")
```

## Custom Data Formats

To use your own data with `openretina`, you need to:

1. Prepare your stimuli as tensors of shape `(channels, time, height, width)`
2. Prepare your responses as tensors of shape `(neurons, time)`
3. Create a custom DataLoader that inherits from `BaseDataLoader`

For example:

```python
import torch
import numpy as np
from openretina.data_io.base_dataloader import BaseDataLoader

class MyExperimentDataLoader(BaseDataLoader):
    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        # Load your data
        self.stimuli = torch.from_numpy(np.load(f"{data_path}/stimuli.npy"))
        self.responses = torch.from_numpy(np.load(f"{data_path}/responses.npy"))
        
    def __len__(self):
        return len(self.stimuli)
        
    def get_item(self, idx):
        return {
            "stimulus": self.stimuli[idx],
            "response": self.responses[idx]
        }
```

## Performance Tips

When working with large datasets:

- Use memory-mapped files for large arrays
- Apply appropriate preprocessing (normalization, cropping, etc.)
- Consider using PyTorch's DataLoader with multiple workers
- Cache processed data when possible

