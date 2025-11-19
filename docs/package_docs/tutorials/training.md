---
title: Training Custom Models
---

# Training Custom Models

This tutorial will guide you through the process of training a custom retina model using OpenRetina.

## Overview

Training a model in OpenRetina involves:

1. Preparing your dataset
2. Defining model architecture
3. Configuring training parameters
4. Running the training loop
5. Evaluating the trained model

## Prerequisites

Before starting, ensure you have installed OpenRetina with development dependencies:

```bash
pip install -e ".[dev]"
```

## Setting Up Your Data

First, you need to prepare your data or use one of the built-in datasets:

## TODO wrong dataloading!! Page was AI generated.

```python
# Import data loading functions
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
train_dataloaders = dataloaders["train"]
validation_dataloaders = dataloaders["validation"]
test_dataloaders = dataloaders["test"]
```

## Defining Your Model

Next, define your model architecture:

```python
import torch
from pytorch_lightning import LightningModule
from openretina.modules.core.base_core import Core
from openretina.modules.readout.base import Readout
from openretina.models.core_readout import CoreReadout

class SimpleCore(Core):
    def __init__(self, input_channels=2, hidden_channels=32):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv3d(input_channels, hidden_channels, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            torch.nn.ReLU()
        )
        
    def forward(self, x):
        return self.features(x)
        
    def stimulus_shape(self, time_steps, num_batches=1):
        return (num_batches, 2, time_steps, 16, 18)

class SimpleReadout(Readout):
    def __init__(self, input_shape, num_neurons=150):
        super().__init__()
        self.num_neurons = num_neurons
        _, channels, _, height, width = input_shape
        self.spatial_dims = height * width
        self.linear = torch.nn.Linear(channels * height * width, num_neurons)
        
    def forward(self, x):
        # x has shape (batch, channels, time, height, width)
        batch, channels, time, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # (batch, time, channels, height, width)
        x = x.reshape(batch * time, channels, height * width)
        x = x.reshape(batch * time, channels * height * width)
        x = self.linear(x)
        x = x.reshape(batch, time, self.num_neurons)
        return x

class RetinaLightningModel(LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        
        # Define the core
        self.core = SimpleCore(input_channels=2, hidden_channels=32)
        
        # Define the readout
        input_shape = (1, 32, 1, 16, 18)  # (batch, channels, time, height, width)
        self.readout = SimpleReadout(input_shape, num_neurons=150)
        
        # Combine core and readout
        self.model = CoreReadout(core=self.core, readout=self.readout)
        
        # Save hyperparameters
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        session_key, (stimulus, response) = batch
        y_hat = self(stimulus)
        
        # Calculate loss (e.g., Poisson loss for neural data)
        loss = torch.nn.functional.poisson_nll_loss(y_hat, response)
        
        # Log the loss
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        session_key, (stimulus, response) = batch
        y_hat = self(stimulus)
        
        # Calculate validation metrics
        loss = torch.nn.functional.poisson_nll_loss(y_hat, response)
        self.log("val_loss", loss)
        
        # Calculate correlation coefficient
        with torch.no_grad():
            pred = y_hat.reshape(-1, y_hat.shape[-1])
            target = response.reshape(-1, response.shape[-1])
            corr = calculate_correlation(pred, target)
            self.log("val_correlation", corr.mean())
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def calculate_correlation(pred, target):
    """Calculate correlation coefficient between predictions and targets."""
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    
    pred_std = torch.sqrt(torch.sum(pred_centered**2, dim=0))
    target_std = torch.sqrt(torch.sum(target_centered**2, dim=0))
    
    correlation = torch.sum(pred_centered * target_centered, dim=0) / (pred_std * target_std + 1e-8)
    return correlation
```

## Configuring Training with Hydra

OpenRetina uses Hydra for configuration management. Create a configuration file `config.yaml`:

```yaml
# config.yaml
training:
  max_epochs: 100
  batch_size: 32
  learning_rate: 1e-3
  
model:
  core:
    type: "SimpleCore"
    input_channels: 2
    hidden_channels: 32
  
  readout:
    type: "SimpleReadout"
    num_neurons: 150

data:
  dataset: "hoefling_2024"
  validation_clip_indices: [0, 1]
  batch_size: 32
```

## Running the Training

To train your model:

```python
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from openretina.data_io.cyclers import LongCycler

@hydra.main(config_path="config", config_name="config")
def train(cfg):
    # Initialize model
    model = RetinaLightningModel(learning_rate=cfg.training.learning_rate)
    
    # Load data
    responses = get_all_responses()
    movies = get_all_movies()
    
    dataloaders = natmov_dataloaders_v2(
        neuron_data_dictionary=responses,
        movies_dictionary=movies,
        validation_clip_indices=cfg.data.validation_clip_indices,
        batch_size=cfg.data.batch_size
    )
    
    # Create cyclers for multi-session training
    train_cycler = LongCycler(dataloaders["train"], shuffle=True)
    val_cycler = LongCycler(dataloaders["validation"], shuffle=False)
    
    # Wrap cyclers in DataLoader
    train_loader = DataLoader(train_cycler, batch_size=None)
    val_loader = DataLoader(val_cycler, batch_size=None)
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_correlation",
        mode="max",
        save_top_k=3,
        filename="{epoch}-{val_correlation:.4f}"
    )
    
    early_stopping = EarlyStopping(
        monitor="val_correlation",
        patience=10,
        mode="max"
    )
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Save the final model
    trainer.save_checkpoint("final_model.ckpt")
    
    return model

if __name__ == "__main__":
    train()
```

## Evaluating the Trained Model

After training, evaluate your model on a test set:

```python
# Load the best model
best_model = RetinaLightningModel.load_from_checkpoint("best_model.ckpt")

# Create test cycler
test_cycler = LongCycler(dataloaders["test"], shuffle=False)
test_loader = DataLoader(test_cycler, batch_size=None)

# Initialize the trainer for testing
trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)

# Test the model
test_results = trainer.test(best_model, test_loader)
print(f"Test results: {test_results}")
```

## Visualizing Model Filters

Visualize what your model has learned:

```python
import matplotlib.pyplot as plt
import numpy as np
from openretina.utils.plotting import plot_stimulus_composition

# Extract filters from the first convolutional layer
filters = best_model.core.features[0].weight.detach().cpu()

# Plot the first few filters
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat[:5]):
    if i < filters.shape[0]:
        # Plot spatial filter (first channel, middle time step)
        time_idx = filters.shape[2] // 2
        spatial_filter = filters[i, 0, time_idx].numpy()
        im = ax.imshow(spatial_filter, cmap='RdBu_r')
        ax.set_title(f"Filter {i+1}")
        ax.axis('off')
plt.colorbar(im, ax=axes.ravel().tolist())
plt.tight_layout()
plt.savefig("model_filters.png")
plt.show()
```

## Tips for Successful Training

1. **Regularization**: Use appropriate regularization to prevent overfitting
2. **Learning Rate**: Start with a conservative learning rate (e.g., 1e-3) and adjust as needed
3. **Batch Size**: Use the largest batch size that fits in your GPU memory
4. **Data Augmentation**: Consider applying data augmentation for more robust models
5. **Model Complexity**: Start with a simple model and gradually increase complexity

## Next Steps

After training your model, you can:

- Analyze its behavior using [in-silico experiments](../insilico/index.md)
- Fine-tune it for specific applications
- Save and share it with the community 
