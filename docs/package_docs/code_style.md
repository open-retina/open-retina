---
title: Code Style Guidelines
---

# Code Style Guidelines

This document outlines the coding conventions and style guidelines for `openretina`. Following these guidelines ensures consistency, readability, and maintainability across the codebase.

## General Principles

1. **Consistency**: Follow established patterns in the codebase
2. **Readability**: Write code that is easy to read and understand
3. **Simplicity**: Prefer simple, clear solutions over complex ones
4. **Documentation**: Document your code thoroughly
5. **Testing**: Write tests for all new functionality, if possible

## Python Code Style

### Formatting and Linting

`openretina` uses **Ruff** for both linting and code formatting:

- **Line length**: 120 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **String quotes**: Prefer double quotes `"` over single quotes `'`

### Naming Conventions

#### Variables and Functions
- Use `snake_case` for variables and functions
- Use descriptive names that clearly indicate purpose
- Avoid abbreviations unless they are well-known

```python
# Good
def compute_correlation_coefficient(predictions, targets):
    ...

# Avoid
def comp_corr_coef(pred, tgt):
    ...
```

#### Classes
- Use `PascalCase` for class names
- Choose names that clearly describe the class purpose

```python
# Good
class CoreReadoutModel(LightningModule):
    ...

class StimulusOptimizer:
    ...

# Avoid
class CRModel:
    ...
```

#### Constants
- Use `UPPER_SNAKE_CASE` for constants

```python
DEFAULT_BATCH_SIZE = 32
MAX_EPOCHS = 100
DEVICE_AUTO = "auto"
```

#### Private Members
- Use single underscore prefix for internal use
- Use double underscore prefix for name mangling (rare)

```python
class Model:
    def __init__(self):
        self.public_attr = "visible"
        self._internal_attr = "internal use"
        self.__private_attr = "name mangled"
```

### Type Hints

Use type hints for all public functions and class methods:

```python
from typing import Optional, Union, Dict, List, Tuple

def load_model(
    model_name: str, 
    device: str = "cpu",
    strict_loading: bool = True
) -> torch.nn.Module:
    """Load a pre-trained model."""
    ...

class DataLoader:
    def __init__(
        self, 
        data_path: str, 
        batch_size: int = 32,
        transform: Optional[callable] = None
    ) -> None:
        ...
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
```

### Error Handling

Use specific exception types and provide informative error messages:

```python
# Good
def load_dataset(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    if not path.endswith('.h5'):
        raise ValueError(f"Expected .h5 file, got: {path}")
    
    try:
        return h5py.File(path, 'r')
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {path}: {e}")

# Avoid
def load_dataset(path):
    try:
        return h5py.File(path, 'r')
    except:
        raise Exception("Error loading dataset")
```

## Documentation Style

### Docstrings

Use **Google-style docstrings** for all public functions, classes, and methods:

```python
def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int,
    learning_rate: float = 1e-3
) -> Dict[str, float]:
    """Train a neural network model.
    
    This function trains the provided model using the given dataloader
    for the specified number of epochs.
    
    Args:
        model: The neural network model to train.
        dataloader: DataLoader providing training data.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer. Defaults to 1e-3.
        
    Returns:
        A dictionary containing training metrics:
        - 'loss': Final training loss
        - 'accuracy': Final training accuracy
        
    Raises:
        ValueError: If num_epochs is not positive.
        RuntimeError: If training fails due to GPU/memory issues.
        
    Example:
        >>> model = create_model()
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> metrics = train_model(model, loader, num_epochs=10)
        >>> print(f"Final loss: {metrics['loss']:.4f}")
        Final loss: 0.0234
    """
```

### Class Documentation

Document classes with their purpose, key attributes, and usage:

```python
class CoreReadoutModel(LightningModule):
    """A core-readout model for predicting retinal responses.
    
    This model consists of a convolutional core for feature extraction
    and multiple readout layers for predicting neural responses.
    
    Attributes:
        core: The convolutional feature extractor.
        readout: Dictionary of readout layers for different cell types.
        loss_fn: Loss function used for training.
        
    Example:
        >>> model = CoreReadoutModel(
        ...     core_config={'n_filters': 64},
        ...     readout_config={'n_neurons': 100}
        ... )
        >>> responses = model(stimuli)
    """
```

### Module Documentation

Include module-level docstrings explaining the module's purpose:

```python
"""Core neural network modules for retina modeling.

This module contains the building blocks for constructing retinal models,
including convolutional cores, readout layers, and loss functions.

Classes:
    ConvCore: Convolutional feature extractor
    GaussianReadout: Gaussian-weighted spatial readout
    PoissonLoss: Poisson loss for spike count data
"""
```

## PyTorch and Lightning Conventions

### Model Structure

Follow PyTorch Lightning conventions for model organization:

```python
class RetinalModel(LightningModule):
    """Standard structure for retinal models."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.core = self._build_core(config['core'])
        self.readout = self._build_readout(config['readout'])
        self.loss_fn = self._build_loss(config['loss'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        features = self.core(x)
        return self.readout(features)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        stimuli, responses = batch
        predictions = self(stimuli)
        loss = self.loss_fn(predictions, responses)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
```

### Tensor Operations

- Use descriptive variable names for tensors
- Include tensor shape information in comments
- Use type hints for tensor shapes when helpful, using the `jaxtyping` module.

```python
def process_stimuli(stimuli: torch.Tensor) -> torch.Tensor:
    """Process visual stimuli through preprocessing steps.
    
    Args:
        stimuli: Input stimuli tensor of shape (batch, time, height, width, channels)
        
    Returns:
        Processed stimuli of shape (batch, time, height, width, channels)
    """
    # stimuli: (batch, time, height, width, channels)
    batch_size, n_frames, height, width, n_channels = stimuli.shape
    
    # Normalize to [0, 1]
    stimuli_norm = stimuli / 255.0
    
    # Apply temporal filtering
    stimuli_filtered = temporal_filter(stimuli_norm)  # (batch, time, height, width, channels)
    
    return stimuli_filtered
```

## Testing Style

### Unit Tests

Write clear, focused unit tests:

```python
import pytest
import torch
from openretina.models import CoreReadoutModel


class TestCoreReadoutModel:
    """Test suite for CoreReadoutModel."""
    
    def test_model_creation(self):
        """Test that model can be created with valid config."""
        config = {
            'core': {'n_filters': 64},
            'readout': {'n_neurons': 100}
        }
        model = CoreReadoutModel(config)
        assert isinstance(model, CoreReadoutModel)
    
    def test_forward_pass(self):
        """Test forward pass with dummy data."""
        model = CoreReadoutModel({'core': {'n_filters': 8}, 'readout': {'n_neurons': 10}})
        stimuli = torch.randn(2, 50, 32, 32, 3)  # (batch, time, height, width, channels)
        
        responses = model(stimuli)
        
        assert responses.shape == (2, 10)  # (batch, n_neurons)
        assert not torch.isnan(responses).any()
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_different_batch_sizes(self, batch_size):
        """Test model works with different batch sizes."""
        model = CoreReadoutModel({'core': {'n_filters': 8}, 'readout': {'n_neurons': 10}})
        stimuli = torch.randn(batch_size, 10, 16, 16, 3)
        
        responses = model(stimuli)
        
        assert responses.shape[0] == batch_size
```

## File Organization

### Project Structure

Maintain consistent project structure:

```
openretina/
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── base.py              # Base classes
│   ├── core_readout.py      # Specific model implementations
│   └── linear_nonlinear.py
├── modules/
│   ├── __init__.py
│   ├── core/                # Core modules
│   ├── readout/             # Readout modules
│   └── losses/              # Loss functions
├── data_io/
│   ├── __init__.py
│   ├── base_dataloader.py   # Base dataloader
│   └── hoefling_2024/       # Dataset-specific loaders
└── utils/
    ├── __init__.py
    └── visualization.py     # Utility functions
```

### File Naming

- Use `snake_case` for file names
- Be descriptive but concise
- Group related functionality in modules

## Configuration Style

### Hydra Configs

Use clear, hierarchical configuration structures:

```yaml
# config.yaml
model:
  _target_: openretina.models.CoreReadoutModel
  core:
    _target_: openretina.modules.ConvCore
    n_layers: 4
    filters: [16, 32, 64, 128]
    kernel_sizes: [7, 5, 5, 5]
    activation: "relu"
  readout:
    _target_: openretina.modules.GaussianReadout
    bias: true
    init_mu_range: 0.1

trainer:
  max_epochs: 100
  accelerator: "gpu"
  devices: 1

dataloader:
  batch_size: 32
  num_workers: 4
  shuffle: true
```

## Performance Considerations

### Memory Efficiency

- Use generators for large datasets
- Implement lazy loading where possible
- Clean up GPU memory explicitly

```python
def efficient_data_loading(data_path: str):
    """Generator for memory-efficient data loading."""
    with h5py.File(data_path, 'r') as f:
        for i in range(len(f['stimuli'])):
            stimulus = f['stimuli'][i]  # Load one item at a time
            response = f['responses'][i]
            yield torch.from_numpy(stimulus), torch.from_numpy(response)

# Clean up GPU memory
def cleanup_gpu():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Code Efficiency

- Vectorize operations when possible
- Use appropriate PyTorch operations
- Profile code for bottlenecks

```python
# Good: Vectorized operation
correlations = torch.corrcoef(torch.stack([predictions.flatten(), targets.flatten()]))[0, 1]

# Avoid: Loop-based calculation
correlations = []
for i in range(predictions.shape[0]):
    corr = torch.corrcoef(torch.stack([predictions[i].flatten(), targets[i].flatten()]))[0, 1]
    correlations.append(corr)
```

## Version Control

### Branch Naming

Use descriptive branch names:
- `feature/dataset-karamanlis-2024`
- `bugfix/gpu-memory-leak`
- `docs/api-reference-update`

Following these guidelines will help maintain a clean, consistent, and maintainable codebase that is easy for new contributors to understand and work with.
