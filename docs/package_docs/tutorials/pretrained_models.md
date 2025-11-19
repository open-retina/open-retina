---
title: Loading Pre-trained Models
---

## TODO check page: AI generated

This tutorial will guide you through loading and using pre-trained retinal models in `openretina`. You'll learn how to download models, run inference, and analyze the results.

## Overview

`openretina` provides several pre-trained models from published research that you can use immediately for:
- Neural response prediction
- Model analysis and interpretation  
- Transfer learning for new datasets
- Benchmarking custom models

## Getting Started

### Installation

First, ensure you have `openretina` installed:

```bash
pip install openretina
```

### Basic Model Loading

The simplest way to get started is with the `load_core_readout_from_remote` function:

```python
import torch
from openretina.models import load_core_readout_from_remote

# Load a pre-trained model (automatically downloads if needed)
model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")

print(f"Model loaded successfully!")
print(f"Model type: {type(model)}")
```

## Available Models

### Model Overview

`openretina` includes models from three major publications:

| Model Name                    | Source                      | Species    | Architecture     | Input Shape    |
| ----------------------------- | --------------------------- | ---------- | ---------------- | -------------- |
| `hoefling_2024_base_low_res`  | Höfling et al. 2024         | Mouse      | Core-Readout     | (2, T, 16, 18) |
| `hoefling_2024_base_high_res` | Höfling et al. 2024         | Mouse      | Core-Readout     | (2, T, 32, 36) |
| `karamanlis_2024_base`        | Karamanlis et al. 2024      | Macaque    | Core-Readout     | (1, T, H, W)   |
| `karamanlis_2024_gru`         | Karamanlis et al. 2024      | Macaque    | GRU Core-Readout | (1, T, H, W)   |
| `maheswaranathan_2023_base`   | Maheswaranathan et al. 2023 | Salamander | Core-Readout     | (1, T, H, W)   |
| `maheswaranathan_2023_gru`    | Maheswaranathan et al. 2023 | Salamander | GRU Core-Readout | (1, T, H, W)   |

### Choosing the Right Model

**For mouse retina analysis**: Use Höfling et al. models
- Low-res for faster inference: `hoefling_2024_base_low_res`
- High-res for detailed analysis: `hoefling_2024_base_high_res`

**For primate retina**: Use Karamanlis et al. models
- Standard model: `karamanlis_2024_base`
- Temporal dynamics: `karamanlis_2024_gru`

**For salamander retina**: Use Maheswaranathan et al. models
- Standard model: `maheswaranathan_2023_base`
- Recurrent processing: `maheswaranathan_2023_gru`

## Step-by-Step Tutorial

### Step 1: Load a Model

```python
import torch
from openretina.models import load_core_readout_from_remote

# Choose device (GPU recommended for larger models)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Höfling et al. low-resolution model
model = load_core_readout_from_remote("hoefling_2024_base_low_res", device)

# Model is now ready for use
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Step 2: Understand Model Input Requirements

```python
# Get the correct input shape for this model
batch_size = 4
time_steps = 50

input_shape = model.stimulus_shape(time_steps=time_steps, num_batches=batch_size)
print(f"Required input shape: {input_shape}")
# Output: torch.Size([4, 50, 16, 18, 2])
# Format: (batch, time, height, width, channels)

# For the Höfling model:
# - 2 channels represent UV and Green wavelengths
# - Spatial resolution is 16×18 pixels
# - Time dimension is flexible (30+ frames recommended)
```

### Step 3: Create Sample Input

```python
# Create random stimulus (in practice, this would be your actual visual stimulus)
random_stimulus = torch.rand(input_shape, device=device)

# For real data, you might load from a file:
# stimulus = torch.load("my_stimulus.pt")
# stimulus = stimulus.to(device)

print(f"Stimulus shape: {random_stimulus.shape}")
print(f"Stimulus value range: [{random_stimulus.min():.3f}, {random_stimulus.max():.3f}]")
```

### Step 4: Run Model Inference

```python
# Set model to evaluation mode
model.eval()

# Run inference (no gradients needed for prediction)
with torch.no_grad():
    predicted_responses = model(random_stimulus)

print(f"Predicted responses shape: {predicted_responses.shape}")
print(f"Number of neurons: {predicted_responses.shape[-1]}")

# Response format: (batch, neurons)
# Each value represents predicted neural activity
```

### Step 5: Analyze Model Outputs

```python
import matplotlib.pyplot as plt
import numpy as np

# Extract responses for the first batch item
responses_batch_0 = predicted_responses[0].cpu().numpy()

# Plot response distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(responses_batch_0, bins=30, alpha=0.7)
plt.xlabel('Predicted Response')
plt.ylabel('Count')
plt.title('Response Distribution')

plt.subplot(1, 3, 2)
plt.plot(responses_batch_0[:50])  # First 50 neurons
plt.xlabel('Neuron Index')
plt.ylabel('Response')
plt.title('Neural Responses')

plt.subplot(1, 3, 3)
# Show correlation between neurons (first 20 for visibility)
if len(responses_batch_0) >= 20:
    corr_matrix = np.corrcoef(predicted_responses[:, :20].cpu().numpy().T)
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Neuron Correlations')

plt.tight_layout()
plt.show()
```

## Working with Real Data

### Loading Your Own Stimuli

```python
import numpy as np

# Example: Load stimulus from file
def load_custom_stimulus(file_path, target_shape):
    """Load and preprocess custom visual stimulus."""
    
    # Load your data (adjust based on your format)
    # stimulus = np.load(file_path)  # For .npy files
    # stimulus = h5py.File(file_path)['stimulus'][:]  # For HDF5
    
    # For demonstration, create a moving grating
    batch, time, height, width, channels = target_shape
    stimulus = np.zeros((batch, time, height, width, channels))
    
    for t in range(time):
        # Create moving sinusoidal grating
        x = np.linspace(0, 4*np.pi, width)
        y = np.linspace(0, 4*np.pi, height)
        X, Y = np.meshgrid(x, y)
        
        # Moving grating pattern
        phase = t * 0.2  # Speed of movement
        pattern = np.sin(X + phase) * np.cos(Y)
        
        # Set both channels (for 2-channel models like Höfling)
        if channels == 2:
            stimulus[0, t, :, :, 0] = pattern  # UV channel
            stimulus[0, t, :, :, 1] = pattern * 0.8  # Green channel
        else:
            stimulus[0, t, :, :, 0] = pattern
    
    # Normalize to [0, 1] range
    stimulus = (stimulus - stimulus.min()) / (stimulus.max() - stimulus.min())
    
    return torch.tensor(stimulus, dtype=torch.float32)

# Use with model
input_shape = model.stimulus_shape(time_steps=100, num_batches=1)
custom_stimulus = load_custom_stimulus("my_data.npy", input_shape)
custom_stimulus = custom_stimulus.to(device)

# Get predictions
with torch.no_grad():
    responses = model(custom_stimulus)
    
print(f"Responses to custom stimulus: {responses.shape}")
```

### Batch Processing

```python
def process_multiple_stimuli(model, stimuli_list, batch_size=8):
    """Process multiple stimuli efficiently in batches."""
    
    all_responses = []
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(stimuli_list), batch_size):
            # Get batch
            batch_stimuli = stimuli_list[i:i+batch_size]
            
            # Stack into single tensor
            batch_tensor = torch.stack(batch_stimuli).to(device)
            
            # Process batch
            batch_responses = model(batch_tensor)
            all_responses.append(batch_responses.cpu())
            
            print(f"Processed {i+len(batch_stimuli)}/{len(stimuli_list)} stimuli")
    
    # Concatenate all responses
    return torch.cat(all_responses, dim=0)

# Example usage
# stimuli_list = [load_stimulus(f"stimulus_{i}.npy") for i in range(20)]
# all_responses = process_multiple_stimuli(model, stimuli_list)
```

## Model Introspection

### Accessing Model Components

```python
# Examine model architecture
print("Model structure:")
print(model)

print("\nModel hyperparameters:")
for key, value in model.hparams.items():
    print(f"  {key}: {value}")

# Access specific components
core = model.core
readout = model.readout

print(f"\nCore type: {type(core)}")
print(f"Readout type: {type(readout)}")
print(f"Number of output neurons: {readout.n_neurons}")
```

### Analyzing Receptive Fields

```python
def visualize_filters(model, layer_idx=0, num_filters=8):
    """Visualize convolutional filters from the core."""
    
    # Get first convolutional layer
    conv_layer = model.core.features[layer_idx].conv
    weights = conv_layer.weight_spatial.detach().cpu()
    
    # weights shape: (out_channels, in_channels, height, width)
    fig, axes = plt.subplots(2, num_filters//2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(min(num_filters, weights.shape[0])):
        filter_weights = weights[i, 0]  # First input channel
        
        axes[i].imshow(filter_weights, cmap='coolwarm', 
                      vmin=-weights.abs().max(), vmax=weights.abs().max())
        axes[i].set_title(f'Filter {i}')
        axes[i].axis('off')
    
    plt.suptitle(f'Convolutional Filters - Layer {layer_idx}')
    plt.tight_layout()
    plt.show()

# Visualize filters
visualize_filters(model, layer_idx=0, num_filters=8)
```

### Examining Readout Locations

```python
def plot_readout_locations(model):
    """Plot the spatial locations of readout units."""
    
    if hasattr(model.readout, 'mu'):
        # Gaussian readout locations
        mu = model.readout.mu.detach().cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(mu[:, 0], mu[:, 1], alpha=0.6, s=20)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Readout Unit Locations')
        plt.grid(True, alpha=0.3)
        
        # Add some statistics
        plt.text(0.02, 0.98, f'Number of units: {len(mu)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.show()
        
        return mu
    else:
        print("Model does not have spatial readout locations (mu parameter)")
        return None

# Plot readout locations
readout_locations = plot_readout_locations(model)
```

## Performance Analysis

### Response Quality Metrics

```python
def analyze_response_statistics(responses):
    """Analyze basic statistics of model responses."""
    
    responses_np = responses.detach().cpu().numpy()
    
    stats = {
        'mean': np.mean(responses_np),
        'std': np.std(responses_np),
        'min': np.min(responses_np),
        'max': np.max(responses_np),
        'sparsity': np.mean(responses_np == 0),
        'dynamic_range': np.max(responses_np) - np.min(responses_np)
    }
    
    print("Response Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    return stats

# Analyze your predictions
stats = analyze_response_statistics(predicted_responses)
```

### Model Timing

```python
import time

def benchmark_model(model, input_shape, num_runs=100):
    """Benchmark model inference speed."""
    
    # Warm up
    dummy_input = torch.rand(input_shape, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"Inference FPS: {fps:.1f}")
    
    return avg_time

# Benchmark the model
input_shape = model.stimulus_shape(time_steps=50, num_batches=1)
avg_time = benchmark_model(model, input_shape)
```

## Saving and Loading Results

### Saving Predictions

```python
import os
from datetime import datetime

def save_predictions(responses, metadata, save_dir="results"):
    """Save model predictions with metadata."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{timestamp}.pt"
    filepath = os.path.join(save_dir, filename)
    
    # Save data
    torch.save({
        'responses': responses.cpu(),
        'metadata': metadata,
        'model_name': model.__class__.__name__,
        'timestamp': timestamp
    }, filepath)
    
    print(f"Predictions saved to: {filepath}")
    return filepath

# Save your results
metadata = {
    'model_name': 'hoefling_2024_base_low_res',
    'stimulus_type': 'random',
    'time_steps': 50,
    'batch_size': 4
}

save_path = save_predictions(predicted_responses, metadata)
```

### Loading Saved Results

```python
def load_predictions(filepath):
    """Load previously saved predictions."""
    
    data = torch.load(filepath)
    
    print(f"Loaded predictions from: {filepath}")
    print(f"Model: {data['metadata']['model_name']}")
    print(f"Timestamp: {data['timestamp']}")
    print(f"Response shape: {data['responses'].shape}")
    
    return data['responses'], data['metadata']

# Load results
# responses, metadata = load_predictions(save_path)
```

## Advanced Usage

### Model Ensemble

```python
def create_model_ensemble(model_names, device):
    """Create an ensemble of multiple models."""
    
    models = {}
    for name in model_names:
        print(f"Loading {name}...")
        models[name] = load_core_readout_from_remote(name, device)
        models[name].eval()
    
    return models

def ensemble_prediction(models, stimulus):
    """Get ensemble prediction by averaging multiple models."""
    
    predictions = []
    
    with torch.no_grad():
        for name, model in models.items():
            # Ensure stimulus has correct shape for this model
            target_shape = model.stimulus_shape(
                time_steps=stimulus.shape[1], 
                num_batches=stimulus.shape[0]
            )
            
            if stimulus.shape != target_shape:
                print(f"Warning: stimulus shape {stimulus.shape} doesn't match "
                      f"model {name} requirements {target_shape}")
                continue
                
            pred = model(stimulus)
            predictions.append(pred)
    
    if predictions:
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred
    else:
        raise ValueError("No compatible models found for given stimulus")

# Example: ensemble of GRU models
# gru_models = create_model_ensemble([
#     "karamanlis_2024_gru", 
#     "maheswaranathan_2023_gru"
# ], device)
# ensemble_result = ensemble_prediction(gru_models, stimulus)
```

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```python
# Solution: Reduce batch size or use CPU
model = load_core_readout_from_remote("model_name", "cpu")
# Or process smaller batches
```

**2. Input shape mismatch**
```python
# Solution: Use model.stimulus_shape() to get correct dimensions
correct_shape = model.stimulus_shape(time_steps=50, num_batches=1)
stimulus = torch.rand(correct_shape)
```

**3. Model download fails**
```python
# Solution: Check internet connection and cache permissions
from openretina.utils.file_utils import get_cache_directory
cache_dir = get_cache_directory()
print(f"Cache directory: {cache_dir}")
# Ensure directory exists and is writable
```

### Performance Tips

1. **Use GPU when available**: Models run much faster on GPU
2. **Batch processing**: Process multiple stimuli together for efficiency
3. **Appropriate time lengths**: Use 30+ frames for temporal models
4. **Memory management**: Clear GPU cache between large operations

```python
# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## Next Steps

After mastering pre-trained model usage, you might want to:

1. **Analyze model behavior**: Use the in-silico experiments module
2. **Train custom models**: See the [Training Tutorial](./training.md)
3. **Optimize stimuli**: Try [Stimulus Optimization](./stimulus_optimization.md)
4. **Integrate your data**: Create custom dataloaders for your datasets

## Additional Resources

- [Model Zoo](../model_zoo.md): Complete list of available models
- [API Reference](../../api_reference/models/core_readout.md): Detailed model documentation
- [Datasets](../datasets.md): Information about training datasets
- [GitHub Examples](https://github.com/open-retina/open-retina/tree/main/notebooks): Jupyter notebook examples
