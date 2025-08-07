---
title: Stimulus Optimization Tutorial
---

# Stimulus Optimization Tutorial

This tutorial will guide you through using OpenRetina's stimulus optimization tools to find visual stimuli that optimally drive retinal neurons. You'll learn how to generate most exciting inputs (MEIs) and most discriminative stimuli (MDS) for understanding neural function.

## Overview

Stimulus optimization in OpenRetina allows you to:

- **Find optimal stimuli**: Generate visual patterns that maximally activate specific neurons
- **Create discriminative stimuli**: Find patterns that differentiate between cell types
- **Understand receptive fields**: Reveal the visual features that neurons prefer
- **Analyze population responses**: Study how groups of neurons respond to optimized inputs

## Key Concepts

### Most Exciting Input (MEI)
A visual stimulus that maximally activates a target neuron or group of neurons.

### Most Discriminative Stimulus (MDS)  
A stimulus that maximally activates one group of neurons while minimizing responses in other groups, useful for understanding functional differences between cell types.

### Gradient-based Optimization
Uses backpropagation through the neural network to iteratively improve stimuli based on objective functions.

## Getting Started

### Installation and Imports

```python
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

# OpenRetina imports
from openretina.models import load_core_readout_from_remote
from openretina.insilico import (
    optimize_stimulus,
    IncreaseObjective,
    ContrastiveNeuronObjective,
    OptimizationStopper,
    MeanReducer,
    SliceMeanReducer
)
from openretina.insilico.stimulus_optimization.regularizer import (
    ChangeNormJointlyClipRangeSeparately,
    TemporalGaussianLowPassFilterProcessor
)
from openretina.utils.plotting import plot_stimulus_composition
```

### Load a Pre-trained Model

```python
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model
model = load_core_readout_from_remote("hoefling_2024_base_low_res", device)
model.eval()  # Set to evaluation mode

print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Number of output neurons: {model.readout.n_neurons}")
```

## Basic Stimulus Optimization

### Step 1: Create Initial Stimulus

```python
# Define stimulus shape
time_steps = 40
stimulus_shape = model.stimulus_shape(time_steps=time_steps, num_batches=1)
print(f"Stimulus shape: {stimulus_shape}")

# Initialize random stimulus (starting point for optimization)
torch.manual_seed(42)  # For reproducibility
stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)

print(f"Initial stimulus range: [{stimulus.min():.3f}, {stimulus.max():.3f}]")
```

### Step 2: Define Objective Function

```python
# Choose a neuron to optimize for (e.g., neuron index 50)
target_neuron = 50

# Create response reducer to focus on specific time frames
# This averages responses from frames 10-20 (out of 40 total)
reducer = SliceMeanReducer(axis=0, start=10, length=10)

# Create objective to maximize target neuron's response
objective = IncreaseObjective(
    model=model,
    neuron_indices=target_neuron,
    data_key=None,  # For single session models
    response_reducer=reducer
)

print(f"Objective: Maximize response of neuron {target_neuron}")
print(f"Using time frames {reducer.start} to {reducer.start + reducer.length - 1}")
```

### Step 3: Set Up Regularization and Post-processing

```python
# Stimulus post-processors to maintain realistic stimulus properties

# 1. Clip stimulus values to expected physiological range
stimulus_clipper = ChangeNormJointlyClipRangeSeparately(
    min_max_values=[(-0.6, 6.2), (-0.9, 6.2)],  # Range for [UV, Green] channels
    norm=30.0  # Joint normalization factor
)

# 2. Apply temporal smoothing to avoid unrealistic high-frequency components
temporal_filter = TemporalGaussianLowPassFilterProcessor(
    sigma=0.5,      # Gaussian kernel standard deviation
    kernel_size=5,  # Temporal kernel size
    device=device
)

# Combine post-processors
stimulus_postprocessors = [stimulus_clipper, temporal_filter]

# Initialize stimulus with reasonable values
stimulus.data = stimulus_clipper.process(stimulus.data * 0.1)
print(f"Processed initial stimulus range: [{stimulus.min():.3f}, {stimulus.max():.3f}]")
```

### Step 4: Run Optimization

```python
# Define optimizer initialization function
optimizer_init_fn = partial(torch.optim.SGD, lr=100.0)

# Set optimization stopping criteria
stopper = OptimizationStopper(max_iterations=50)

# Run optimization
print("Starting optimization...")
initial_response = objective.forward(stimulus).item()
print(f"Initial objective value: {initial_response:.4f}")

optimize_stimulus(
    stimulus=stimulus,
    optimizer_init_fn=optimizer_init_fn,
    objective_object=objective,
    optimization_stopper=stopper,
    stimulus_regularization_loss=None,  # No additional regularization
    stimulus_postprocessor=stimulus_postprocessors
)

# Check final objective value
final_response = objective.forward(stimulus).item()
print(f"Final objective value: {final_response:.4f}")
print(f"Improvement: {final_response - initial_response:.4f}")
```

### Step 5: Visualize Results

```python
def plot_optimized_stimulus(stimulus, title="Optimized Stimulus"):
    """Plot the optimized stimulus with temporal, spatial, and frequency components."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convert to numpy for plotting
    stim_np = stimulus[0].detach().cpu().numpy()  # Remove batch dimension
    
    # Plot stimulus composition
    plot_stimulus_composition(
        stimulus=stim_np,
        temporal_trace_ax=axes[0, 0],
        freq_ax=axes[0, 1], 
        spatial_ax=axes[1, 0],
        highlight_x_list=[(reducer.start, reducer.start + reducer.length - 1)]
    )
    
    # Add temporal evolution plot
    axes[1, 1].imshow(stim_np[0, :, 8, :].T, aspect='auto', cmap='coolwarm')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('X Position') 
    axes[1, 1].set_title('Temporal Evolution (UV Channel)')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Plot the optimized stimulus
plot_optimized_stimulus(stimulus, f"MEI for Neuron {target_neuron}")
```

## Advanced Optimization: Most Discriminative Stimuli

### Understanding Cell Type Classification

```python
# For the Höfling dataset, we can work with functional cell types
# Let's create groups based on ON/OFF responses

# Example cell type groupings (you would get these from your model's metadata)
def get_cell_type_groups(model):
    """Get example cell type groupings for demonstration."""
    n_neurons = model.readout.n_neurons
    
    # Create example groupings (in practice, use actual cell type labels)
    off_cells = list(range(0, n_neurons//3))
    on_off_cells = list(range(n_neurons//3, 2*n_neurons//3))  
    on_cells = list(range(2*n_neurons//3, n_neurons))
    
    return {
        'OFF': off_cells,
        'ON-OFF': on_off_cells,
        'ON': on_cells
    }

cell_groups = get_cell_type_groups(model)
print("Cell type groups:")
for group_name, indices in cell_groups.items():
    print(f"  {group_name}: {len(indices)} cells")
```

### Create Contrastive Objective

```python
# Select target group and contrast groups
target_group = cell_groups['ON'][:10]  # Target: first 10 ON cells
contrast_groups = [
    cell_groups['OFF'][:20],    # Contrast against OFF cells
    cell_groups['ON-OFF'][:15]  # Contrast against ON-OFF cells
]

print(f"Target group: {len(target_group)} ON cells")
print(f"Contrast groups: {[len(g) for g in contrast_groups]} cells")

# Create contrastive objective
contrastive_objective = ContrastiveNeuronObjective(
    model=model,
    on_cluster_idc=target_group,
    off_cluster_idc_list=contrast_groups,
    data_key=None,
    response_reducer=reducer,
    temperature=1.6  # Controls sharpness of contrast
)

print("Created contrastive objective for most discriminative stimulus")
```

### Optimize Discriminative Stimulus

```python
# Create new stimulus for MDS optimization
torch.manual_seed(123)
mds_stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)
mds_stimulus.data = stimulus_clipper.process(mds_stimulus.data * 0.1)

print("Starting MDS optimization...")
initial_contrast = contrastive_objective.forward(mds_stimulus).item()
print(f"Initial contrastive objective: {initial_contrast:.4f}")

# Run optimization with same settings
optimize_stimulus(
    stimulus=mds_stimulus,
    optimizer_init_fn=optimizer_init_fn,
    objective_object=contrastive_objective,
    optimization_stopper=OptimizationStopper(max_iterations=50),
    stimulus_postprocessor=stimulus_postprocessors
)

final_contrast = contrastive_objective.forward(mds_stimulus).item()
print(f"Final contrastive objective: {final_contrast:.4f}")
print(f"Improvement: {final_contrast - initial_contrast:.4f}")

# Plot the discriminative stimulus
plot_optimized_stimulus(mds_stimulus, "Most Discriminative Stimulus (ON vs OFF/ON-OFF)")
```

## Comparing MEI vs MDS

### Side-by-Side Analysis

```python
def compare_stimuli_responses(mei_stim, mds_stim, model, cell_groups):
    """Compare responses of MEI and MDS across different cell types."""
    
    model.eval()
    with torch.no_grad():
        mei_responses = model(mei_stim)[0].cpu().numpy()  # Remove batch dim
        mds_responses = model(mds_stim)[0].cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot responses by cell type
    for i, (group_name, indices) in enumerate(cell_groups.items()):
        if i >= 3:  # Limit to 3 groups for visualization
            break
            
        # MEI responses
        axes[0, i].hist(mei_responses[indices], bins=20, alpha=0.7, label='MEI')
        axes[0, i].set_title(f'MEI: {group_name} Cells')
        axes[0, i].set_xlabel('Response')
        axes[0, i].set_ylabel('Count')
        
        # MDS responses  
        axes[1, i].hist(mds_responses[indices], bins=20, alpha=0.7, 
                       label='MDS', color='orange')
        axes[1, i].set_title(f'MDS: {group_name} Cells')
        axes[1, i].set_xlabel('Response')
        axes[1, i].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nResponse Summary:")
    print("=" * 50)
    for group_name, indices in cell_groups.items():
        mei_mean = np.mean(mei_responses[indices])
        mds_mean = np.mean(mds_responses[indices])
        print(f"{group_name:>8}: MEI={mei_mean:.3f}, MDS={mds_mean:.3f}")

# Compare the optimized stimuli
compare_stimuli_responses(stimulus, mds_stimulus, model, cell_groups)
```

## Advanced Features

### Custom Regularization

```python
from openretina.insilico.stimulus_optimization.regularizer import RangeRegularizationLoss

# Create custom regularization to encourage specific properties
range_regularizer = RangeRegularizationLoss(
    min_values=torch.tensor([-1.0, -1.0], device=device),
    max_values=torch.tensor([7.0, 7.0], device=device),
    penalty_weight=0.1
)

# Use in optimization
optimize_stimulus(
    stimulus=stimulus,
    optimizer_init_fn=optimizer_init_fn,
    objective_object=objective,
    optimization_stopper=OptimizationStopper(max_iterations=30),
    stimulus_regularization_loss=range_regularizer,  # Add regularization
    stimulus_postprocessor=stimulus_postprocessors
)
```

### Multi-neuron Optimization

```python
# Optimize for multiple neurons simultaneously
target_neurons = [10, 25, 40, 55]  # Multiple neuron indices

multi_objective = IncreaseObjective(
    model=model,
    neuron_indices=target_neurons,  # List of neurons
    data_key=None,
    response_reducer=reducer
)

print(f"Optimizing for {len(target_neurons)} neurons: {target_neurons}")

# Run optimization
multi_stimulus = torch.randn(stimulus_shape, requires_grad=True, device=device)
multi_stimulus.data = stimulus_clipper.process(multi_stimulus.data * 0.1)

optimize_stimulus(
    stimulus=multi_stimulus,
    optimizer_init_fn=optimizer_init_fn,
    objective_object=multi_objective,
    optimization_stopper=OptimizationStopper(max_iterations=50),
    stimulus_postprocessor=stimulus_postprocessors
)

plot_optimized_stimulus(multi_stimulus, f"MEI for Neurons {target_neurons}")
```

### Parameter Sensitivity Analysis

```python
def sensitivity_analysis(base_stimulus, objective, param_ranges):
    """Analyze sensitivity to optimization parameters."""
    
    results = {}
    
    for param_name, values in param_ranges.items():
        results[param_name] = []
        
        for value in values:
            # Create fresh stimulus
            test_stimulus = base_stimulus.clone().detach()
            test_stimulus.requires_grad = True
            
            if param_name == 'learning_rate':
                opt_fn = partial(torch.optim.SGD, lr=value)
            else:
                opt_fn = optimizer_init_fn
                
            # Run short optimization
            optimize_stimulus(
                stimulus=test_stimulus,
                optimizer_init_fn=opt_fn,
                objective_object=objective,
                optimization_stopper=OptimizationStopper(max_iterations=20),
                stimulus_postprocessor=stimulus_postprocessors
            )
            
            final_obj = objective.forward(test_stimulus).item()
            results[param_name].append(final_obj)
            
            print(f"{param_name}={value}: final_objective={final_obj:.4f}")
    
    return results

# Test different learning rates
param_ranges = {
    'learning_rate': [1.0, 10.0, 50.0, 100.0, 200.0]
}

sensitivity_results = sensitivity_analysis(stimulus, objective, param_ranges)

# Plot results
plt.figure(figsize=(10, 6))
for param_name, values in sensitivity_results.items():
    plt.plot(param_ranges[param_name], values, 'o-', label=param_name)
plt.xlabel('Parameter Value')
plt.ylabel('Final Objective Value')
plt.title('Parameter Sensitivity Analysis')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Best Practices

### Optimization Tips

1. **Start with small learning rates**: High learning rates can lead to unstable optimization
2. **Use appropriate regularization**: Prevent unrealistic stimuli with proper constraints
3. **Monitor convergence**: Check that objective values are improving
4. **Use temporal smoothing**: Apply low-pass filtering for realistic temporal dynamics
5. **Clip stimulus ranges**: Ensure values stay within physiological bounds

### Choosing Parameters

```python
# Recommended parameter ranges for different scenarios

# For detailed, high-resolution optimization
detailed_params = {
    'learning_rate': 10.0,
    'max_iterations': 100,
    'temporal_sigma': 0.3,  # Sharper temporal features
    'clip_range': [(-1.0, 8.0), (-1.5, 8.0)]
}

# For quick exploration
quick_params = {
    'learning_rate': 50.0,
    'max_iterations': 20,
    'temporal_sigma': 1.0,  # Smoother temporal features
    'clip_range': [(-0.5, 6.0), (-0.8, 6.0)]
}

# For biological realism
biological_params = {
    'learning_rate': 20.0,
    'max_iterations': 50,
    'temporal_sigma': 0.5,
    'clip_range': [(-0.6, 6.2), (-0.9, 6.2)]  # Based on Höfling et al. data
}

print("Use these parameter sets based on your optimization goals")
```

### Troubleshooting

```python
def diagnose_optimization(stimulus, objective, n_steps=10):
    """Diagnose optimization issues by tracking progress."""
    
    optimizer = torch.optim.SGD([stimulus], lr=50.0)
    objectives = []
    
    for i in range(n_steps):
        obj = objective.forward(stimulus)
        loss = -obj  # Negative because we minimize loss but maximize objective
        
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norm = stimulus.grad.norm().item()
        
        optimizer.step()
        
        objectives.append(obj.item())
        print(f"Step {i}: objective={obj.item():.4f}, grad_norm={grad_norm:.6f}")
        
        if grad_norm < 1e-6:
            print("Warning: Very small gradients detected!")
            break
    
    # Plot progress
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(objectives, 'o-')
    plt.xlabel('Optimization Step')
    plt.ylabel('Objective Value')
    plt.title('Optimization Progress')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(stimulus.detach().cpu().numpy().flatten(), bins=30)
    plt.xlabel('Stimulus Values')
    plt.ylabel('Count')
    plt.title('Stimulus Value Distribution')
    
    plt.tight_layout()
    plt.show()

# Diagnose optimization issues
# diagnose_optimization(stimulus, objective)
```

## Saving and Loading Results

### Save Optimized Stimuli

```python
import os
from datetime import datetime

def save_optimization_results(stimulus, objective, metadata, save_dir="optimization_results"):
    """Save optimized stimulus and metadata."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save stimulus
    stimulus_path = os.path.join(save_dir, f"stimulus_{timestamp}.pt")
    torch.save(stimulus.detach().cpu(), stimulus_path)
    
    # Save metadata
    final_objective = objective.forward(stimulus).item()
    metadata.update({
        'final_objective': final_objective,
        'timestamp': timestamp,
        'stimulus_shape': list(stimulus.shape)
    })
    
    metadata_path = os.path.join(save_dir, f"metadata_{timestamp}.pt")
    torch.save(metadata, metadata_path)
    
    print(f"Results saved:")
    print(f"  Stimulus: {stimulus_path}")
    print(f"  Metadata: {metadata_path}")
    
    return stimulus_path, metadata_path

# Save results
metadata = {
    'model_name': 'hoefling_2024_base_low_res',
    'target_neuron': target_neuron,
    'optimization_type': 'MEI',
    'learning_rate': 100.0,
    'max_iterations': 50
}

save_optimization_results(stimulus, objective, metadata)
```

## Applications and Use Cases

### 1. Receptive Field Mapping
Use MEI optimization to understand what visual features each neuron prefers.

### 2. Cell Type Characterization
Use MDS optimization to find stimuli that best differentiate between functional cell types.

### 3. Model Validation
Compare optimized stimuli with known biological receptive field properties.

### 4. Feature Discovery
Discover unexpected visual features that strongly drive neural responses.

### 5. Stimulus Design
Create optimized stimuli for experimental validation in real retinal recordings.

## Next Steps

After mastering stimulus optimization:

1. **Combine with real data**: Use optimized stimuli to validate model predictions
2. **Explore inner representations**: Optimize for intermediate model layers
3. **Population analysis**: Study how populations of neurons respond to optimized stimuli
4. **Cross-species comparison**: Compare optimal stimuli across different retinal models

## Additional Resources

- [In-silico Experiments Overview](../insilico/index.md)
- [API Reference: Stimulus Optimization](../api_reference/insilico/stimulus_optimization.md)
- [Example Notebook: Most Discriminative Stimuli](https://github.com/open-retina/open-retina/blob/main/notebooks/most_discriminative_stimulus.ipynb)
- [Model Zoo](../model_zoo.md): Available pre-trained models for optimization
