---
title: In-silico Experiments
---

# In-silico Experiments

OpenRetina provides tools for performing in-silico (computational) experiments with trained retina models. These experiments help in understanding the function and behavior of modeled retinal neurons.

## What are In-silico Experiments?

In-silico experiments are computational studies that simulate experiments you might perform on real retinal tissue. They allow researchers to:

- Probe neural response properties systematically
- Generate optimal stimuli for neurons
- Test hypotheses about neural coding
- Analyze population-level properties

## Available Experimental Techniques

### Stimulus Optimization
# TODO this page was written by AI, double check
The `insilico.stimulus_optimization` module provides tools to find stimuli that optimally drive individual neurons or neuron groups:

```python
import torch
from openretina.models import load_core_readout_from_remote
from openretina.insilico.stimulus_optimization.objective import IncreaseObjective, MeanReducer
from openretina.insilico.stimulus_optimization.optimization_stopper import OptimizationStopper
from openretina.insilico.stimulus_optimization.optimizer import optimize_stimulus
from openretina.insilico.stimulus_optimization.regularizer import TotalVariationRegularizer, ClipStimulus

# Load a pre-trained model
model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")

# Initialize a random stimulus (starting point for optimization)
stimulus_shape = model.stimulus_shape(time_steps=30, num_batches=1)
stimulus = torch.randn(stimulus_shape, requires_grad=True)

# Select a neuron index to optimize for
neuron_idx = 5

# Create an objective to maximize the response of the selected neuron
objective = IncreaseObjective(
    model=model,
    reducer=MeanReducer(neuron_dim=1, indices=neuron_idx)
)

# Set regularization to promote smooth, natural-looking stimuli
tv_reg = TotalVariationRegularizer(weight=0.01)

# Set up the optimizer and stopping criteria
optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.05)
stopper = OptimizationStopper(max_iterations=200, patience=20)

# Run the optimization
optimize_stimulus(
    stimulus=stimulus,
    optimizer_init_fn=optimizer_fn,
    objective_object=objective,
    optimization_stopper=stopper,
    stimulus_regularization_loss=tv_reg,
    stimulus_postprocessor=ClipStimulus(0.0, 1.0)
)

# The stimulus has been optimized in-place
print(f"Optimized stimulus shape: {stimulus.shape}")
```

The stimulus optimization module is documented in the [API reference](../api_reference/insilico/stimulus_optimization.md).

### Tuning Analysis

The `insilico.tuning_analyses` module contains tools for characterizing neural tuning properties (coming soon).

## Feature Visualization

Techniques for visualizing what features neurons are sensitive to:

```python
from openretina.insilico import visualize_receptive_fields

# Visualize receptive fields for all neurons
receptive_fields = visualize_receptive_fields(
    model,
    resolution=(64, 64)  # Higher resolution for visualization
)

# Plot the receptive fields
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
for i, ax in enumerate(axes.flat):
    if i < len(receptive_fields):
        ax.imshow(receptive_fields[i], cmap='RdBu_r')
        ax.set_title(f"Neuron {i}")
        ax.axis('off')
plt.tight_layout()
plt.show()
```

## Implementing Custom Experiments

You can implement your own in-silico experiments using OpenRetina's models:

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from openretina.data_io.artificial_stimuli import generate_gratings

def custom_experiment(model, parameter_range):
    """
    Custom experiment testing model responses across a parameter range.
    
    Args:
        model: A trained OpenRetina model
        parameter_range: Range of parameter values to test
        
    Returns:
        results: Results of the experiment
    """
    results = []
    
    for parameter in parameter_range:
        # Generate stimulus based on parameter
        stimulus = generate_gratings(
            shape=model.stimulus_shape(time_steps=30),
            temporal_frequency=parameter
        )
        
        # Get model response
        with torch.no_grad():
            response = model(stimulus).mean(dim=0)  # Average over time
        
        # Store results
        results.append(response.cpu().numpy())
    
    return np.array(results)  # Shape: [parameter_values, neurons]

# Example usage
model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")
frequencies = np.linspace(0.5, 10, 20)  # Test 20 frequencies from 0.5 to 10 Hz
results = custom_experiment(model, frequencies)

# Plot results for a specific neuron
neuron_idx = 0
plt.figure()
plt.plot(frequencies, results[:, neuron_idx])
plt.xlabel("Temporal Frequency (Hz)")
plt.ylabel(f"Response of Neuron {neuron_idx}")
plt.title("Temporal Frequency Tuning")
plt.show()
```

## Analyzing Model Behavior

Understanding how your model behaves in response to different stimuli can provide insights into both the model and the biological system it represents:

1. **Compare to biological data**: Test if model responses match known retinal properties
2. **Identify specialized neurons**: Find neurons sensitive to specific stimulus features
3. **Analyze population coding**: Study how information is represented across neurons
4. **Test hypotheses**: Use the model to test theoretical predictions

## Best Practices

When performing in-silico experiments:

1. **Control comparisons**: Use consistent settings when comparing across conditions
2. **Statistical validation**: Repeat experiments with different random seeds
3. **Parameter exploration**: Systematically explore parameter spaces
4. **Biological plausibility**: Consider the biological relevance of your experiments
5. **Computational efficiency**: Optimize code for large-scale experiments 
