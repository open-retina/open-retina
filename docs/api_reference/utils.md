---
title: Utilities API Reference
---

# Utilities API Reference

The utils package provides helper functions and utilities used throughout OpenRetina for file handling, visualization, model management, and data processing.

## Overview

The utils module contains:

- **File Utilities**: Functions for downloading, caching, and file management
- **Plotting**: Visualization tools for stimuli, responses, and model components
- **Model Utilities**: Helper functions for model training and evaluation
- **Data Handling**: Tools for working with HDF5 files and data formats
- **Miscellaneous**: General utility functions for reproducibility and debugging

## File Utilities

### File Management

::: openretina.utils.file_utils.get_local_file_path
    options:
        show_root_heading: false
        show_source: false

::: openretina.utils.file_utils.optionally_download_from_url
    options:
        show_root_heading: false
        show_source: false

::: openretina.utils.file_utils.unzip_and_cleanup
    options:
        show_root_heading: false
        show_source: false

### Cache Management

::: openretina.utils.file_utils.get_cache_directory
    options:
        show_root_heading: false
        show_source: false

## Plotting and Visualization

### Stimulus Visualization

::: openretina.utils.plotting.plot_stimulus_composition
    options:
        show_root_heading: false
        show_source: false

::: openretina.utils.plotting.save_stimulus_to_mp4_video
    options:
        show_root_heading: false
        show_source: false

### Video Processing

::: openretina.utils.plotting.undo_video_normalization
    options:
        show_root_heading: false
        show_source: false

## HDF5 Data Handling

### File Operations

::: openretina.utils.h5_handling.load_h5_into_dict
    options:
        show_root_heading: false
        show_source: false

::: openretina.utils.h5_handling.load_dataset_from_h5
    options:
        show_root_heading: false
        show_source: false

### Data Structure Exploration

::: openretina.utils.h5_handling.print_h5_structure
    options:
        show_root_heading: false
        show_source: false

::: openretina.utils.h5_handling.h5_to_folders
    options:
        show_root_heading: false
        show_source: false

## Model Utilities

### Training Helpers

::: openretina.utils.model_utils.eval_state
    options:
        show_root_heading: false
        show_source: false

::: openretina.utils.model_utils.OptimizerResetCallback
    options:
        show_root_heading: false
        show_source: false

## Video Analysis

### Frequency Analysis

::: openretina.utils.video_analysis.calculate_fft
    options:
        show_root_heading: false
        show_source: false

::: openretina.utils.video_analysis.weighted_main_frequency
    options:
        show_root_heading: false
        show_source: false

::: openretina.utils.video_analysis.decompose_kernel
    options:
        show_root_heading: false
        show_source: false

### Filtering

::: openretina.utils.video_analysis.butter_lowpass_filter
    options:
        show_root_heading: false
        show_source: false

## Miscellaneous Utilities

### Reproducibility

::: openretina.utils.misc.set_seed
    options:
        show_root_heading: false
        show_source: false

### Output Capture

::: openretina.utils.capture_output.CaptureOutputAndWarnings
    options:
        show_root_heading: false
        show_source: false

## Usage Examples

### File Management

```python
from openretina.utils.file_utils import get_local_file_path, get_cache_directory

# Download and cache a remote file
remote_url = "https://example.com/data/file.h5"
local_path = get_local_file_path(remote_url)
print(f"File cached at: {local_path}")

# Check cache directory
cache_dir = get_cache_directory()
print(f"Cache directory: {cache_dir}")

# Download with custom cache location
custom_cache = "/path/to/custom/cache"
local_path = get_local_file_path(remote_url, custom_cache)
```

### Stimulus Visualization

```python
import torch
import matplotlib.pyplot as plt
from openretina.utils.plotting import plot_stimulus_composition, save_stimulus_to_mp4_video

# Create sample stimulus (channels=2, time=50, height=16, width=18)
stimulus = torch.randn(2, 50, 16, 18)

# Plot stimulus composition
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_stimulus_composition(
    stimulus=stimulus.numpy(),
    temporal_trace_ax=axes[0, 0],
    freq_ax=axes[0, 1],
    spatial_ax=axes[1, 0],
    highlight_x_list=[(10, 20), (30, 40)]  # Highlight specific time ranges
)
plt.show()

# Save stimulus as video
save_stimulus_to_mp4_video(
    stimulus=stimulus,
    filepath="stimulus_video.mp4",
    fps=30,
    start_at_frame=0
)
```

### HDF5 Data Handling

```python
from openretina.utils.h5_handling import load_h5_into_dict, print_h5_structure

# Explore HDF5 file structure
print_h5_structure("data/responses.h5")

# Load specific datasets
data_dict = load_h5_into_dict("data/responses.h5")
print("Available datasets:")
for key, value in data_dict.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: {value.shape}")
    else:
        print(f"  {key}: {type(value)}")

# Load with specific group
responses_data = load_dataset_from_h5(
    "data/responses.h5",
    dataset_key="session1/responses",
    start_idx=0,
    end_idx=1000
)
```

### Model Training Utilities

```python
import torch
from openretina.utils.model_utils import eval_state
from openretina.utils.misc import set_seed

# Set reproducible seed
set_seed(42, seed_torch=True)

# Use eval_state context manager
model = torch.nn.Linear(10, 1)
model.train()  # Model in training mode

with eval_state(model):
    # Model temporarily in eval mode
    print(f"In context: training={model.training}")
    output = model(torch.randn(5, 10))

# Model back to original training mode
print(f"After context: training={model.training}")
```

### Video Analysis

```python
import numpy as np
from openretina.utils.video_analysis import calculate_fft, weighted_main_frequency

# Analyze temporal kernel frequency content
temporal_kernel = np.random.randn(50)  # 50 frame temporal kernel
sampling_freq = 30.0  # 30 Hz

# Calculate FFT
frequencies, fft_magnitude = calculate_fft(
    temporal_kernel=temporal_kernel,
    sampling_frequency=sampling_freq,
    lowpass_cutoff=10.0
)

# Find dominant frequency
main_freq = weighted_main_frequency(frequencies, fft_magnitude)
print(f"Dominant frequency: {main_freq:.2f} Hz")

# Plot frequency spectrum
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(temporal_kernel)
plt.title('Temporal Kernel')
plt.xlabel('Frame')

plt.subplot(1, 2, 2)
plt.plot(frequencies, fft_magnitude)
plt.axvline(main_freq, color='red', linestyle='--', label=f'Main freq: {main_freq:.2f} Hz')
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.legend()

plt.tight_layout()
plt.show()
```

### Advanced Stimulus Plotting

```python
from openretina.utils.plotting import plot_stimulus_composition
import numpy as np

# Create more complex stimulus with patterns
time_steps, height, width = 100, 32, 32
stimulus = np.zeros((2, time_steps, height, width))

# Create moving grating in UV channel
for t in range(time_steps):
    x = np.linspace(0, 4*np.pi, width)
    y = np.linspace(0, 4*np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    # Moving sine wave
    phase = t * 0.2
    stimulus[0, t] = np.sin(X + phase) * np.cos(Y)
    
    # Different pattern in Green channel
    stimulus[1, t] = np.cos(X - phase) * np.sin(Y + phase)

# Plot with custom highlighting
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

highlight_periods = [(20, 30), (50, 60), (80, 90)]
plot_stimulus_composition(
    stimulus=stimulus,
    temporal_trace_ax=axes[0, 0],
    freq_ax=axes[0, 1],
    spatial_ax=axes[1, 0],
    highlight_x_list=highlight_periods
)

# Add custom analysis in the fourth subplot
mean_intensity = np.mean(stimulus, axis=(2, 3))  # Average over spatial dimensions
axes[1, 1].plot(mean_intensity[0], label='UV Channel', alpha=0.7)
axes[1, 1].plot(mean_intensity[1], label='Green Channel', alpha=0.7)
axes[1, 1].set_xlabel('Time (frames)')
axes[1, 1].set_ylabel('Mean Intensity')
axes[1, 1].set_title('Temporal Evolution')
axes[1, 1].legend()

# Highlight the same periods
for start, end in highlight_periods:
    axes[1, 1].axvspan(start, end, alpha=0.2, color='red')

plt.suptitle('Comprehensive Stimulus Analysis', fontsize=16)
plt.tight_layout()
plt.show()
```

### Batch File Processing

```python
import os
from pathlib import Path
from openretina.utils.file_utils import get_local_file_path

def process_multiple_files(file_urls, output_dir):
    """Download and process multiple files."""
    
    os.makedirs(output_dir, exist_ok=True)
    processed_files = []
    
    for i, url in enumerate(file_urls):
        print(f"Processing file {i+1}/{len(file_urls)}: {url}")
        
        try:
            # Download file
            local_path = get_local_file_path(url)
            
            # Process based on file type
            if local_path.suffix == '.h5':
                # Process HDF5 file
                data = load_h5_into_dict(local_path)
                print(f"  Loaded {len(data)} datasets")
                
            elif local_path.suffix in ['.mp4', '.avi']:
                # Process video file
                print(f"  Video file: {local_path}")
                
            # Copy to output directory
            output_path = Path(output_dir) / f"processed_{i}_{local_path.name}"
            import shutil
            shutil.copy2(local_path, output_path)
            processed_files.append(output_path)
            
        except Exception as e:
            print(f"  Error processing {url}: {e}")
    
    return processed_files

# Example usage
urls = [
    "https://example.com/data1.h5",
    "https://example.com/data2.h5",
    "https://example.com/video1.mp4"
]

# processed_files = process_multiple_files(urls, "output_data/")
```

### Custom Output Capture

```python
from openretina.utils.capture_output import CaptureOutputAndWarnings
import warnings

# Capture both stdout and warnings
with CaptureOutputAndWarnings() as captured:
    print("This will be captured")
    warnings.warn("This warning will be captured")
    print("More output")

print("Captured stdout:")
print(captured.stdout)
print("\nCaptured warnings:")
print(captured.warnings)
```

### Configuration and Constants

```python
# Access package constants
from openretina.utils.constants import *

# File utilities use these constants
from openretina.utils.file_utils import HUGGINGFACE_REPO_ID, GIN_BASE_URL

print(f"Default HuggingFace repo: {HUGGINGFACE_REPO_ID}")
print(f"GIN repository base URL: {GIN_BASE_URL}")
```

## Performance Tips

1. **File Caching**: Downloaded files are automatically cached to avoid repeated downloads
2. **HDF5 Loading**: Use `start_idx` and `end_idx` parameters to load only needed data portions
3. **Video Rendering**: Adjust FPS and quality settings for video output based on needs
4. **Memory Management**: Large visualizations can consume significant memory; consider reducing resolution

## Troubleshooting

### Common Issues

**Download failures**:
```python
# Check internet connection and URL accessibility
import requests
response = requests.head(url)
print(f"Status: {response.status_code}")

# Use custom cache directory if permissions issues
custom_cache = "/tmp/openretina_cache"
local_path = get_local_file_path(url, custom_cache)
```

**Visualization memory issues**:
```python
# Reduce stimulus size for plotting
stimulus_subset = stimulus[:, ::2, ::2, ::2]  # Downsample
plot_stimulus_composition(stimulus_subset, ...)

# Or plot only specific time ranges
start_frame, end_frame = 20, 50
plot_stimulus_composition(stimulus[:, start_frame:end_frame], ...)
```

**HDF5 file access errors**:
```python
# Check file exists and is readable
import os
assert os.path.exists(file_path), f"File not found: {file_path}"
assert os.access(file_path, os.R_OK), f"Cannot read file: {file_path}"

# Check file integrity
import h5py
try:
    with h5py.File(file_path, 'r') as f:
        print("File opened successfully")
except Exception as e:
    print(f"File error: {e}")
```

## Configuration

Utilities can be configured through environment variables:

```bash
# Set custom cache directory
export OPENRETINA_CACHE_DIR="/path/to/cache"

# Set download timeout
export OPENRETINA_DOWNLOAD_TIMEOUT="60"

# Set matplotlib backend for headless environments
export MPLBACKEND="Agg"
```

## See Also

- [Installation Guide](../installation.md): Setting up the environment
- [Models API](./models.md): Using utilities with models
- [Data I/O API](./data_io.md): Data handling utilities
- [Plotting Examples](../tutorials/pretrained_models.md): Visualization examples
