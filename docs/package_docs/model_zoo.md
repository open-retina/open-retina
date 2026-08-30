---
title: Model Zoo
---

# Model Zoo

OpenRetina provides a collection of pre-trained retinal models from published research. These models can be easily loaded and used for inference, analysis, or as starting points for further training.

## Available Pre-trained Models

All models are automatically downloaded and cached when first used. They are hosted on [Hugging Face](https://huggingface.co/datasets/open-retina/open-retina).

The following identifiers are accepted by `load_core_readout_from_remote`:

| Model identifier | Dataset | Variant |
| --- | --- | --- |
| `hoefling_2024_low_res` | Höfling et al., 2024 | Low-resolution mouse model |
| `hoefling_2024_high_res` | Höfling et al., 2024 | High-resolution mouse model |
| `karamanlis_2024_mouse` | Karamanlis et al., 2024 | Mouse model |
| `karamanlis_2024_marmoset` | Karamanlis et al., 2024 | Marmoset model |
| `maheswaranathan_2023` | Maheswaranathan et al., 2023 | Multi-session tiger salamander model |
| `sridhar_2025` | Sridhar et al., 2025 | Marmoset model |
| `goldin_2022_mouse` | Goldin et al., 2022 | Mouse model |
| `goldin_2022_axolotl` | Goldin et al., 2022 | Axolotl model |

The legacy identifier `hoefling_2024_base_low_res` remains available for analyses that require the model used in the first version of the OpenRetina preprint.

See the dataset references for [Höfling et al., 2024](../api_reference/data_io/hoefling_2024.md), [Karamanlis et al., 2024](../api_reference/data_io/karamanlis_2024.md), [Maheswaranathan et al., 2023](../api_reference/data_io/maheswaranathan_2023.md), [Sridhar et al., 2025](../api_reference/data_io/sridhar_2025.md), and [Goldin et al., 2022](../api_reference/data_io/goldin_2022.md).

```python
import torch

from openretina.models import load_core_readout_from_remote

model = load_core_readout_from_remote("hoefling_2024_low_res", "cpu")
stimulus = torch.rand(model.stimulus_shape(time_steps=50))
responses = model(stimulus)
```

## Loading and Using Models

### Basic Loading

```python
import torch
from openretina.models import load_core_readout_from_remote

# Load any available model
model = load_core_readout_from_remote("hoefling_2024_low_res", "cpu")

# Check model properties
print(f"Readout sessions: {model.readout.readout_keys()}")
print(f"Input shape for 50 time steps: {model.stimulus_shape(time_steps=50)}")
```

### Device Handling

```python
# Load on GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_core_readout_from_remote("karamanlis_2024_mouse", device)

# Move existing model to different device
model = model.to("cuda")
```

## Model Storage and Caching

Models are automatically cached in your local filesystem:

- **Default cache location**: `~/openretina_cache/`
- **Custom cache location**: Set via `OPENRETINA_CACHE_DIRECTORY` environment variable or the cache_directory_path argument in function calls.
- **Manual cache management**: Use `openretina.utils.file_utils` functions

```python
from openretina.utils.file_utils import get_cache_directory

# Check cache location
cache_dir = get_cache_directory()
print(f"Models cached in: {cache_dir}")

# Load with custom cache location
model = load_core_readout_from_remote(
    "hoefling_2024_low_res",
    "cpu", 
    cache_directory_path="/custom/path"
)
```

## Troubleshooting

### Common Issues

**Model download fails**:

- Check internet connection
- Verify cache directory permissions
- Try different cache location

**Out of memory errors**:

- Use CPU instead of GPU for inference
- Reduce batch size or temporal length
- Use lower resolution models

**Input shape mismatches**:

- Use `model.stimulus_shape()` to get correct input dimensions
- Check channel ordering (some models expect specific color channels)
- Verify temporal length is appropriate

### Getting Help

For model-specific issues:

1. Check the [FAQ](../faq.md)
2. Review original paper documentation
3. Open an issue on [GitHub](https://github.com/open-retina/open-retina/issues)
4. Contact the model authors
