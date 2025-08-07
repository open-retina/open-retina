---
template: home.html
title: OpenRetina
homepage: true
hide:
  - navigation
  - toc
---
## Welcome!

`openretina` is an open-source neural network toolkit for modeling retinal responses to visual stimuli. It provides pre-trained models for inference and interpretability, as well as components for training new retina models with your own data.

---

<div class="grid cards" markdown>

-  [:simple-github:{ .lg .middle } __Package repository__](https://github.com/open-retina/open-retina/){:target="_blank"}

    ---

    Check out the main package repository on GitHub


-   [:fontawesome-regular-newspaper:{ .lg .middle } __Read the paper__](https://www.biorxiv.org/content/10.1101/2025.03.07.642012v1){:target="_blank"}

    ---

    `openretina` has a biorxiv pre-print!


-   [:material-clock-fast:{ .lg .middle } __Install__](package_docs/installation.md)

    ---
    === "git clone"

        Default, most flexible option. Comes with config files and notebooks.
        ```bash
        git clone git@github.com:open-retina/open-retina.git
        cd open-retina
        pip install -e .
        ```

    === "pip"
        Simplest installation for using pre-trained models
        ```bash
        pip install openretina
        ```


-   [:material-book-open:{ .lg .middle } __Read the documentation__](package_docs/index.md)

    ---

    Browse detailed documentation on models, data loading, and in-silico experiments

</div>

---
## Quick Start

```python
import torch
from openretina.models import load_core_readout_from_remote

# Load a pre-trained model
model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")

# Run a forward pass with random input
responses = model.forward(torch.rand(model.stimulus_shape(time_steps=50)))
print(f"Model predicted responses shape: {responses.shape}")
```
