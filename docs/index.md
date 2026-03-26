---
template: home.html
title: OpenRetina
homepage: true
hide:
  - navigation
  - toc
  - footer
---
## Welcome!

`openretina` is an open-source neural network toolkit for modeling retinal responses to visual stimuli. It provides pre-trained models for inference and interpretability, as well as components for training new retina models with your own data.

<div style="text-align: center; margin: 0.8rem 0 0.3rem 0;" markdown>

[:material-book-open: Read the documentation](package_docs/index.md){ .md-button .landing-docs-btn }

</div>

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


-   [:simple-huggingface:{ .lg .middle } __Datasets on HuggingFace__](https://huggingface.co/datasets/open-retina/open-retina){:target="_blank"}

    ---

    Browse and download retinal datasets from our HuggingFace repository.

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

---
## Participating Labs

<div class="grid cards" markdown>

-   :material-flask:{ .lg .middle } __Euler Lab__

    ---

    University of Tübingen, Center for Integrative Neuroscience

    [:octicons-link-external-16: Lab website](https://eulerlab.de/){ target="_blank" }

-   :material-flask:{ .lg .middle } __Bethge Lab__

    ---

    University of Tübingen, AI Center

    [:octicons-link-external-16: Lab website](https://bethgelab.org/){ target="_blank" }

-   :material-flask:{ .lg .middle } __Marre Lab__

    ---

    Institut de la Vision, Paris

    [:octicons-link-external-16: Lab website](https://www.institut-vision.org/en/researchers/olivier-marre){ target="_blank" }

-   :material-flask:{ .lg .middle } __Ecker Lab__

    ---

    University of Göttingen

    [:octicons-link-external-16: Lab website](https://eckerlab.org/){ target="_blank" }

-   :material-flask:{ .lg .middle } __Gollisch Lab__

    ---

    University Medical Center Göttingen, Department of Ophthalmology

    [:octicons-link-external-16: Lab website](https://www.retina.uni-goettingen.de/){ target="_blank" }


</div>

---

__Lead maintainers__
:   Federico D'Agostino — University of Tübingen, AI Center  
    Thomas Zenkel — University of Tübingen, Center for Integrative Neuroscience  
    Larissa Höfling — University of Tübingen, AI Center

__Contributors__
:   Baptiste Lorenzi — Institut de la Vision, Paris  
    Michaela Vystrčilová — University of Göttingen  
    Dominic Gonschorek — University of Tübingen, Center for Integrative Neuroscience  
    Samuel Suhai — University of Tübingen, Center for Integrative Neuroscience  
    Shashwat Sridhar — University Medical Center Göttingen  
    Samuele Virgili — Institut de la Vision, Paris
