# OpenRetina <img src="https://raw.githubusercontent.com/open-retina/open-retina/7aacfa64267930f787b16f24e4bc17047f285c25/assets/openretina_logo.png" align="right" width="120" />

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type%20checked-mypy-039dfc)](https://github.com/python/mypy)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch\&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning\&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![DOI](https://zenodo.org/badge/722208169.svg)](https://doi.org/10.5281/zenodo.14988814)

[![huggingface](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/open-retina/open-retina)

Open-source repository containing neural network models of the retina.
The models in this repository are inspired by and partially contain adapted code of [sinzlab/neuralpredictors](https://github.com/sinzlab/neuralpredictors). Accompanying preprint: [openretina: Collaborative Retina Modelling Across Datasets and Species](https://www.biorxiv.org/content/10.1101/2025.03.07.642012v1).

## Installation

`openretina` supports simply installing via pip.

```bash
# (Recommended) using a package manager like uv
uv pip install openretina

# Or directly via pip if you prefer
pip install openretina
```

If you want to train your own models, run Jupyter notebooks, contribute to the project, or modify the source code of `openretina`, we recommend to install from source.
Consider using `uv`, a fast and flexible project and package manager. If you are not familiar with uv, check out their [simple quickstart guide](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:open-retina/open-retina.git
cd open-retina

# Sync with uv
uv sync --extra dev

# Alternatively, install in editable mode via pip. 
pip install -e .[dev]
```

Test openretina by downloading a model and running a forward pass:

```python
import torch
from openretina.models import *

model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")
responses = model.forward(torch.rand(model.stimulus_shape(time_steps=50)))
```

## Contributing

Before raising a PR please run:

```bash
# Fix formatting of python files
make fix-formatting

# Run type checks and unit tests
make test-all
```

## Design decisions and structure

With this repository we provide already pre-trained retina models that can be used for inference and intepretability out of the box, and dataloaders together with model architectures to train new models.
For training new models, we rely on [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) in combination with [hydra](https://hydra.cc/docs/intro/) to manage the configurations for training and dataloading.

The openretina package is structured as follows:

* modules: pytorch modules that define layers and losses
* models: pytorch lightning models that define models that can be trained and evaluated (i.e. models from specific papers)
* data\_io: dataloaders to manage access of data to be used for training
* insilico: Methods perform *insilico* experiments with above models
  * stimulus\_optimization: optimize inputs for neurons of above models according to interpretable objectives (e.g. most exciting inputs)
  * future options: gradient analysis, data analysis
* utils: Utility functions that are used across above submodules

## Related papers and data sources

The Core + Readout model was developed in the paper [A chromatic feature detector in the retina signals visual context changes](https://elifesciences.org/articles/86860). All datasets used in openretina are shared under a CC-BY Share-Alike license, and we acknowledge and credit the original sources below:

* hoefling\_2024: Originally published by Höfling et al. (2024), eLife
  * Paper: [A chromatic feature detector in the retina signals visual context changes](https://doi.org/10.7554/eLife.86860).
  * Dataset originally deposited at: https://gin.g-node.org/eulerlab/rgc-natstim
* karamanlis\_2024: Originally published by Karamanlis et al. (2024), Nature.
  * Paper: [Nonlinear receptive fields evoke redundant retinal coding of natural scenes](https://doi.org/10.1038/s41586-024-08212-3)
  * Dataset: Karamanlis D, Gollisch T (2023) Dataset - Marmoset and mouse retinal ganglion cell responses to natural stimuli and supporting data. G-Node. https://doi.org/10.12751/g-node.ejk8kx
* maheswaranathan\_2023: Originally published by Maheswaranathan et al. (2023), Neuron
  * Paper: [Interpreting the retinal neural code for natural scenes: From computations to neurons](https://doi.org/10.1016/j.neuron.2023.06.007)
  * Dataset: Maheswaranathan, N., McIntosh, L., Tanaka, H., Grant, S., Kastner, D., Melander, J., Nayebi, A., Brezovec, L., Wang, J. Ganguli, S. Baccus, S. (2023). Interpreting the retinal neural code for natural scenes: from computations to neurons. Stanford Digital Repository. Available at https://purl.stanford.edu/rk663dm5577

The paper [Most discriminative stimuli for functional cell type clustering](https://openreview.net/forum?id=9W6KaAcYlr) explains the discriminatory stimulus objective we showcase in [notebooks/most\_discriminative\_stimulus](https://github.com/open-retina/open-retina/blob/main/notebooks/most_discriminative_stimulus.ipynb).
