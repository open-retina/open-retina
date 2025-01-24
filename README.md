# OpenRetina

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type%20checked-mypy-039dfc)](https://github.com/python/mypy)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

Open-source repository containing neural network models of the retina.
The models in this repository are inspired by and partially contain adapted code of [sinzlab/neuralpredictors](https://github.com/sinzlab/neuralpredictors).

## Installation

For normal usage:

```
pip install openretina
```

Test openretina by downloading a model and running a forward pass:
```python
import torch
from openretina.models import *

model = load_core_readout_from_remote("hoefling_2024_base_low_res", "cpu")
responses = model.forward(torch.rand(model.stimulus_shape(time_steps=50)))
```

For development and to have access to Jupyter notebooks:
```
git clone git@github.com:open-retina/open-retina.git
cd open-retina
pip install -e .
```

Before raising a PR please run:
```
# Fix formatting of python files
make fix-formatting
# Run type checks and unit tests
make test-all
```

## Design decisions and structure
With this repository we provide already pre-trained retina models that can be used for inference and intepretability out of the box, and dataloaders together with model architectures to train new models.
For training new models, we rely on [pytorch lightning](https://lightning.ai/docs/pytorch/stable/) in combination with [hydra](https://hydra.cc/docs/intro/) to manage the configurations for training and dataloading.

The openretina package is structured as follows:
- modules: pytorch modules that define layers and losses
- models: pytorch lightning models that define models that can be trained and evaluated (i.e. models from specific papers)
- data_io: dataloaders to manage access of data to be used for training
- insilico: Methods perform _insilico_ experiments with above models
    - stimulus_optimization: optimize inputs for neurons of above models according to interpretable objectives (e.g. most exciting inputs)
    - future options: gradient analysis, data analysis
- utils: Utility functions that are used across above submodules


## Related papers

The model in `openretina/hoefling_2024` was developed in the paper [A chromatic feature detector in the retina signals visual context changes](https://elifesciences.org/articles/86860) and can be cited as:

```
@article {10.7554/eLife.86860,
article_type = {journal},
title = {A chromatic feature detector in the retina signals visual context changes},
author = {Höfling, Larissa and Szatko, Klaudia P and Behrens, Christian and Deng, Yuyao and Qiu, Yongrong and Klindt, David Alexander and Jessen, Zachary and Schwartz, Gregory W and Bethge, Matthias and Berens, Philipp and Franke, Katrin and Ecker, Alexander S and Euler, Thomas},
editor = {Rieke, Fred and Smith, Lois EH and Rieke, Fred and Baccus, Stephen A and Wei, Wei},
volume = 13,
year = 2024,
month = {oct},
pub_date = {2024-10-04},
pages = {e86860},
citation = {eLife 2024;13:e86860},
doi = {10.7554/eLife.86860},
url = {https://doi.org/10.7554/eLife.86860},
keywords = {retina, computational modelling, visual ecology, convolutional neural networks, 2P imaging, natural stimuli},
journal = {eLife},
issn = {2050-084X},
publisher = {eLife Sciences Publications, Ltd},
}
```

The paper [Most discriminative stimuli for functional cell type clustering](https://openreview.net/forum?id=9W6KaAcYlr) explains a method to automatically cluster and interpret the modeled neurons and was also used with above model (for code see [ecker-lab/most-discriminative-stimuli](https://github.com/ecker-lab/most-discriminative-stimuli)).
