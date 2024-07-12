# OpenRetina

Open source repository containing neural network models of the retina.
The models in this repository are inspired by and partially contain adapted code of [sinzlab/neuralpredictors](https://github.com/sinzlab/neuralpredictors).

## Installation
For normal usage:
```
pip install openretina
```

For development:
```
git clone git@github.com:open-retina/open-retina.git
cd open-retina
pip install -e .
```

## Related papers
The model in `openretina/hoefling_2024` was used in the paper [A chromatic feature detector in the retina signals visual context changes](https://www.biorxiv.org/content/10.1101/2022.11.30.518492.abstract) and can be cited as:
```
@article{hofling2022chromatic,
  title={A chromatic feature detector in the retina signals visual context changes},
  author={H{\"o}fling, Larissa and Szatko, Klaudia P and Behrens, Christian and Deng, Yuyao and Qiu, Yongrong and Klindt, David A and Jessen, Zachary and Schwartz, Gregory W and Bethge, Matthias and Berens, Philipp and others},
  journal={bioRxiv},
  pages={2022--11},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

The paper [Most discriminative stimuli for functional cell type clustering](https://openreview.net/forum?id=9W6KaAcYlr) explains a method to automatically cluster and interpret the modeled neurons and was also used with above model (for code see [ecker-lab/most-discriminative-stimuli](https://github.com/ecker-lab/most-discriminative-stimuli)).

