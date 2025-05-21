---
title: FAQs
---
# Frequently Asked Questions

## General Questions

### What is OpenRetina?

OpenRetina is an open-source Python package for building, training, and analyzing neural network models of the retina. It provides tools to predict how retinal neurons respond to visual stimuli using deep learning approaches.

### Who should use OpenRetina?

OpenRetina is designed for:
- Computational neuroscientists studying retinal function
- Vision researchers interested in modeling neural responses
- Machine learning researchers working on biologically-inspired vision models
- Neuroscience students learning about neural modeling

### What license is OpenRetina released under?

OpenRetina is released under an open-source license that allows free use, modification, and distribution. See the [LICENSE](https://github.com/open-retina/open-retina/blob/main/LICENSE) file for details.

## Technical Questions

### What are the system requirements?

OpenRetina requires:
- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (recommended for training, but not required for inference)

### How do I install OpenRetina?

```bash
# Simple installation
pip install openretina

# For development
git clone git@github.com:open-retina/open-retina.git
cd open-retina
pip install -e .
```

See the [Installation Guide](package_docs/installation.md) for more details.

### How large are the pre-trained models?

Pre-trained models range in size from ~10MB to ~100MB, depending on the complexity and input/output dimensions.

### Can I train models without a GPU?

Yes, but training will be significantly slower. For small models or simple experiments, CPU training is possible. For larger models, a CUDA-compatible GPU is strongly recommended.

## Models and Data

### What types of retina models does OpenRetina support?

OpenRetina supports:
- Core-readout architectures (CNN-based)
- Linear-nonlinear cascade models
- Sparse autoencoder models

### What datasets are included?

OpenRetina provides loaders for several published datasets:
- Höfling et al., 2024 (mouse retina calcium imaging)
- Karamanlis et al., 2024
- Maheswaranathan et al., 2023

### Can I use my own data with OpenRetina?

Yes! OpenRetina provides base classes for creating custom data loaders. Your data should follow certain formatting conventions (stimuli and responses), but the package is designed to be flexible.

### How do I create a custom model?

You can create custom models by combining different core and readout modules, or by implementing your own PyTorch modules that follow the OpenRetina interfaces. See the [Training Tutorial](package_docs/tutorials/training.md) for examples.

## Performance and Troubleshooting

### Why is my model training slow?

Model training speed depends on several factors:
- GPU availability and speed
- Model complexity
- Dataset size
- Batch size

Try reducing model complexity, using a smaller batch size, or ensuring you're using GPU acceleration if available.

### How do I evaluate model performance?

OpenRetina provides tools for computing standard metrics like correlation coefficient between model predictions and actual neural responses. You can also use the in-silico experiments to analyze model behavior.

### I'm getting an "out of memory" error. What should I do?

Try:
1. Reducing batch size
2. Reducing model complexity
3. Downsampling your input data
4. Using gradient accumulation
5. Moving to a GPU with more memory

### Where can I get help if I'm having problems?

- Check the [documentation](https://open-retina.org/)
- Open an issue on [GitHub](https://github.com/open-retina/open-retina/issues)
- Contact the authors mentioned in the papers

## Contributing

### How can I contribute to OpenRetina?

We welcome contributions! You can:
- Report bugs
- Suggest features
- Add missing documentation
- Implement new models or datasets
- Improve performance

See the [Contributing Guide](package_docs/contributing.md) for more details.

### How do I cite OpenRetina?

If you use OpenRetina in your research, please cite both the OpenRetina paper and any specific model papers:
TODO: check and extend
```
@article{openretina2025,
  title={OpenRetina: An open-source toolkit for modeling retinal responses to visual stimuli},
  author={...},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.03.07.642012}
}
```

For models like the Höfling et al. model, also cite the original paper:

```
@article{hoefling2024chromatic,
  title={A chromatic feature detector in the retina signals visual context changes},
  author={Höfling, Larissa and others},
  journal={eLife},
  volume={13},
  pages={e86860},
  year={2024},
  doi={10.7554/eLife.86860}
}
```
