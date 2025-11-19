---
title: Development Setup
---

# Development Setup

This guide walks you through setting up a development environment for OpenRetina, including all necessary tools and workflows for contributing to the project.

## Prerequisites

Before setting up the development environment, ensure you have:

- **Python 3.10 or higher**
- **Git** installed and configured
- **CUDA-compatible GPU** (recommended for training models)
- **GitHub account** for contributing

## Environment Setup

### 1. Fork and Clone the Repository

First, fork the OpenRetina repository on GitHub, then clone your fork:

```bash
git clone git@github.com:yourusername/open-retina.git
cd open-retina
```

### 2. Create a Virtual Environment

We recommend using a virtual environment to isolate dependencies:

```bash
# Using venv
python -m venv openretina-dev
source openretina-dev/bin/activate  # On Windows: openretina-dev\Scripts\activate

# Or using conda
conda create -n openretina-dev python=3.10
conda activate openretina-dev
```

### 3. Install Development Dependencies

Install the package in editable mode with all development dependencies:

```bash
pip install -e ".[dev]"
```

This installs:
- `ruff` - For linting and formatting
- `mypy` - For type checking
- `pytest` - For running tests
- `jupyterlab` - For notebook development
- Various type stubs for better type checking

### 4. Install Additional Dependencies for Model Development

If you plan to develop new models or work extensively with existing ones:

```bash
pip install -e ".[devmodels]"
```

This adds:
- `neuralpredictors` - For compatibility with existing model architectures

### 5. Verify Installation

Test that everything is working:

```bash
# Test basic functionality
python -c "import openretina; print('OpenRetina imported successfully')"

# Test development tools
ruff --version
mypy --version
pytest --version

# Test GPU availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Development Tools and Workflow

### Code Quality Tools

OpenRetina uses several tools to maintain code quality:

#### Ruff (Linting and Formatting)

Ruff handles both linting and code formatting:

```bash
# Check code style
ruff check openretina/ tests/

# Fix code style issues automatically
ruff check openretina/ tests/ --fix

# Format code
ruff format openretina/ tests/

# Check formatting without modifying files
ruff format --check --diff openretina/ tests/
```

#### MyPy (Type Checking)

Type checking helps catch errors early:

```bash
# Run type checking
mypy openretina/ tests/
```

#### Pytest (Testing)

Run tests to ensure your changes don't break existing functionality:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage report
pytest tests/ --cov=openretina

# Run notebook tests
pytest --nbmake notebooks/
```

### Makefile Commands

The project includes a Makefile with convenient commands:

```bash
# Quality checks
make test-types         # Run type checking
make test-codestyle     # Check code style
make test-formatting    # Check code formatting
make test-unittests     # Run unit tests
make test-all          # Run all checks

# Auto-fixing
make fix-codestyle     # Fix code style issues
make fix-formatting    # Format code
make fix-all          # Fix both style and formatting

# Specialized tests
make test-notebooks    # Test Jupyter notebooks
make test-corereadout  # Test core-readout model training
```

### Git Workflow

#### 1. Create Feature Branches

Always work on feature branches:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-description
# or
git checkout -b docs/documentation-update
```

#### 2. Make Commits

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add support for new dataset format

- Implement DataLoader for XYZ dataset
- Add tests for data loading functionality
- Update documentation with usage examples"
```

#### 3. Keep Your Branch Updated

Regularly sync with the main repository:

```bash
git fetch upstream
git rebase upstream/main
```

### Pre-commit Workflow

Before committing changes, always run:

```bash
# Fix any formatting issues
make fix-all

# Run all tests
make test-all
```

Only commit if all tests pass.

## IDE Configuration

### VS Code

Recommended extensions:
- Python
- Pylance
- Ruff
- MyPy Type Checker
- Jupyter

Example `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./openretina-dev/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "none",
    "ruff.args": ["--config", "pyproject.toml"],
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

### PyCharm

1. Configure Python interpreter to use your virtual environment
2. Enable Ruff for linting and formatting
3. Configure MyPy for type checking
4. Set up pytest as the default test runner

## Working with Models

### Testing Model Changes

When developing or modifying models:

```bash
# Quick test with limited data
make test-corereadout

# Test with synthetic data
make test-h5train
```

### GPU Development

For GPU-intensive development:

1. **Use CUDA-compatible PyTorch**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Test GPU availability**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Monitor GPU usage** during development:
   ```bash
   nvidia-smi -l 1  # Update every second
   ```

## Working with Documentation

### Building Documentation Locally

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material mkdocstrings

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### Documentation Development

- Documentation is written in Markdown
- API documentation is auto-generated using mkdocstrings
- Follow the [contributing guide](./contributing.md) for documentation standards

## Debugging and Profiling

### Debugging

For debugging complex issues:

```bash
# Run with Python debugger
python -m pdb script.py

# Or use ipdb for enhanced debugging
pip install ipdb
```

### Profiling

For performance optimization:

```bash
# Profile Python code
python -m cProfile -o profile.stats script.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler script.py
```

## Troubleshooting

### Common Issues

#### CUDA/GPU Issues
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Import Errors
```bash
# Reinstall in editable mode
pip uninstall openretina
pip install -e ".[dev]"
```

#### Test Failures
```bash
# Run tests with verbose output
pytest tests/ -v

# Run specific failing test
pytest tests/test_specific.py::test_function -v -s
```

### Getting Help

If you encounter issues:

1. Check existing [GitHub issues](https://github.com/open-retina/open-retina/issues)
2. Review the [FAQ](../faq.md)
3. Ask questions in GitHub discussions
4. Contact the maintainers

## Next Steps

After setting up your development environment:

1. Read the [Contributing Guide](./contributing.md)
2. Review [Code Style Guidelines](./code_style.md)
3. Explore the codebase structure
4. Pick an issue to work on from GitHub issues
5. Start contributing!

Happy coding! 🧠🔬
