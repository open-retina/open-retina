---
title: Contributing Guide
---

# Contributing to `openretina`

We welcome contributions to `openretina`! This guide will help you get started with contributing to the project, whether you're fixing bugs, adding features, improving documentation, or adding new models and datasets.

## How to Contribute

### Types of Contributions

We appreciate the following types of contributions:

- **Bug reports**: Report issues you encounter
- **Feature requests**: Suggest new features or improvements
- **Documentation improvements**: Fix typos, add examples, or improve explanations
- **New models**: Implement new neural network architectures
- **New datasets**: Add support for additional retinal datasets
- **Code improvements**: Optimize performance, improve code quality
- **Tests**: Add or improve test coverage

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine:
   ```bash
   git clone git@github.com:yourusername/open-retina.git
   cd open-retina
   ```
3. **Create a new branch** for your contribution:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Environment Setup

1. **Install the package in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```
   This installs `openretina` with all development dependencies including:
   - `ruff` for linting and formatting
   - `mypy` for type checking
   - `pytest` for testing
   - `jupyterlab` for notebook development

2. **For model development**, also install:
   ```bash
   pip install -e ".[devmodels]"
   ```

### Development Workflow

Before submitting any changes, please follow this workflow:

1. **Make your changes** following our [code style guidelines](./code_style.md)

2. **Run the development checks**:
   ```bash
   # Fix code formatting
   make fix-formatting
   
   # Run all tests (type checks, code style, unit tests)
   make test-all
   ```

3. **Test your changes thoroughly**:
   - Ensure existing tests pass
   - Add new tests for new functionality, if possible
   - Test with different datasets if applicable

4. **Update documentation** if needed:
   - Update docstrings for new functions/classes
   - Update user-facing documentation
   - Add examples for new features

### Submitting Changes

1. **Commit your changes** with descriptive commit messages:
   ```bash
   git add .
   git commit -m "Add feature: brief description of what you added"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub with:
   - A clear title and description
   - Reference to any related issues
   - List of changes made
   - Any breaking changes highlighted

## Guidelines for Specific Contributions

### Adding New Models

When contributing new model architectures:

1. **Follow the existing structure**:
   - Implement models in `openretina/models/`
   - Use PyTorch Lightning for training logic
   - Follow naming conventions for consistency

2. **Include comprehensive documentation, if you can**:
   - Docstrings for all classes and methods
   - Example usage in documentation
   - Reference to original paper where the method is coming from, if applicable

3. **Add tests**:
   - Unit tests for model components
   - Integration tests for training/inference

### Adding New Datasets

When adding support for new datasets:

1. **Implement in `openretina/data_io/`**:
   - Create a new subdirectory for your dataset
   - Implement data loaders following the base classes
   - Include stimulus and response handling

2. **Follow data conventions**:
   - Use consistent data formats across datasets
   - Include proper metadata handling

3. **Provide examples**:
   - Add example usage in documentation
   - Include data format specifications
   - Provide download instructions if data is publicly available, or add it to our [HuggingFace](https://huggingface.co/datasets/open-retina/open-retina)

### Documentation Guidelines

When contributing to documentation:

1. **Use clear, concise language**
2. **Include code examples** where applicable
3. **Follow the existing documentation structure**
4. **Update the navigation** in `mkdocs.yml` if adding new pages
5. **Test documentation builds** locally before submitting

For detailed documentation standards, see our [documentation writing guidelines](#documentation-writing-guidelines) below.

## Testing

### Running Tests

```bash
# Run all tests
make test-all

# Run specific test categories
make test-types        # Type checking with mypy
make test-codestyle    # Code style with ruff
make test-formatting   # Code formatting with ruff
make test-unittests    # Unit tests with pytest

# Test notebooks
make test-notebooks
```

### Writing Tests

- Write unit tests for all new functions and classes
- Use `pytest` for test framework
- Place tests in the `tests/` directory
- Follow existing test structure and naming conventions
- Include both positive and negative test cases
- Test edge cases and error conditions

## Documentation Writing Guidelines

When writing or updating documentation:

### General Principles

1. **Be accurate**: Always reflect the actual code behavior
2. **Be complete**: Provide all necessary information for users
3. **Be clear**: Use simple, direct language
4. **Be consistent**: Follow established patterns and terminology
5. **Include examples**: Show don't just tell

### Docstring Standards

Use Google-style docstrings for all public functions and classes:

```python
def example_function(param1: int, param2: str = "default") -> bool:
    """Brief description of what the function does.
    
    Longer description if needed. Explain the purpose, behavior,
    and any important details.
    
    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with default value.
        
    Returns:
        Description of what the function returns.
        
    Raises:
        ValueError: When and why this exception is raised.
        
    Example:
        >>> result = example_function(42, "test")
        >>> print(result)
        True
    """
```

### API Documentation

For API reference pages:

1. **Use mkdocstrings** for automatic documentation generation
2. **Include module-level documentation** explaining the purpose
3. **Group related functions/classes** logically
4. **Provide usage examples** for complex APIs

### Tutorial Documentation

For tutorials and guides:

1. **Start with clear objectives**: What will the user learn?
2. **Provide complete, runnable examples**
3. **Explain each step** and why it's necessary
4. **Include expected outputs** where helpful
5. **Address common issues** and troubleshooting

### Code Examples

- **Always test code examples** to ensure they work
- **Use realistic data** in examples when possible
- **Show imports** and setup required
- **Keep examples focused** on the specific concept being demonstrated

## Code Review Process

1. **All contributions require review** before merging
2. **Address all feedback** before expecting merge
3. **Maintain backwards compatibility** unless explicitly breaking changes
4. **Update version numbers** appropriately for significant changes

## Getting Help

If you need help with contributing:

- **Check existing documentation** first
- **Open an issue** on GitHub for bugs or feature requests
- **Join discussions** in existing issues and pull requests
- **Contact maintainers** if you have questions about the contribution process

## Recognition

Contributors are recognized in:

- The project's contributor list, and Zenodo's software reference.
- Release notes for significant contributions
- Academic publications when appropriate

Thank you for contributing to `openretina`!
