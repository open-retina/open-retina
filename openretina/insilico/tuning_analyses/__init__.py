"""Tuning analyses: probe a trained model with parametrized stimulus families.

Unlike :mod:`openretina.insilico.stimulus_optimization`, which *searches* for a stimulus,
the modules here *sweep* a low-dimensional parameter grid and record how the model's
response varies over it.

This file is deliberately non-empty: ``pyproject.toml`` uses ``setuptools`` ``find``
(not ``find_namespace``), so without an ``__init__.py`` this sub-package -- and every
module in it -- is silently dropped from an installed wheel.
"""
