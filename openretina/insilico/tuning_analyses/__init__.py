"""Tuning analyses: probe a trained model with parametrized stimulus families.

Unlike :mod:`openretina.insilico.stimulus_optimization`, which *searches* for a stimulus,
the modules here *sweep* a low-dimensional parameter grid and record how the model's
response varies over it.
"""
