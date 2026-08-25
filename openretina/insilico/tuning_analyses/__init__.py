"""Tuning analyses: probe a trained model with parametrized stimulus families.

Unlike :mod:`openretina.insilico.stimulus_optimization`, which *searches* for a stimulus,
the modules here *sweep* a low-dimensional parameter grid and record how the model's
response varies over it.

This file is deliberately non-empty: ``pyproject.toml`` uses ``setuptools`` ``find``
(not ``find_namespace``), so without an ``__init__.py`` this sub-package -- and every
module in it -- is silently dropped from an installed wheel.
"""

from openretina.insilico.tuning_analyses.behavior_modulation import (
    DEFAULT_SWEEP_VALUES,
    ResponseGrid,
    behavior_response_grid,
    pupil_center_response_grid,
    shift_to_core_pixels,
    shifter_shift_grid,
)

__all__ = [
    "DEFAULT_SWEEP_VALUES",
    "ResponseGrid",
    "behavior_response_grid",
    "pupil_center_response_grid",
    "shift_to_core_pixels",
    "shifter_shift_grid",
]
