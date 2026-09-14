"""Lightning callbacks used by the training CLI.

Deliberately non-empty: ``pyproject.toml`` uses ``setuptools`` ``find`` (not ``find_namespace``),
so without an ``__init__.py`` this sub-package -- and every module in it -- is silently dropped
from an installed wheel.
"""
