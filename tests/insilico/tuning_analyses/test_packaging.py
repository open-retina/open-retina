"""Pins that every openretina sub-package is actually shipped.

``pyproject.toml`` leaves ``[tool.setuptools.packages]`` empty, so setuptools auto-discovers with
``find`` rather than ``find_namespace``. ``find`` only walks directories that contain an
``__init__.py``, so a sub-package without one is dropped from the built wheel silently -- the source
tree keeps working (editable installs resolve through a path hook) and only an installed wheel is
missing the modules. ``tuning_analyses`` was in exactly that state, taking ``response_gradient``
with it.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "openretina"


def _directories_holding_modules() -> list[Path]:
    return [
        directory
        for directory in PACKAGE_ROOT.rglob("*")
        if directory.is_dir()
        and not directory.name.startswith((".", "__"))
        and any(entry.suffix == ".py" for entry in directory.iterdir() if entry.is_file())
    ]


def test_every_directory_of_modules_is_a_package() -> None:
    assert PACKAGE_ROOT.is_dir(), f"expected the package at {PACKAGE_ROOT}"

    missing = sorted(
        directory.relative_to(PACKAGE_ROOT).as_posix()
        for directory in _directories_holding_modules()
        if not (directory / "__init__.py").exists()
    )

    assert not missing, (
        "these directories contain modules but no __init__.py, so setuptools' `find` drops them "
        f"from any built wheel: {missing}"
    )


def test_response_gradient_is_importable_from_its_package() -> None:
    """The module that the missing __init__.py was actually costing us."""
    assert (PACKAGE_ROOT / "insilico" / "tuning_analyses" / "__init__.py").exists()

    from openretina.insilico.tuning_analyses import response_gradient

    assert hasattr(response_gradient, "get_gradient_grid")
