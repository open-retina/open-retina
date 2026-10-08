"""Pins that every openretina sub-package is actually shipped.

setuptools' ``find`` only discovers directories with an ``__init__.py``; any other sub-package is
silently dropped from the built wheel, while editable installs keep working.
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
