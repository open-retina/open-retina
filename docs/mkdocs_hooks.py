from __future__ import annotations

import shutil
from pathlib import Path

NOTEBOOK_NAMES = [
    "hoefling_gradient_analysis.ipynb",
    "hyperparameter_search_hoefling_2024.ipynb",
    # Add more notebook filenames here as needed
]


def on_pre_build(config):  # type: ignore[unused-argument]
    """Copy selected notebooks into docs/notebooks before each build/serve run."""
    repo_root = Path(__file__).resolve().parents[1]
    notebook_source = repo_root / "notebooks"
    docs_notebooks = repo_root / "docs" / "notebooks"
    docs_notebooks.mkdir(parents=True, exist_ok=True)

    for name in NOTEBOOK_NAMES:
        source_path = notebook_source / name
        if not source_path.exists():
            raise FileNotFoundError(f"Notebook not found: {source_path}")
        destination_path = docs_notebooks / name
        shutil.copy2(source_path, destination_path)
