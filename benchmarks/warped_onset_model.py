"""Compatibility access to the user-facing nonmatching contact-patch model."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from examples.contact_patch.model import (
        ContactPatchModel,
        _minimum_reference_determinant,
        build_model,
        solver_options,
    )
except ModuleNotFoundError as error:  # Direct execution from the repository root.
    if error.name != "examples":
        raise
    repository_root = Path(__file__).resolve().parents[1]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from examples.contact_patch.model import (
        ContactPatchModel,
        _minimum_reference_determinant,
        build_model,
        solver_options,
    )

BenchmarkModel = ContactPatchModel
model = build_model
options = solver_options

__all__ = [
    "BenchmarkModel",
    "ContactPatchModel",
    "_minimum_reference_determinant",
    "build_model",
    "model",
    "options",
    "solver_options",
]
