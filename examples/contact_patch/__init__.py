"""Small runnable nonmatching frictionless contact-patch example."""

from .model import ContactPatchModel, build_model, solver_options
from .run import run

__all__ = ["ContactPatchModel", "build_model", "run", "solver_options"]
