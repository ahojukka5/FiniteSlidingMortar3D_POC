"""Runnable nonmatching sandwiched-beam bending example."""

from .direct import run
from .model import build_model
from .run import solver_options

__all__ = ["build_model", "run", "solver_options"]
