"""Compatibility façade for bulk equilibrium assembly and Newton solves.

Mechanics contracts live under :mod:`contact3d.mechanics` and Newton algorithms
live under :mod:`contact3d.solvers`. This flat module remains only until the
compatibility cleanup in #136.
"""

from .mechanics import (
    DeadLoad,
    DirichletConstraints,
    EquilibriumEvaluation,
    EquilibriumProblem,
    evaluate_equilibrium,
)
from .solvers.newton import solve_equilibrium, solve_load_steps
from .solvers.results import (
    NewtonIteration,
    NewtonOptions,
    NewtonResult,
)

__all__ = [
    "DeadLoad",
    "DirichletConstraints",
    "EquilibriumEvaluation",
    "EquilibriumProblem",
    "NewtonIteration",
    "NewtonOptions",
    "NewtonResult",
    "evaluate_equilibrium",
    "solve_equilibrium",
    "solve_load_steps",
]
