"""Compatibility façade for coupling and coupled nonlinear-solver APIs.

New code should import assembly contracts from :mod:`contact3d.coupling` and
solution algorithms from :mod:`contact3d.solvers`. This module retains the
historical import path while migration continues.
"""

from .coupling import (
    ContactBranchSignature,
    ContactInterfaceEvaluation,
    ContactInterfaceUpdate,
    CoupledContactInterface,
    CoupledEquilibriumEvaluation,
    CoupledEquilibriumProblem,
    MortarContactInterface,
    evaluate_coupled_equilibrium,
)
from .solvers import (
    AugmentationIteration,
    AugmentedContactOptions,
    AugmentedContactResult,
    AugmentedTerminationReason,
    ContactEventPolicy,
    CoupledNewtonIteration,
    CoupledNewtonResult,
    CoupledTerminationReason,
    solve_augmented_contact,
    solve_coupled_equilibrium,
)
from .solvers.newton import _linear_failure_reason as _linear_failure_reason

__all__ = [
    "AugmentationIteration",
    "AugmentedContactOptions",
    "AugmentedContactResult",
    "AugmentedTerminationReason",
    "ContactBranchSignature",
    "ContactEventPolicy",
    "ContactInterfaceEvaluation",
    "ContactInterfaceUpdate",
    "CoupledContactInterface",
    "CoupledEquilibriumEvaluation",
    "CoupledEquilibriumProblem",
    "CoupledNewtonIteration",
    "CoupledNewtonResult",
    "CoupledTerminationReason",
    "MortarContactInterface",
    "evaluate_coupled_equilibrium",
    "solve_augmented_contact",
    "solve_coupled_equilibrium",
]
