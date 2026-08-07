"""Nonlinear and linear solution algorithms for coupled contact problems."""

from .augmented import solve_augmented_contact
from .linear import (
    LinearBackend,
    LinearPreconditioner,
    LinearSolveDiagnostics,
    LinearSolveResult,
    LinearSolverOptions,
    PreconditionerFactory,
    PreconditionerKind,
    block_jacobi_preconditioner_factory,
    extract_csr_submatrix,
    field_split_preconditioner_factory,
    solve_linear_system,
    solve_reduced_system,
)
from .newton import solve_coupled_equilibrium
from .results import (
    AugmentationIteration,
    AugmentedContactOptions,
    AugmentedContactResult,
    AugmentedTerminationReason,
    ContactEventPolicy,
    CoupledNewtonIteration,
    CoupledNewtonResult,
    CoupledTerminationReason,
    NewtonOptions,
)
from .scaling import (
    ScaleAwareAugmentationIteration,
    ScaleAwareAugmentedContactResult,
    ScaleAwareNewtonIteration,
    solve_scale_aware_augmented_contact,
)

__all__ = [
    "AugmentationIteration",
    "AugmentedContactOptions",
    "AugmentedContactResult",
    "AugmentedTerminationReason",
    "ContactEventPolicy",
    "CoupledNewtonIteration",
    "CoupledNewtonResult",
    "CoupledTerminationReason",
    "LinearBackend",
    "LinearPreconditioner",
    "LinearSolveDiagnostics",
    "LinearSolveResult",
    "LinearSolverOptions",
    "NewtonOptions",
    "PreconditionerFactory",
    "PreconditionerKind",
    "ScaleAwareAugmentationIteration",
    "ScaleAwareAugmentedContactResult",
    "ScaleAwareNewtonIteration",
    "block_jacobi_preconditioner_factory",
    "extract_csr_submatrix",
    "field_split_preconditioner_factory",
    "solve_augmented_contact",
    "solve_coupled_equilibrium",
    "solve_linear_system",
    "solve_reduced_system",
    "solve_scale_aware_augmented_contact",
]
