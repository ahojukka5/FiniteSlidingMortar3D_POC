"""Compatibility façade for linear solver algorithms.

Linear solver implementation lives under :mod:`contact3d.solvers.linear`.
This flat module remains only until the compatibility cleanup in #136.
"""

from .solvers.linear import (
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

__all__ = [
    "LinearBackend",
    "LinearPreconditioner",
    "LinearSolveDiagnostics",
    "LinearSolveResult",
    "LinearSolverOptions",
    "PreconditionerFactory",
    "PreconditionerKind",
    "block_jacobi_preconditioner_factory",
    "extract_csr_submatrix",
    "field_split_preconditioner_factory",
    "solve_linear_system",
    "solve_reduced_system",
]
