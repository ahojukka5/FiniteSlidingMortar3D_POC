"""Linear-solver API used by nonlinear solution algorithms."""

from ..linear_solver import (
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
