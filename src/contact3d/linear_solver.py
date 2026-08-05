"""Configurable dense and SciPy sparse linear-system backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .mechanics import CSRMatrix
from .mechanics.model import FloatArray, IntArray

LinearBackend = Literal["auto", "dense", "sparse_lu", "gmres", "bicgstab"]
PreconditionerKind = Literal["none", "jacobi", "ilu"]


@runtime_checkable
class LinearPreconditioner(Protocol):
    """Matrix-dependent left preconditioner used by Krylov backends."""

    def apply(self, vector: FloatArray) -> FloatArray: ...


PreconditionerFactory = Callable[[CSRMatrix], LinearPreconditioner]


@dataclass(frozen=True, slots=True)
class LinearSolverOptions:
    """Linear backend and stopping settings used by Newton iterations."""

    backend: LinearBackend = "dense"
    preconditioner: PreconditionerKind = "none"
    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 0.0
    maximum_iterations: int = 500
    restart: int = 50
    dense_threshold: int = 96
    ilu_drop_tolerance: float = 1.0e-4
    ilu_fill_factor: float = 10.0
    preconditioner_factory: PreconditionerFactory | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.backend not in ("auto", "dense", "sparse_lu", "gmres", "bicgstab"):
            raise ValueError("unsupported linear-solver backend")
        if self.preconditioner not in ("none", "jacobi", "ilu"):
            raise ValueError("unsupported preconditioner")
        if self.maximum_iterations <= 0 or self.restart <= 0:
            raise ValueError("linear iteration limits must be positive")
        if self.dense_threshold < 0:
            raise ValueError("dense_threshold must be nonnegative")
        for name, value in (
            ("relative_tolerance", self.relative_tolerance),
            ("absolute_tolerance", self.absolute_tolerance),
            ("ilu_drop_tolerance", self.ilu_drop_tolerance),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.ilu_fill_factor) or self.ilu_fill_factor <= 0.0:
            raise ValueError("ilu_fill_factor must be finite and positive")
        if self.relative_tolerance == 0.0 and self.absolute_tolerance == 0.0:
            raise ValueError("at least one linear tolerance must be positive")
        if self.backend in ("dense", "sparse_lu") and (
            self.preconditioner != "none" or self.preconditioner_factory is not None
        ):
            raise ValueError("preconditioners require a Krylov or auto backend")


@dataclass(frozen=True, slots=True)
class LinearSolveDiagnostics:
    """Machine-readable timing and convergence record for one linear solve."""

    requested_backend: LinearBackend
    backend: str
    preconditioner: str
    converged: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    residual_history: tuple[float, ...]
    setup_seconds: float
    solve_seconds: float
    matrix_shape: tuple[int, int]
    matrix_nnz: int
    materialized_dense: bool
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LinearSolveResult:
    solution: FloatArray | None
    diagnostics: LinearSolveDiagnostics


def extract_csr_submatrix(
    matrix: CSRMatrix,
    rows: IntArray,
    columns: IntArray,
) -> CSRMatrix:
    """Extract a deterministic CSR submatrix in the requested index ordering."""

    row_indices = np.asarray(rows, dtype=np.int64)
    column_indices = np.asarray(columns, dtype=np.int64)
    if row_indices.ndim != 1 or column_indices.ndim != 1:
        raise ValueError("CSR submatrix indices must be one-dimensional")
    if np.any(row_indices < 0) or np.any(row_indices >= matrix.shape[0]):
        raise ValueError("requested row is out of range")
    if np.any(column_indices < 0) or np.any(column_indices >= matrix.shape[1]):
        raise ValueError("requested column is out of range")
    if len(np.unique(row_indices)) != len(row_indices):
        raise ValueError("requested rows must be unique")
    if len(np.unique(column_indices)) != len(column_indices):
        raise ValueError("requested columns must be unique")

    lookup = {int(column): position for position, column in enumerate(column_indices)}
    indptr = [0]
    output_indices: list[int] = []
    output_data: list[float] = []
    for row in row_indices:
        start = int(matrix.indptr[int(row)])
        stop = int(matrix.indptr[int(row) + 1])
        entries = [
            (lookup[int(column)], float(value))
            for column, value in zip(
                matrix.indices[start:stop],
                matrix.data[start:stop],
                strict=True,
            )
            if int(column) in lookup
        ]
        entries.sort(key=lambda item: item[0])
        output_indices.extend(column for column, _ in entries)
        output_data.extend(value for _, value in entries)
        indptr.append(len(output_indices))
    return CSRMatrix(
        (len(row_indices), len(column_indices)),
        np.asarray(indptr, dtype=np.int64),
        np.asarray(output_indices, dtype=np.int64),
        np.asarray(output_data, dtype=float),
    )


def _relative_residual(
    matrix: CSRMatrix,
    solution: FloatArray,
    right_hand_side: FloatArray,
) -> tuple[float, float]:
    residual_norm = float(np.linalg.norm(matrix.matvec(solution) - right_hand_side))
    scale = max(float(np.linalg.norm(right_hand_side)), np.finfo(float).tiny)
    return residual_norm, residual_norm / scale


def _record(
    settings: LinearSolverOptions,
    matrix: CSRMatrix,
    backend: str,
    preconditioner: str,
    converged: bool,
    iterations: int,
    residual_norm: float,
    relative_residual: float,
    history: list[float] | tuple[float, ...],
    setup_seconds: float,
    solve_seconds: float,
    materialized_dense: bool,
    failure_reason: str | None = None,
) -> LinearSolveDiagnostics:
    return LinearSolveDiagnostics(
        requested_backend=settings.backend,
        backend=backend,
        preconditioner=preconditioner,
        converged=converged,
        iterations=iterations,
        residual_norm=float(residual_norm),
        relative_residual=float(relative_residual),
        residual_history=tuple(float(value) for value in history),
        setup_seconds=float(setup_seconds),
        solve_seconds=float(solve_seconds),
        matrix_shape=matrix.shape,
        matrix_nnz=matrix.nnz,
        materialized_dense=materialized_dense,
        failure_reason=failure_reason,
    )


def _scipy_modules():
    try:
        from scipy.sparse import csc_matrix, csr_matrix
        from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, spilu, splu
    except ImportError as error:
        raise RuntimeError(
            "SciPy sparse backends require the 'sparse' optional dependency"
        ) from error
    return csc_matrix, csr_matrix, LinearOperator, bicgstab, gmres, spilu, splu


def _scipy_csr(matrix: CSRMatrix):
    _, csr_matrix, *_ = _scipy_modules()
    return csr_matrix(
        (matrix.data, matrix.indices, matrix.indptr),
        shape=matrix.shape,
        copy=True,
    )


def _resolved_backend(settings: LinearSolverOptions, matrix: CSRMatrix) -> str:
    if settings.backend != "auto":
        return settings.backend
    if matrix.shape[0] <= settings.dense_threshold:
        return "dense"
    if settings.preconditioner != "none" or settings.preconditioner_factory:
        return "gmres"
    return "sparse_lu"


def _preconditioner(matrix: CSRMatrix, settings: LinearSolverOptions):
    _, _, linear_operator, _, _, spilu, _ = _scipy_modules()
    if settings.preconditioner_factory is not None:
        preconditioner = settings.preconditioner_factory(matrix)
        if not isinstance(preconditioner, LinearPreconditioner):
            raise TypeError("preconditioner_factory returned an invalid object")
        return linear_operator(matrix.shape, matvec=preconditioner.apply), "custom"
    if settings.preconditioner == "none":
        return None, "none"
    sparse = _scipy_csr(matrix)
    if settings.preconditioner == "ilu":
        factor = spilu(
            sparse.tocsc(),
            drop_tol=settings.ilu_drop_tolerance,
            fill_factor=settings.ilu_fill_factor,
        )
        return linear_operator(matrix.shape, matvec=factor.solve), "ilu"

    diagonal = sparse.diagonal()
    tolerance = np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(diagonal), initial=0.0)),
    )
    if np.any(np.abs(diagonal) <= tolerance):
        raise np.linalg.LinAlgError("Jacobi preconditioner has a zero diagonal")
    inverse = 1.0 / diagonal
    return linear_operator(matrix.shape, matvec=lambda vector: inverse * vector), "jacobi"


def solve_linear_system(
    matrix: CSRMatrix,
    right_hand_side: FloatArray,
    *,
    options: LinearSolverOptions | None = None,
) -> LinearSolveResult:
    """Solve a square CSR system and retain backend diagnostics."""

    settings = LinearSolverOptions() if options is None else options
    rhs = np.asarray(right_hand_side, dtype=float)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("linear system matrix must be square")
    if rhs.shape != (matrix.shape[0],) or not np.all(np.isfinite(rhs)):
        raise ValueError("right_hand_side must be a finite vector matching the matrix")
    backend = _resolved_backend(settings, matrix)
    rhs_norm = float(np.linalg.norm(rhs))
    if matrix.shape[0] == 0 or rhs_norm == 0.0:
        diagnostics = _record(
            settings,
            matrix,
            backend,
            "none",
            True,
            0,
            0.0,
            0.0,
            (),
            0.0,
            0.0,
            False,
        )
        return LinearSolveResult(np.zeros_like(rhs), diagnostics)

    if backend == "dense":
        setup_start = perf_counter()
        dense = matrix.to_dense()
        setup_seconds = perf_counter() - setup_start
        solve_start = perf_counter()
        try:
            solution = np.linalg.solve(dense, rhs)
        except np.linalg.LinAlgError:
            diagnostics = _record(
                settings,
                matrix,
                backend,
                "none",
                False,
                0,
                np.inf,
                np.inf,
                (),
                setup_seconds,
                perf_counter() - solve_start,
                True,
                "singular_matrix",
            )
            return LinearSolveResult(None, diagnostics)
        solve_seconds = perf_counter() - solve_start
        residual_norm, relative = _relative_residual(matrix, solution, rhs)
        converged = bool(np.all(np.isfinite(solution)))
        diagnostics = _record(
            settings,
            matrix,
            backend,
            "none",
            converged,
            1,
            residual_norm,
            relative,
            (relative,),
            setup_seconds,
            solve_seconds,
            True,
            None if converged else "nonfinite_solution",
        )
        return LinearSolveResult(solution if converged else None, diagnostics)

    try:
        csc_matrix, _, _, bicgstab, gmres, _, splu = _scipy_modules()
    except RuntimeError:
        diagnostics = _record(
            settings,
            matrix,
            backend,
            settings.preconditioner,
            False,
            0,
            np.inf,
            np.inf,
            (),
            0.0,
            0.0,
            False,
            "scipy_unavailable",
        )
        return LinearSolveResult(None, diagnostics)
    sparse = _scipy_csr(matrix)

    if backend == "sparse_lu":
        setup_start = perf_counter()
        try:
            factor = splu(csc_matrix(sparse))
        except RuntimeError:
            diagnostics = _record(
                settings,
                matrix,
                backend,
                "none",
                False,
                0,
                np.inf,
                np.inf,
                (),
                perf_counter() - setup_start,
                0.0,
                False,
                "factorization_failed",
            )
            return LinearSolveResult(None, diagnostics)
        setup_seconds = perf_counter() - setup_start
        solve_start = perf_counter()
        solution = factor.solve(rhs)
        solve_seconds = perf_counter() - solve_start
        residual_norm, relative = _relative_residual(matrix, solution, rhs)
        converged = bool(np.all(np.isfinite(solution)))
        diagnostics = _record(
            settings,
            matrix,
            backend,
            "none",
            converged,
            1,
            residual_norm,
            relative,
            (relative,),
            setup_seconds,
            solve_seconds,
            False,
            None if converged else "nonfinite_solution",
        )
        return LinearSolveResult(solution if converged else None, diagnostics)

    setup_start = perf_counter()
    try:
        preconditioner, preconditioner_name = _preconditioner(matrix, settings)
    except (RuntimeError, TypeError, np.linalg.LinAlgError):
        name = "custom" if settings.preconditioner_factory else settings.preconditioner
        diagnostics = _record(
            settings,
            matrix,
            backend,
            name,
            False,
            0,
            np.inf,
            np.inf,
            (),
            perf_counter() - setup_start,
            0.0,
            False,
            "preconditioner_failed",
        )
        return LinearSolveResult(None, diagnostics)
    setup_seconds = perf_counter() - setup_start
    history: list[float] = []
    solve_start = perf_counter()
    if backend == "gmres":
        solution, info = gmres(
            sparse,
            rhs,
            rtol=settings.relative_tolerance,
            atol=settings.absolute_tolerance,
            restart=settings.restart,
            maxiter=settings.maximum_iterations,
            M=preconditioner,
            callback=lambda value: history.append(float(value)),
            callback_type="pr_norm",
        )
    elif backend == "bicgstab":

        def record(iterate: FloatArray) -> None:
            _, relative = _relative_residual(matrix, np.asarray(iterate), rhs)
            history.append(relative)

        solution, info = bicgstab(
            sparse,
            rhs,
            rtol=settings.relative_tolerance,
            atol=settings.absolute_tolerance,
            maxiter=settings.maximum_iterations,
            M=preconditioner,
            callback=record,
        )
    else:
        raise AssertionError(f"unexpected backend {backend}")
    solve_seconds = perf_counter() - solve_start
    residual_norm, relative = _relative_residual(matrix, solution, rhs)
    threshold = max(settings.absolute_tolerance, settings.relative_tolerance * rhs_norm)
    roundoff = 10.0 * np.finfo(float).eps * rhs_norm
    converged = bool(info == 0 and residual_norm <= max(threshold, roundoff))
    if not history or history[-1] != relative:
        history.append(relative)
    failure_reason = None if converged else ("breakdown" if info < 0 else "maximum_iterations")
    if not np.all(np.isfinite(solution)):
        converged = False
        failure_reason = "nonfinite_solution"
    diagnostics = _record(
        settings,
        matrix,
        backend,
        preconditioner_name,
        converged,
        len(history),
        residual_norm,
        relative,
        history,
        setup_seconds,
        solve_seconds,
        False,
        failure_reason,
    )
    return LinearSolveResult(solution if converged else None, diagnostics)


def solve_reduced_system(
    matrix: CSRMatrix,
    free_dofs: IntArray,
    right_hand_side: FloatArray,
    *,
    options: LinearSolverOptions | None = None,
) -> LinearSolveResult:
    """Extract and solve the free-free block for strong Dirichlet reduction."""

    free = np.asarray(free_dofs, dtype=np.int64)
    reduced = extract_csr_submatrix(matrix, free, free)
    return solve_linear_system(reduced, right_hand_side, options=options)


@dataclass(frozen=True, slots=True)
class _BlockJacobiPreconditioner:
    blocks: tuple[IntArray, ...]
    inverse_blocks: tuple[FloatArray, ...]
    size: int

    def apply(self, vector: FloatArray) -> FloatArray:
        values = np.asarray(vector, dtype=float)
        if values.shape != (self.size,):
            raise ValueError("preconditioner vector has incompatible size")
        result = np.zeros_like(values)
        for indices, inverse in zip(self.blocks, self.inverse_blocks, strict=True):
            result[indices] = inverse @ values[indices]
        return result


def block_jacobi_preconditioner_factory(
    blocks: tuple[IntArray, ...],
) -> PreconditionerFactory:
    """Build independent dense block inverses for body partitions."""

    normalized = tuple(np.asarray(block, dtype=np.int64).copy() for block in blocks)

    def factory(matrix: CSRMatrix) -> LinearPreconditioner:
        if not normalized:
            raise ValueError("block partition must contain at least one block")
        joined = np.concatenate(normalized)
        expected = np.arange(matrix.shape[0])
        if len(joined) != len(expected) or not np.array_equal(np.sort(joined), expected):
            raise ValueError("block partition must cover every row exactly once")
        inverses = tuple(
            np.linalg.inv(extract_csr_submatrix(matrix, block, block).to_dense())
            for block in normalized
        )
        return _BlockJacobiPreconditioner(normalized, inverses, matrix.shape[0])

    return factory


def field_split_preconditioner_factory(
    fields: tuple[IntArray, ...],
) -> PreconditionerFactory:
    """Build block Jacobi for body/contact-aware field partitions."""

    return block_jacobi_preconditioner_factory(fields)
