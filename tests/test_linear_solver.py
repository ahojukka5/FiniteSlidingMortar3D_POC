from __future__ import annotations

import numpy as np
import pytest

from contact3d import CSRMatrix
from contact3d.linear_solver import (
    LinearSolverOptions,
    block_jacobi_preconditioner_factory,
    extract_csr_submatrix,
    field_split_preconditioner_factory,
    solve_linear_system,
    solve_reduced_system,
)


def _csr(values: np.ndarray) -> CSRMatrix:
    matrix = np.asarray(values, dtype=float)
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for row in matrix:
        columns = np.flatnonzero(row)
        indices.extend(int(column) for column in columns)
        data.extend(float(row[column]) for column in columns)
        indptr.append(len(indices))
    return CSRMatrix(
        matrix.shape,
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(data, dtype=float),
    )


def _nonsymmetric_system() -> tuple[CSRMatrix, np.ndarray, np.ndarray]:
    dense = np.array(
        [
            [6.0, -2.0, 0.0, 0.0, 1.0],
            [1.0, 5.0, -1.0, 0.0, 0.0],
            [0.0, 2.0, 7.0, -3.0, 0.0],
            [0.0, 0.0, 1.0, 4.0, -1.0],
            [-2.0, 0.0, 0.0, 1.0, 5.0],
        ]
    )
    right_hand_side = np.array([2.0, -1.0, 3.0, 0.5, 4.0])
    return _csr(dense), right_hand_side, np.linalg.solve(dense, right_hand_side)


def test_dense_backend_matches_numpy_and_records_materialization() -> None:
    matrix, right_hand_side, expected = _nonsymmetric_system()

    result = solve_linear_system(matrix, right_hand_side)

    assert result.diagnostics.converged
    assert result.diagnostics.backend == "dense"
    assert result.diagnostics.materialized_dense
    assert result.diagnostics.iterations == 1
    assert result.diagnostics.failure_reason is None
    np.testing.assert_allclose(result.solution, expected, atol=2.0e-15)


@pytest.mark.parametrize(
    ("backend", "preconditioner"),
    [("sparse_lu", "none"), ("gmres", "jacobi"), ("bicgstab", "ilu")],
)
def test_scipy_backends_match_dense_without_dense_materialization(
    backend: str,
    preconditioner: str,
) -> None:
    pytest.importorskip("scipy")
    matrix, right_hand_side, expected = _nonsymmetric_system()
    options = LinearSolverOptions(
        backend=backend,
        preconditioner=preconditioner,
        relative_tolerance=1.0e-12,
        absolute_tolerance=1.0e-14,
    )

    result = solve_linear_system(matrix, right_hand_side, options=options)

    assert result.diagnostics.converged
    assert result.diagnostics.backend == backend
    assert not result.diagnostics.materialized_dense
    assert result.diagnostics.residual_history
    assert result.diagnostics.relative_residual <= 1.0e-11
    np.testing.assert_allclose(result.solution, expected, atol=1.0e-11)


def test_reduced_system_preserves_requested_dof_ordering() -> None:
    dense = np.array(
        [
            [8.0, 1.0, 2.0, 0.0],
            [3.0, 7.0, 0.0, 1.0],
            [4.0, 0.0, 6.0, 2.0],
            [0.0, 5.0, 1.0, 9.0],
        ]
    )
    matrix = _csr(dense)
    free = np.array([3, 1, 0])
    reduced = extract_csr_submatrix(matrix, free, free)
    right_hand_side = np.array([1.0, 2.0, 3.0])

    result = solve_reduced_system(matrix, free, right_hand_side)

    np.testing.assert_allclose(reduced.to_dense(), dense[np.ix_(free, free)])
    np.testing.assert_allclose(
        result.solution,
        np.linalg.solve(dense[np.ix_(free, free)], right_hand_side),
    )


def test_auto_selects_sparse_lu_above_dense_threshold() -> None:
    pytest.importorskip("scipy")
    matrix, right_hand_side, _ = _nonsymmetric_system()

    small = solve_linear_system(
        matrix,
        right_hand_side,
        options=LinearSolverOptions(backend="auto", dense_threshold=5),
    )
    sparse = solve_linear_system(
        matrix,
        right_hand_side,
        options=LinearSolverOptions(backend="auto", dense_threshold=4),
    )

    assert small.diagnostics.backend == "dense"
    assert sparse.diagnostics.backend == "sparse_lu"
    assert not sparse.diagnostics.materialized_dense


@pytest.mark.parametrize(
    "factory",
    [
        block_jacobi_preconditioner_factory(
            (np.array([0, 1, 2]), np.array([3, 4]))
        ),
        field_split_preconditioner_factory(
            (np.array([0, 2, 4]), np.array([1, 3]))
        ),
    ],
)
def test_custom_body_and_field_preconditioners(factory) -> None:
    pytest.importorskip("scipy")
    matrix, right_hand_side, expected = _nonsymmetric_system()
    options = LinearSolverOptions(
        backend="gmres",
        preconditioner_factory=factory,
        relative_tolerance=1.0e-12,
        absolute_tolerance=1.0e-14,
    )

    result = solve_linear_system(matrix, right_hand_side, options=options)

    assert result.diagnostics.converged
    assert result.diagnostics.preconditioner == "custom"
    np.testing.assert_allclose(result.solution, expected, atol=1.0e-11)


def test_singular_system_returns_machine_readable_failure() -> None:
    matrix = _csr(np.array([[1.0, 2.0], [2.0, 4.0]]))

    result = solve_linear_system(matrix, np.array([1.0, 2.0]))

    assert result.solution is None
    assert not result.diagnostics.converged
    assert result.diagnostics.failure_reason == "singular_matrix"
