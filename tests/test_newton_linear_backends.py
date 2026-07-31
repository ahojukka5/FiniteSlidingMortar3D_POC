from __future__ import annotations

import numpy as np

from contact3d import (
    CoupledEquilibriumProblem,
    DeadLoad,
    DirichletConstraints,
    EquilibriumProblem,
    LinearSolveDiagnostics,
    LinearSolveResult,
    LinearSolverOptions,
    NeoHookeanMaterial,
    NewtonOptions,
    Tet4Mesh,
    evaluate_tet4_mesh,
    solve_coupled_equilibrium,
    solve_equilibrium,
)
from contact3d.coupled_oracle import FrozenMatchingMortarInterface


def _cube_star_mesh() -> Tet4Mesh:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )
    triangles = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ]
    elements = np.asarray([(8, *triangle) for triangle in triangles], dtype=np.int64)
    return Tet4Mesh(nodes, elements)


def _bulk_problem() -> tuple[EquilibriumProblem, np.ndarray]:
    mesh = _cube_star_mesh()
    material = NeoHookeanMaterial.from_young_poisson(210.0, 0.3)
    deformation = np.array(
        [[1.0, 0.0, 0.45], [0.0, 1.0, 0.10], [0.0, 0.0, 0.72]]
    )
    target = mesh.reference_nodes @ deformation.T - mesh.reference_nodes
    fixed_nodes = np.flatnonzero(mesh.reference_nodes[:, 2] == 0.0)
    constraints = DirichletConstraints.fixed_nodes(fixed_nodes)
    force = evaluate_tet4_mesh(mesh, target, material).residual.ravel().copy()
    force[constraints.dofs] = 0.0
    return EquilibriumProblem(mesh, material, constraints, DeadLoad(force)), target


def _block_nodes(z_origin: float) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, z_origin],
            [1.0, 0.0, z_origin],
            [1.0, 1.0, z_origin],
            [0.0, 1.0, z_origin],
            [0.0, 0.0, z_origin + 1.0],
            [1.0, 0.0, z_origin + 1.0],
            [1.0, 1.0, z_origin + 1.0],
            [0.0, 1.0, z_origin + 1.0],
            [0.5, 0.5, z_origin + 0.5],
        ]
    )


def _block_elements(offset: int) -> np.ndarray:
    triangles = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ]
    return np.asarray(
        [(offset + 8, offset + a, offset + b, offset + c) for a, b, c in triangles],
        dtype=np.int64,
    )


def _coupled_problem() -> CoupledEquilibriumProblem:
    nodes = np.vstack([_block_nodes(0.0), _block_nodes(1.0)])
    elements = np.vstack([_block_elements(0), _block_elements(9)])
    mesh = Tet4Mesh(nodes, elements)
    interface = FrozenMatchingMortarInterface(
        np.array([9, 12, 11, 10], dtype=np.int64),
        np.array([4, 7, 6, 5], dtype=np.int64),
        np.array([0.0, 0.0, -1.0]),
        6400.0,
    )
    dofs: list[int] = []
    values: list[float] = []
    for node in (0, 1, 2, 3):
        for component in range(3):
            dofs.append(3 * node + component)
            values.append(0.0)
    for node in (13, 14, 15, 16):
        for component in range(3):
            dofs.append(3 * node + component)
            values.append(-0.12 if component == 2 else 0.0)
    return CoupledEquilibriumProblem(
        mesh,
        NeoHookeanMaterial.from_young_poisson(210.0, 0.3),
        DirichletConstraints(np.asarray(dofs), np.asarray(values)),
        DeadLoad(np.zeros(3 * mesh.node_count)),
        (interface,),
    )


def _sparse_newton_options(*, maximum_iterations: int = 40) -> NewtonOptions:
    return NewtonOptions(
        maximum_iterations=maximum_iterations,
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-10,
        linear_solver=LinearSolverOptions(
            backend="auto",
            dense_threshold=0,
        ),
    )


def test_bulk_newton_uses_reduced_sparse_backend() -> None:
    problem, target = _bulk_problem()
    result = solve_equilibrium(problem, options=_sparse_newton_options())

    assert result.converged
    assert result.linear_solve_failure is None
    assert result.history
    free = problem.constraints.free_dofs(3 * problem.mesh.node_count)
    np.testing.assert_allclose(
        result.displacement[free],
        target.ravel()[free],
        atol=3.0e-10,
        rtol=3.0e-10,
    )
    for row in result.history:
        diagnostics = row.linear_solve
        assert diagnostics.backend == "sparse_lu"
        assert not diagnostics.materialized_dense
        assert diagnostics.matrix_shape == (len(free), len(free))
        assert diagnostics.matrix_nnz < len(free) ** 2


def test_coupled_newton_uses_nonsymmetric_sparse_backend() -> None:
    problem = _coupled_problem()
    result = solve_coupled_equilibrium(
        problem,
        problem.initial_states(),
        event_policy="restart",
        options=_sparse_newton_options(),
    )

    assert result.converged
    assert result.linear_solve_failure is None
    assert result.history
    assert result.contact_event_restarts >= 1
    free_count = len(result.evaluation.free_dofs)
    for row in result.history:
        diagnostics = row.linear_solve
        assert diagnostics.backend == "sparse_lu"
        assert diagnostics.converged
        assert not diagnostics.materialized_dense
        assert diagnostics.matrix_shape == (free_count, free_count)


def test_bulk_newton_retains_linear_failure(monkeypatch) -> None:
    import contact3d.equilibrium as module

    problem, _ = _bulk_problem()
    failure = LinearSolveDiagnostics(
        requested_backend="gmres",
        backend="gmres",
        preconditioner="none",
        converged=False,
        iterations=1,
        residual_norm=1.0,
        relative_residual=1.0,
        residual_history=(1.0,),
        setup_seconds=0.0,
        solve_seconds=0.0,
        matrix_shape=(15, 15),
        matrix_nnz=1,
        materialized_dense=False,
        failure_reason="maximum_iterations",
    )
    monkeypatch.setattr(
        module,
        "solve_reduced_system",
        lambda *args, **kwargs: LinearSolveResult(None, failure),
    )

    result = solve_equilibrium(problem)

    assert not result.converged
    assert result.termination_reason == "linear_solve_failed"
    assert result.linear_solve_failure is failure
    assert not result.history
