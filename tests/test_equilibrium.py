from __future__ import annotations

import numpy as np

from contact3d.mechanics import (
    DeadLoad,
    DirichletConstraints,
    EquilibriumProblem,
    NeoHookeanMaterial,
    SparseAccumulator,
    Tet4Mesh,
    Tet4Sparsity,
    assemble_tet4_sparse,
    evaluate_tet4_mesh,
)
from contact3d.solvers import NewtonOptions, solve_equilibrium, solve_load_steps


def cube_star_mesh() -> Tet4Mesh:
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
    surface_triangles = [
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
    elements = np.array(
        [(8, *triangle) for triangle in surface_triangles],
        dtype=np.int64,
    )
    return Tet4Mesh(nodes, elements)


def manufactured_problem(
    deformation: np.ndarray,
) -> tuple[EquilibriumProblem, np.ndarray]:
    mesh = cube_star_mesh()
    material = NeoHookeanMaterial.from_young_poisson(210.0, 0.3)
    target = mesh.reference_nodes @ deformation.T - mesh.reference_nodes
    fixed_nodes = np.flatnonzero(mesh.reference_nodes[:, 2] == 0.0)
    constraints = DirichletConstraints.fixed_nodes(fixed_nodes)
    evaluation = evaluate_tet4_mesh(mesh, target, material)
    force = evaluation.residual.ravel().copy()
    force[constraints.dofs] = 0.0
    return EquilibriumProblem(mesh, material, constraints, DeadLoad(force)), target


def test_sparse_accumulator_combines_duplicate_entries() -> None:
    accumulator = SparseAccumulator((3, 3))
    rows = np.array([0, 2])
    columns = np.array([0, 1])
    accumulator.add_block(rows, columns, np.array([[1.0, 2.0], [3.0, 4.0]]))
    accumulator.add_block(rows, columns, np.ones((2, 2)))
    matrix = accumulator.to_csr()

    np.testing.assert_allclose(
        matrix.to_dense(),
        np.array([[2.0, 3.0, 0.0], [0.0, 0.0, 0.0], [4.0, 5.0, 0.0]]),
    )
    np.testing.assert_allclose(
        matrix.matvec(np.array([2.0, -1.0, 7.0])),
        np.array([1.0, 0.0, 3.0]),
    )
    np.testing.assert_allclose(
        matrix.extract_dense(np.array([2, 0]), np.array([1, 0])),
        np.array([[5.0, 4.0], [3.0, 2.0]]),
    )


def test_sparse_tet4_assembly_matches_dense_assembly() -> None:
    mesh = cube_star_mesh()
    material = NeoHookeanMaterial.from_young_poisson(210.0, 0.3)
    rng = np.random.default_rng(4)
    displacement = 0.04 * rng.normal(size=mesh.reference_nodes.shape)
    dense = evaluate_tet4_mesh(mesh, displacement, material)
    sparsity = Tet4Sparsity.from_mesh(mesh)
    sparse = assemble_tet4_sparse(
        mesh,
        displacement,
        material,
        sparsity=sparsity,
    )

    np.testing.assert_allclose(sparse.residual, dense.residual, atol=1.0e-13)
    np.testing.assert_allclose(sparse.tangent.to_dense(), dense.tangent, atol=1.0e-13)
    assert sparse.tangent.nnz == sparsity.nnz
    assert sparse.tangent.nnz < dense.tangent.size


def test_dirichlet_constraints_sort_and_apply() -> None:
    constraints = DirichletConstraints(
        np.array([5, 0, 2]),
        np.array([0.5, -0.1, 0.2]),
    )
    result = constraints.apply(np.zeros(9))

    np.testing.assert_array_equal(constraints.dofs, np.array([0, 2, 5]))
    np.testing.assert_allclose(result[[0, 2, 5]], np.array([-0.1, 0.2, 0.5]))
    np.testing.assert_array_equal(
        constraints.free_dofs(9),
        np.array([1, 3, 4, 6, 7, 8]),
    )


def test_newton_recovers_manufactured_large_deformation() -> None:
    deformation = np.array(
        [[1.0, 0.0, 0.45], [0.0, 1.0, 0.10], [0.0, 0.0, 0.72]]
    )
    problem, target = manufactured_problem(deformation)
    result = solve_equilibrium(
        problem,
        options=NewtonOptions(
            maximum_iterations=30,
            absolute_tolerance=1.0e-12,
            relative_tolerance=1.0e-12,
        ),
    )

    assert result.converged
    assert result.termination_reason == "converged"
    free = problem.constraints.free_dofs(3 * problem.mesh.node_count)
    np.testing.assert_allclose(
        result.displacement[free],
        target.ravel()[free],
        atol=2.0e-12,
        rtol=2.0e-12,
    )
    assert result.evaluation.free_residual_norm < 2.0e-12
    assert result.evaluation.bulk.minimum_jacobian > 0.7
    assert min(item.accepted_step for item in result.history) < 1.0


def test_load_steps_converge_and_reactions_balance_external_load() -> None:
    deformation = np.array(
        [[1.0, 0.0, 0.35], [0.0, 1.0, 0.06], [0.0, 0.0, 0.78]]
    )
    problem, target = manufactured_problem(deformation)
    results = solve_load_steps(
        problem,
        np.array([0.25, 0.5, 0.75, 1.0]),
        options=NewtonOptions(
            absolute_tolerance=2.0e-10,
            relative_tolerance=2.0e-10,
        ),
    )

    assert len(results) == 4
    assert all(result.converged for result in results)
    final = results[-1]
    free = final.evaluation.free_dofs
    np.testing.assert_allclose(
        final.displacement[free],
        target.ravel()[free],
        atol=3.0e-10,
        rtol=3.0e-10,
    )
    reaction = np.sum(final.evaluation.reaction.reshape((-1, 3)), axis=0)
    external = np.sum(problem.load.force.reshape((-1, 3)), axis=0)
    np.testing.assert_allclose(reaction + external, np.zeros(3), atol=2.0e-10)


def test_iteration_limit_returns_machine_readable_failure() -> None:
    deformation = np.array(
        [[1.0, 0.0, 0.35], [0.0, 1.0, 0.06], [0.0, 0.0, 0.78]]
    )
    problem, _ = manufactured_problem(deformation)
    result = solve_equilibrium(
        problem,
        options=NewtonOptions(
            maximum_iterations=1,
            absolute_tolerance=1.0e-14,
            relative_tolerance=1.0e-14,
        ),
    )

    assert not result.converged
    assert result.termination_reason == "maximum_iterations"
    assert result.iteration_count == 1
