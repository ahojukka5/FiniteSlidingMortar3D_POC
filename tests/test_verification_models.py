from __future__ import annotations

import numpy as np
import pytest

from contact3d import LinearSolverOptions, NewtonOptions, solve_coupled_equilibrium
from contact3d.verification_models import stacked_matching_block_contact_model


def _options(backend: str) -> NewtonOptions:
    return NewtonOptions(
        maximum_iterations=40,
        absolute_tolerance=1.0e-10,
        relative_tolerance=1.0e-10,
        linear_solver=LinearSolverOptions(
            backend=backend,
            relative_tolerance=1.0e-12,
            absolute_tolerance=1.0e-13,
        ),
    )


def test_stacked_block_model_has_expected_refinement_counts() -> None:
    model = stacked_matching_block_contact_model(2, layers=2)
    problem = model.problem

    assert problem.mesh.node_count == 54
    assert problem.mesh.element_count == 96
    assert model.total_dofs == 162
    assert len(model.free_dofs) == 108
    assert model.interface_count == 4
    assert model.interface_area == pytest.approx(0.25)
    assert problem.mesh.reference_volume == pytest.approx(1.0)
    assert len(model.bottom_nodes) == 9
    assert len(model.top_nodes) == 9
    assert all(interface.area == pytest.approx(0.25) for interface in problem.interfaces)


def test_stacked_block_dense_and_sparse_lu_solutions_agree() -> None:
    model = stacked_matching_block_contact_model(1, layers=1, indentation=0.03)
    problem = model.problem
    states = problem.initial_states()

    dense = solve_coupled_equilibrium(
        problem,
        states,
        event_policy="restart",
        options=_options("dense"),
    )
    sparse = solve_coupled_equilibrium(
        problem,
        states,
        event_policy="restart",
        options=_options("sparse_lu"),
    )

    assert dense.converged
    assert sparse.converged
    np.testing.assert_allclose(
        sparse.displacement,
        dense.displacement,
        atol=2.0e-10,
        rtol=2.0e-10,
    )
    assert sparse.history
    assert all(row.linear_solve.backend == "sparse_lu" for row in sparse.history)
    assert all(not row.linear_solve.materialized_dense for row in sparse.history)
