from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from contact3d import (
    AugmentedContactOptions,
    CoupledEquilibriumProblem,
    DeadLoad,
    DirichletConstraints,
    MortarContactInterface,
    NeoHookeanMaterial,
    NewtonOptions,
    Tet4Mesh,
    evaluate_coupled_equilibrium,
    solve_augmented_contact,
    solve_coupled_equilibrium,
)
from contact3d.coupled_oracle import FrozenMatchingMortarInterface


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


def problem(*, penalty: float = 6400.0) -> CoupledEquilibriumProblem:
    nodes = np.vstack([_block_nodes(0.0), _block_nodes(1.0)])
    elements = np.vstack([_block_elements(0), _block_elements(9)])
    mesh = Tet4Mesh(nodes, elements)
    interface = FrozenMatchingMortarInterface(
        np.array([9, 12, 11, 10], dtype=np.int64),
        np.array([4, 7, 6, 5], dtype=np.int64),
        np.array([0.0, 0.0, -1.0]),
        penalty,
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


def options() -> AugmentedContactOptions:
    return AugmentedContactOptions(
        maximum_augmentations=16,
        gap_tolerance=1.0e-8,
        complementarity_tolerance=1.0e-7,
        projection_tolerance=1.0e-5,
        multiplier_tolerance=1.0e-8,
        newton=NewtonOptions(maximum_iterations=40),
    )


def test_coupled_sparsity_contains_cross_body_contact_blocks() -> None:
    coupled_problem = problem()
    lower_interface_dof = 3 * 4 + 2
    upper_interface_dof = 3 * 9 + 2
    dense = coupled_problem.sparsity.matrix(
        np.ones(coupled_problem.sparsity.nnz)
    ).to_dense()

    assert dense[lower_interface_dof, upper_interface_dof] == 1.0
    assert coupled_problem.sparsity.additional_positions[0].shape == (24, 24)


def test_coupled_tangent_matches_centered_difference() -> None:
    coupled_problem = problem()
    result = solve_augmented_contact(coupled_problem, options=options())
    assert result.converged

    base = evaluate_coupled_equilibrium(
        coupled_problem,
        result.displacement,
        result.states,
    )
    free = base.free_dofs
    rng = np.random.default_rng(4102)
    direction = np.zeros_like(result.displacement)
    direction[free] = rng.normal(size=len(free))
    direction /= np.linalg.norm(direction)
    assert base.tangent is not None
    exact = base.tangent.to_dense() @ direction

    step = 2.0e-7
    plus = evaluate_coupled_equilibrium(
        coupled_problem,
        result.displacement + step * direction,
        result.states,
        assemble_tangent=False,
    ).residual
    minus = evaluate_coupled_equilibrium(
        coupled_problem,
        result.displacement - step * direction,
        result.states,
        assemble_tangent=False,
    ).residual
    numerical = (plus - minus) / (2.0 * step)
    error = np.linalg.norm(exact[free] - numerical[free]) / np.linalg.norm(
        numerical[free]
    )

    assert error < 2.0e-8


def test_augmented_driver_converges_and_reduces_kkt_residuals() -> None:
    coupled_problem = problem()
    result = solve_augmented_contact(coupled_problem, options=options())

    assert result.converged
    assert result.termination_reason == "converged"
    assert result.history[0].maximum_penetration > 1.0e-3
    assert result.history[-1].maximum_penetration < 1.0e-8
    assert result.history[-1].maximum_projection_residual < 1.0e-5
    assert result.equilibrium.evaluation.free_residual_norm < 1.0e-10
    assert sum(row.contact_event_restarts for row in result.history) >= 1


def test_fixed_multiplier_newton_records_contact_event_restart() -> None:
    coupled_problem = problem()
    result = solve_coupled_equilibrium(
        coupled_problem,
        coupled_problem.initial_states(),
        event_policy="restart",
        options=NewtonOptions(maximum_iterations=40),
    )

    assert result.converged
    assert result.contact_event_restarts >= 1
    assert any(row.contact_branch_changed for row in result.history)


def test_reject_policy_does_not_cross_activation_event() -> None:
    coupled_problem = problem()
    result = solve_coupled_equilibrium(
        coupled_problem,
        coupled_problem.initial_states(),
        event_policy="reject",
        options=NewtonOptions(
            maximum_iterations=20,
            minimum_step=2.0**-12,
            maximum_line_search_iterations=14,
        ),
    )

    assert not result.converged
    assert result.termination_reason == "line_search_failed"


def test_mortar_mapping_validates_reference_coordinates() -> None:
    coupled_problem = problem()
    slave_nodes = np.array([9, 12, 11, 10], dtype=np.int64)
    master_nodes = np.array([4, 7, 6, 5], dtype=np.int64)
    pair = SimpleNamespace(
        slave=SimpleNamespace(
            node_count=4,
            reference_nodes=coupled_problem.mesh.reference_nodes[slave_nodes],
        ),
        master=SimpleNamespace(
            node_count=4,
            reference_nodes=coupled_problem.mesh.reference_nodes[master_nodes],
        ),
    )
    interface = MortarContactInterface(pair, slave_nodes, master_nodes)
    interface.validate_for(coupled_problem.mesh)
    assert interface.dofs.shape == (24,)

    shifted_pair = SimpleNamespace(
        slave=SimpleNamespace(
            node_count=4,
            reference_nodes=pair.slave.reference_nodes + 0.01,
        ),
        master=pair.master,
    )
    with pytest.raises(ValueError, match="slave contact reference coordinates"):
        MortarContactInterface(
            shifted_pair,
            slave_nodes,
            master_nodes,
        ).validate_for(coupled_problem.mesh)


def test_maximum_augmentation_returns_equilibrated_multiplier_state() -> None:
    coupled_problem = problem()
    result = solve_augmented_contact(
        coupled_problem,
        options=AugmentedContactOptions(
            maximum_augmentations=1,
            gap_tolerance=0.0,
            complementarity_tolerance=0.0,
            projection_tolerance=0.0,
            multiplier_tolerance=0.0,
            newton=NewtonOptions(maximum_iterations=40),
        ),
    )

    assert not result.converged
    assert result.termination_reason == "maximum_augmentations"
    assert result.states[0].augmentation == 0
    evaluated_state = result.equilibrium.evaluation.contacts[0].raw[0]
    np.testing.assert_allclose(result.states[0].multipliers, evaluated_state.multipliers)
