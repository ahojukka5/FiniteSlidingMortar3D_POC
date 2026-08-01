from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from contact3d import RigidBodyBoundaryPath
from contact3d.equilibrium import DeadLoad, DirichletConstraints
from contact3d.load_path import LinearPathValue


@dataclass(frozen=True, slots=True)
class Mesh:
    reference_nodes: np.ndarray

    @property
    def node_count(self) -> int:
        return len(self.reference_nodes)


@dataclass(frozen=True, slots=True)
class Problem:
    mesh: Mesh
    constraints: DirichletConstraints
    load: DeadLoad
    sparsity: object


def problem() -> Problem:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    constraints = DirichletConstraints.fixed_nodes(np.array([0, 1, 2]))
    return Problem(
        Mesh(nodes),
        constraints,
        DeadLoad(np.arange(9, dtype=float)),
        object(),
    )


def test_axis_angle_path_preserves_fixed_constraints_and_sparsity() -> None:
    original = problem()
    marker = original.sparsity
    path = RigidBodyBoundaryPath.from_problem(
        original,
        np.array([1, 2]),
        pivot=np.zeros(3),
        axis=np.array([0.0, 0.0, 4.0]),
        end_angle=np.pi / 2.0,
        end_translation=np.array([1.0, -0.5, 0.25]),
        proportional_load=True,
        values=(LinearPathValue("phase", 0.0, 2.0),),
    )

    state = path.evaluate(original, 0.5)
    angle = np.pi / 4.0
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.5, -0.25, 0.125])
    reference = original.mesh.reference_nodes[[1, 2]]
    expected = reference @ rotation.T + translation - reference

    np.testing.assert_allclose(state.problem.constraints.values[:3], 0.0)
    np.testing.assert_allclose(path.controlled_displacements(0.5), expected)
    controlled_values = state.problem.constraints.values[3:].reshape((-1, 3))
    np.testing.assert_allclose(controlled_values, expected)
    np.testing.assert_allclose(state.effective_force, 0.5 * original.load.force)
    assert state.solver_load_factor == pytest.approx(1.0)
    assert state.value("rotation_angle") == pytest.approx(angle)
    assert state.value("translation_x") == pytest.approx(0.5)
    assert state.value("translation_y") == pytest.approx(-0.25)
    assert state.value("translation_z") == pytest.approx(0.125)
    assert state.value("phase") == pytest.approx(1.0)
    assert state.problem.sparsity is marker


def test_quarter_turn_displacements_are_exact() -> None:
    original = problem()
    path = RigidBodyBoundaryPath.from_problem(
        original,
        np.array([1, 2]),
        pivot=np.zeros(3),
        end_angle=np.pi / 2.0,
    )
    np.testing.assert_allclose(
        path.controlled_displacements(1.0),
        np.array(
            [
                [-1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
            ]
        ),
        atol=2.0e-15,
    )
    initial_state = path.evaluate(original, 0.0)
    np.testing.assert_allclose(initial_state.prescribed_values, 0.0)


def test_factory_removes_controlled_dofs_from_fixed_constraint_set() -> None:
    original = problem()
    path = RigidBodyBoundaryPath.from_problem(
        original,
        np.array([2]),
        end_angle=0.2,
    )
    np.testing.assert_array_equal(path.fixed_constraints.dofs, np.arange(6))
    np.testing.assert_array_equal(path.controlled_nodes, [2])
    np.testing.assert_allclose(path.pivot, original.mesh.reference_nodes[2])


def test_factory_requires_every_controlled_component_to_be_constrained() -> None:
    original = problem()
    partial = Problem(
        original.mesh,
        DirichletConstraints(np.arange(8), np.zeros(8)),
        original.load,
        original.sparsity,
    )
    with pytest.raises(ValueError, match="every controlled-node DOF"):
        RigidBodyBoundaryPath.from_problem(
            partial,
            np.array([2]),
            end_angle=0.2,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"axis": np.zeros(3)}, "axis must have nonzero length"),
        ({"controlled_nodes": np.array([1, 1])}, "unique and nonnegative"),
        (
            {"values": (LinearPathValue("rotation_angle", 0.0, 1.0),)},
            "reserved names",
        ),
    ],
)
def test_invalid_rigid_path_configuration_is_rejected(kwargs, message) -> None:
    original = problem()
    arguments = {
        "controlled_nodes": np.array([1, 2]),
        "end_angle": 0.5,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        RigidBodyBoundaryPath.from_problem(original, **arguments)


def test_direct_constructor_rejects_fixed_controlled_overlap() -> None:
    original = problem()
    with pytest.raises(ValueError, match="must not overlap"):
        RigidBodyBoundaryPath(
            DirichletConstraints.fixed_nodes(np.array([1])),
            np.array([1]),
            original.mesh.reference_nodes[[1]],
            np.zeros(3),
            np.array([0.0, 0.0, 1.0]),
            0.0,
            0.5,
            np.zeros(3),
            np.zeros(3),
            original.load,
            original.load,
        )
