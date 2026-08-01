from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from contact3d.adaptive import (
    RigidBodyBoundaryPath,
    RigidBodyMotionSegment,
    StagedRigidBodyBoundaryPath,
)
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
    return Problem(
        Mesh(nodes),
        DirichletConstraints.fixed_nodes(np.array([0, 1, 2])),
        DeadLoad(np.arange(9, dtype=float)),
        object(),
    )


def test_compression_then_rotation_uses_absolute_phase_parameters() -> None:
    original = problem()
    marker = original.sparsity
    path = StagedRigidBodyBoundaryPath.compression_then_rotation(
        original,
        np.array([1, 2]),
        pivot=np.zeros(3),
        compression=np.array([0.0, 0.0, -0.4]),
        tangential_translation=np.array([1.0, 0.0, 0.0]),
        end_angle=np.pi / 2.0,
        compression_end=0.25,
        proportional_load=True,
    )

    compression = path.evaluate(original, 0.125)
    np.testing.assert_allclose(
        compression.effective_force,
        0.5 * original.load.force,
    )
    assert compression.parameter == pytest.approx(0.125)
    assert compression.value("phase_index") == pytest.approx(0.0)
    assert compression.value("phase_parameter") == pytest.approx(0.5)
    assert compression.value("rotation_angle") == pytest.approx(0.0)
    assert compression.value("translation_z") == pytest.approx(-0.2)

    boundary = path.evaluate(original, 0.25)
    assert path.phase_name(0.25) == "rotation"
    assert boundary.parameter == pytest.approx(0.25)
    assert boundary.value("phase_index") == pytest.approx(1.0)
    assert boundary.value("phase_parameter") == pytest.approx(0.0)
    assert boundary.value("translation_z") == pytest.approx(-0.4)
    np.testing.assert_allclose(boundary.effective_force, original.load.force)

    final = path.evaluate(original, 1.0)
    assert final.value("rotation_angle") == pytest.approx(np.pi / 2.0)
    assert final.value("translation_x") == pytest.approx(1.0)
    assert final.value("translation_z") == pytest.approx(-0.4)
    assert final.problem.sparsity is marker


def test_staged_path_is_continuous_at_the_phase_boundary() -> None:
    original = problem()
    path = StagedRigidBodyBoundaryPath.compression_then_rotation(
        original,
        np.array([1, 2]),
        compression=np.array([0.0, 0.0, -0.3]),
        tangential_translation=np.array([0.4, -0.2, 0.0]),
        end_angle=0.8,
        compression_end=0.4,
    )
    left = path.segments[0].path.evaluate(original, 1.0)
    right = path.segments[1].path.evaluate(original, 0.0)
    np.testing.assert_allclose(left.prescribed_values, right.prescribed_values)
    np.testing.assert_allclose(left.effective_force, right.effective_force)


def test_staged_path_rejects_gaps_and_discontinuous_motion() -> None:
    original = problem()
    base = StagedRigidBodyBoundaryPath.compression_then_rotation(
        original,
        np.array([1, 2]),
        compression=np.array([0.0, 0.0, -0.3]),
        end_angle=0.8,
        compression_end=0.4,
    )
    first, second = base.segments
    with pytest.raises(ValueError, match="intervals must be contiguous"):
        StagedRigidBodyBoundaryPath(
            (
                first,
                RigidBodyMotionSegment("rotation", 0.5, 1.0, second.path),
            )
        )

    discontinuous = RigidBodyBoundaryPath(
        second.path.fixed_constraints,
        second.path.controlled_nodes,
        second.path.reference_positions,
        second.path.pivot,
        second.path.axis,
        second.path.start_angle,
        second.path.end_angle,
        np.array([0.0, 0.0, -0.1]),
        second.path.end_translation,
        second.path.start_load,
        second.path.end_load,
    )
    with pytest.raises(ValueError, match="prescribed states must be continuous"):
        StagedRigidBodyBoundaryPath(
            (
                first,
                RigidBodyMotionSegment("rotation", 0.4, 1.0, discontinuous),
            )
        )


def test_staged_path_rejects_reserved_values_and_out_of_range_parameters() -> None:
    original = problem()
    invalid = RigidBodyBoundaryPath.from_problem(
        original,
        np.array([1, 2]),
        end_angle=0.0,
        values=(LinearPathValue("phase_index", 0.0, 1.0),),
    )
    with pytest.raises(ValueError, match="reserved phase names"):
        StagedRigidBodyBoundaryPath(
            (RigidBodyMotionSegment("invalid", 0.0, 1.0, invalid),)
        )

    path = StagedRigidBodyBoundaryPath.compression_then_rotation(
        original,
        np.array([1, 2]),
        compression=np.array([0.0, 0.0, -0.3]),
        end_angle=0.8,
    )
    with pytest.raises(ValueError, match="outside the staged path"):
        path.evaluate(original, -0.1)
    with pytest.raises(ValueError, match="outside the staged path"):
        path.evaluate(original, 1.1)


def test_staged_path_rejects_discontinuous_loads() -> None:
    original = problem()
    base = StagedRigidBodyBoundaryPath.compression_then_rotation(
        original,
        np.array([1, 2]),
        compression=np.array([0.0, 0.0, -0.3]),
        end_angle=0.8,
        compression_end=0.4,
    )
    first, second = base.segments
    discontinuous = RigidBodyBoundaryPath(
        second.path.fixed_constraints,
        second.path.controlled_nodes,
        second.path.reference_positions,
        second.path.pivot,
        second.path.axis,
        second.path.start_angle,
        second.path.end_angle,
        second.path.start_translation,
        second.path.end_translation,
        DeadLoad(np.zeros_like(original.load.force)),
        second.path.end_load,
    )
    with pytest.raises(ValueError, match="loads must be continuous"):
        StagedRigidBodyBoundaryPath(
            (
                first,
                RigidBodyMotionSegment("rotation", 0.4, 1.0, discontinuous),
            )
        )
