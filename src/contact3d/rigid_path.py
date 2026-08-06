"""Rigid-body prescribed-displacement paths for coupled continuation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coupled import CoupledEquilibriumProblem
from .load_path import CoupledPathState, LinearPathValue, with_coupled_boundary_data
from .mechanics import DeadLoad, DirichletConstraints, FloatArray, IntArray


def _vector3(value: FloatArray, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result.copy()


def _controlled_dofs(nodes: IntArray) -> IntArray:
    return np.asarray(
        [3 * int(node) + component for node in nodes for component in range(3)],
        dtype=np.int64,
    )


def _rotation_matrix(axis: FloatArray, angle: float) -> FloatArray:
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    return cosine * np.eye(3) + sine * skew + (1.0 - cosine) * np.outer(axis, axis)


@dataclass(frozen=True, slots=True)
class RigidBodyBoundaryPath:
    """Prescribe an axis-angle rigid motion on selected global mesh nodes.

    The path retains ``fixed_constraints`` exactly and replaces all three DOFs of
    every controlled node with the displacement induced by an interpolated rigid
    transform. Translation is the motion of the rotation pivot, so the current
    controlled-node coordinates are

    ``pivot + translation + rotation @ (reference - pivot)``.
    """

    fixed_constraints: DirichletConstraints
    controlled_nodes: IntArray
    reference_positions: FloatArray
    pivot: FloatArray
    axis: FloatArray
    start_angle: float
    end_angle: float
    start_translation: FloatArray
    end_translation: FloatArray
    start_load: DeadLoad
    end_load: DeadLoad
    values: tuple[LinearPathValue, ...] = ()

    def __post_init__(self) -> None:
        nodes = np.asarray(self.controlled_nodes, dtype=np.int64)
        positions = np.asarray(self.reference_positions, dtype=float)
        if nodes.ndim != 1 or len(nodes) == 0:
            raise ValueError("controlled_nodes must be a nonempty flat vector")
        if np.any(nodes < 0) or len(np.unique(nodes)) != len(nodes):
            raise ValueError("controlled_nodes must be unique and nonnegative")
        if positions.shape != (len(nodes), 3) or not np.all(np.isfinite(positions)):
            raise ValueError(
                "reference_positions must be finite with shape (node_count, 3)"
            )

        pivot = _vector3(self.pivot, name="pivot")
        axis = _vector3(self.axis, name="axis")
        magnitude = float(np.linalg.norm(axis))
        if magnitude <= np.finfo(float).eps:
            raise ValueError("axis must have nonzero length")
        axis /= magnitude
        start_translation = _vector3(
            self.start_translation,
            name="start_translation",
        )
        end_translation = _vector3(
            self.end_translation,
            name="end_translation",
        )
        start_angle = float(self.start_angle)
        end_angle = float(self.end_angle)
        if not np.isfinite(start_angle) or not np.isfinite(end_angle):
            raise ValueError("rotation angles must be finite")
        if self.start_load.force.shape != self.end_load.force.shape:
            raise ValueError("rigid-path load endpoints must have equal shapes")

        dofs = _controlled_dofs(nodes)
        if np.intersect1d(self.fixed_constraints.dofs, dofs).size:
            raise ValueError("fixed constraints must not overlap controlled-node DOFs")
        names = [value.name for value in self.values]
        reserved = {"rotation_angle", "translation_x", "translation_y", "translation_z"}
        if len(set(names)) != len(names):
            raise ValueError("rigid-path value names must be unique")
        if reserved.intersection(names):
            raise ValueError("custom rigid-path values must not use reserved names")

        object.__setattr__(self, "controlled_nodes", nodes.copy())
        object.__setattr__(self, "reference_positions", positions.copy())
        object.__setattr__(self, "pivot", pivot)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "start_angle", start_angle)
        object.__setattr__(self, "end_angle", end_angle)
        object.__setattr__(self, "start_translation", start_translation)
        object.__setattr__(self, "end_translation", end_translation)
        object.__setattr__(self, "values", tuple(self.values))

    @classmethod
    def from_problem(
        cls,
        problem: CoupledEquilibriumProblem,
        controlled_nodes: IntArray,
        *,
        end_angle: float,
        pivot: FloatArray | None = None,
        axis: FloatArray | None = None,
        start_angle: float = 0.0,
        start_translation: FloatArray | None = None,
        end_translation: FloatArray | None = None,
        proportional_load: bool = False,
        values: tuple[LinearPathValue, ...] = (),
    ) -> RigidBodyBoundaryPath:
        """Build a path from a problem whose controlled nodes are fully constrained."""

        nodes = np.asarray(controlled_nodes, dtype=np.int64)
        if nodes.ndim != 1 or len(nodes) == 0:
            raise ValueError("controlled_nodes must be a nonempty flat vector")
        if np.any(nodes < 0) or np.any(nodes >= problem.mesh.node_count):
            raise ValueError("controlled node is outside the problem mesh")
        controlled = _controlled_dofs(nodes)
        constrained = set(int(value) for value in problem.constraints.dofs)
        missing = sorted(set(int(value) for value in controlled) - constrained)
        if missing:
            raise ValueError(
                "every controlled-node DOF must be constrained in the problem"
            )
        fixed_mask = ~np.isin(problem.constraints.dofs, controlled)
        fixed = DirichletConstraints(
            problem.constraints.dofs[fixed_mask],
            problem.constraints.values[fixed_mask],
        )
        positions = np.asarray(problem.mesh.reference_nodes, dtype=float)[nodes]
        center = np.mean(positions, axis=0) if pivot is None else pivot
        rotation_axis = np.array([0.0, 0.0, 1.0]) if axis is None else axis
        initial_translation = (
            np.zeros(3) if start_translation is None else start_translation
        )
        final_translation = np.zeros(3) if end_translation is None else end_translation
        start_load = (
            DeadLoad(np.zeros_like(problem.load.force))
            if proportional_load
            else problem.load
        )
        return cls(
            fixed,
            nodes,
            positions,
            center,
            rotation_axis,
            start_angle,
            end_angle,
            initial_translation,
            final_translation,
            start_load,
            problem.load,
            values,
        )

    def controlled_displacements(self, parameter: float) -> FloatArray:
        """Return the prescribed controlled-node displacement array."""

        value = float(parameter)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("path parameter must be finite and nonnegative")
        angle = self.start_angle + value * (self.end_angle - self.start_angle)
        translation = self.start_translation + value * (
            self.end_translation - self.start_translation
        )
        rotation = _rotation_matrix(self.axis, angle)
        relative = self.reference_positions - self.pivot
        current = self.pivot + translation + relative @ rotation.T
        return current - self.reference_positions

    def evaluate(
        self,
        problem: CoupledEquilibriumProblem,
        parameter: float,
    ) -> CoupledPathState:
        value = float(parameter)
        displacements = self.controlled_displacements(value)
        controlled = _controlled_dofs(self.controlled_nodes)
        constraints = DirichletConstraints(
            np.concatenate([self.fixed_constraints.dofs, controlled]),
            np.concatenate([self.fixed_constraints.values, displacements.ravel()]),
        )
        force = self.start_load.force + value * (
            self.end_load.force - self.start_load.force
        )
        load = DeadLoad(force)
        updated = with_coupled_boundary_data(problem, constraints, load)
        angle = self.start_angle + value * (self.end_angle - self.start_angle)
        translation = self.start_translation + value * (
            self.end_translation - self.start_translation
        )
        named = (
            ("rotation_angle", float(angle)),
            ("translation_x", float(translation[0])),
            ("translation_y", float(translation[1])),
            ("translation_z", float(translation[2])),
        ) + tuple((item.name, item.evaluate(value)) for item in self.values)
        return CoupledPathState(
            value,
            updated,
            1.0,
            constraints.dofs,
            constraints.values,
            load.force,
            named,
        )
