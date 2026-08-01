"""Continuous staged composition of rigid-body boundary paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coupled import CoupledEquilibriumProblem
from .load_path import CoupledPathState
from .model import FloatArray, IntArray
from .rigid_path import RigidBodyBoundaryPath


def _vector3(value: FloatArray, *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite three-vector")
    return result.copy()


@dataclass(frozen=True, slots=True)
class RigidBodyMotionSegment:
    """Map one rigid-body path onto an absolute continuation interval."""

    name: str
    start_parameter: float
    end_parameter: float
    path: RigidBodyBoundaryPath

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        start = float(self.start_parameter)
        end = float(self.end_parameter)
        if not name:
            raise ValueError("motion-segment name must be nonempty")
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError("motion-segment bounds must be finite")
        if start < 0.0 or end <= start:
            raise ValueError("motion-segment bounds must satisfy 0 <= start < end")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start_parameter", start)
        object.__setattr__(self, "end_parameter", end)

    def local_parameter(self, parameter: float) -> float:
        """Map an absolute path parameter onto this segment's unit interval."""

        value = float(parameter)
        return (value - self.start_parameter) / (
            self.end_parameter - self.start_parameter
        )


@dataclass(frozen=True, slots=True)
class StagedRigidBodyBoundaryPath:
    """Compose continuous rigid-body motions over contiguous intervals."""

    segments: tuple[RigidBodyMotionSegment, ...]
    continuity_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        tolerance = float(self.continuity_tolerance)
        if not segments:
            raise ValueError("staged rigid-body path must contain at least one segment")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("continuity_tolerance must be finite and positive")
        if abs(segments[0].start_parameter) > tolerance:
            raise ValueError("the first motion segment must start at parameter zero")
        names = [segment.name for segment in segments]
        if len(set(names)) != len(names):
            raise ValueError("motion-segment names must be unique")

        reserved = {"phase_index", "phase_parameter", "phase_start", "phase_end"}
        first = segments[0].path
        for index, segment in enumerate(segments):
            path = segment.path
            if reserved.intersection(value.name for value in path.values):
                raise ValueError("segment path values must not use reserved phase names")
            if not np.array_equal(path.controlled_nodes, first.controlled_nodes):
                raise ValueError("all motion segments must control the same nodes")
            if not np.allclose(
                path.reference_positions,
                first.reference_positions,
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("all motion segments must share reference positions")
            if not np.array_equal(
                path.fixed_constraints.dofs,
                first.fixed_constraints.dofs,
            ) or not np.allclose(
                path.fixed_constraints.values,
                first.fixed_constraints.values,
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("all motion segments must share fixed constraints")
            if index == 0:
                continue
            previous = segments[index - 1]
            if not np.isclose(
                previous.end_parameter,
                segment.start_parameter,
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("motion-segment intervals must be contiguous")
            if not np.allclose(
                previous.path.controlled_displacements(1.0),
                path.controlled_displacements(0.0),
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("motion-segment prescribed states must be continuous")
            if not np.allclose(
                previous.path.end_load.force,
                path.start_load.force,
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError("motion-segment loads must be continuous")

        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "continuity_tolerance", tolerance)

    @property
    def end_parameter(self) -> float:
        """Return the final absolute continuation parameter."""

        return self.segments[-1].end_parameter

    def phase_name(self, parameter: float) -> str:
        """Return the selected motion-segment name."""

        _, segment, _ = self._selection(parameter)
        return segment.name

    def _selection(
        self,
        parameter: float,
    ) -> tuple[int, RigidBodyMotionSegment, float]:
        value = float(parameter)
        tolerance = self.continuity_tolerance
        if not np.isfinite(value):
            raise ValueError("path parameter must be finite")
        if value < -tolerance or value > self.end_parameter + tolerance:
            raise ValueError("path parameter lies outside the staged path")
        value = min(max(value, 0.0), self.end_parameter)
        for index, segment in enumerate(self.segments):
            last = index + 1 == len(self.segments)
            if value < segment.end_parameter - tolerance or last:
                local = segment.local_parameter(value)
                return index, segment, min(max(local, 0.0), 1.0)
            if np.isclose(
                value,
                segment.end_parameter,
                rtol=0.0,
                atol=tolerance,
            ):
                following = self.segments[index + 1]
                return index + 1, following, 0.0
        raise RuntimeError("staged path selection failed")

    def evaluate(
        self,
        problem: CoupledEquilibriumProblem,
        parameter: float,
    ) -> CoupledPathState:
        """Evaluate a continuous staged path at one absolute parameter."""

        index, segment, local = self._selection(parameter)
        state = segment.path.evaluate(problem, local)
        values = (
            ("phase_index", float(index)),
            ("phase_parameter", float(local)),
            ("phase_start", segment.start_parameter),
            ("phase_end", segment.end_parameter),
            *state.values,
        )
        return CoupledPathState(
            float(parameter),
            state.problem,
            state.solver_load_factor,
            state.prescribed_dofs,
            state.prescribed_values,
            state.effective_force,
            values,
        )

    @classmethod
    def compression_then_rotation(
        cls,
        problem: CoupledEquilibriumProblem,
        controlled_nodes: IntArray,
        *,
        compression: FloatArray,
        end_angle: float,
        tangential_translation: FloatArray | None = None,
        compression_end: float = 0.25,
        pivot: FloatArray | None = None,
        axis: FloatArray | None = None,
        proportional_load: bool = False,
        continuity_tolerance: float = 1.0e-12,
    ) -> StagedRigidBodyBoundaryPath:
        """Build a two-stage compression then rotation/translation path."""

        split = float(compression_end)
        if not np.isfinite(split) or not 0.0 < split < 1.0:
            raise ValueError("compression_end must lie strictly between zero and one")
        compression_vector = _vector3(compression, name="compression")
        tangential = (
            np.zeros(3)
            if tangential_translation is None
            else _vector3(tangential_translation, name="tangential_translation")
        )
        compression_path = RigidBodyBoundaryPath.from_problem(
            problem,
            controlled_nodes,
            pivot=pivot,
            axis=axis,
            end_angle=0.0,
            end_translation=compression_vector,
            proportional_load=proportional_load,
        )
        rotation_path = RigidBodyBoundaryPath(
            compression_path.fixed_constraints,
            compression_path.controlled_nodes,
            compression_path.reference_positions,
            compression_path.pivot,
            compression_path.axis,
            0.0,
            end_angle,
            compression_vector,
            compression_vector + tangential,
            compression_path.end_load,
            problem.load,
        )
        return cls(
            (
                RigidBodyMotionSegment(
                    "compression",
                    0.0,
                    split,
                    compression_path,
                ),
                RigidBodyMotionSegment(
                    "rotation",
                    split,
                    1.0,
                    rotation_path,
                ),
            ),
            continuity_tolerance,
        )
