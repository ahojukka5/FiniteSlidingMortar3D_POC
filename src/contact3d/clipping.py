"""Topology-frozen convex clipping and analytical vertex derivatives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .geometry import (
    facet_projection_plane,
    facet_projection_plane_jacobian,
    polygon_signed_area,
    project_to_plane,
    project_to_plane_jacobian,
)
from .model import FloatArray, ProjectionPlane
from .shapes import infer_facet_kind

OperationKind = Literal["retain", "intersection"]


class ClippingTopologyError(ValueError):
    """Raised when the base state lies on a nonsmooth clipping event."""


@dataclass(frozen=True, slots=True)
class ClippingOperation:
    """One output-vertex construction in a Sutherland-Hodgman stage.

    ``first`` is the retained input index for ``retain`` operations and the
    previous input index for ``intersection`` operations. ``second`` is the
    current input index for intersections and ``-1`` otherwise.
    """

    kind: OperationKind
    first: int
    second: int = -1


@dataclass(frozen=True, slots=True)
class ClippingStage:
    """Frozen operations for one oriented clip edge."""

    clip_edge: int
    input_count: int
    inside: tuple[bool, ...]
    operations: tuple[ClippingOperation, ...]


@dataclass(frozen=True, slots=True)
class ClippingTopology:
    """Frozen orientation and operation trace of convex polygon clipping."""

    subject_reversed: bool
    clipper_reversed: bool
    stages: tuple[ClippingStage, ...]


@dataclass(frozen=True, slots=True)
class ClippedPolygonLinearization:
    """Intersection polygon and its coordinate Jacobian.

    ``jacobian`` has axes ``(intersection_vertex, component, input_dof)``.
    """

    polygon: FloatArray
    jacobian: FloatArray
    topology: ClippingTopology


@dataclass(frozen=True, slots=True)
class FacetIntersectionLinearization:
    """Projected facet intersection differentiated with respect to both facets.

    The Jacobian columns are ordered as all slave coordinates followed by all
    master coordinates, each in node-major xyz order.
    """

    plane: ProjectionPlane
    slave_polygon: FloatArray
    master_polygon: FloatArray
    intersection_polygon: FloatArray
    intersection_jacobian: FloatArray
    topology: ClippingTopology


def _cross2(first: FloatArray, second: FloatArray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _cross2_derivative(
    first: FloatArray,
    second: FloatArray,
    derivative_first: FloatArray,
    derivative_second: FloatArray,
) -> FloatArray:
    """Differentiate the scalar 2D cross product for all derivative columns."""

    return (
        second[1] * derivative_first[0]
        - second[0] * derivative_first[1]
        - first[1] * derivative_second[0]
        + first[0] * derivative_second[1]
    )


def _oriented_polygon(
    polygon: FloatArray,
    jacobian: FloatArray | None = None,
    *,
    reversed_orientation: bool | None = None,
) -> tuple[FloatArray, FloatArray | None, bool]:
    values = np.asarray(polygon, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("clipping polygons must have shape (vertex_count, 2)")
    if len(values) < 3:
        raise ValueError("clipping polygons must contain at least three vertices")
    derivative = None if jacobian is None else np.asarray(jacobian, dtype=float)
    if derivative is not None and derivative.shape[:2] != values.shape:
        raise ValueError("polygon Jacobian must begin with the polygon shape")

    reverse = (
        polygon_signed_area(values) < 0.0
        if reversed_orientation is None
        else bool(reversed_orientation)
    )
    if reverse:
        values = values[::-1].copy()
        if derivative is not None:
            derivative = derivative[::-1].copy()
    else:
        values = values.copy()
        if derivative is not None:
            derivative = derivative.copy()
    return values, derivative, reverse


def _line_intersection(
    previous: FloatArray,
    current: FloatArray,
    clip_start: FloatArray,
    clip_end: FloatArray,
    *,
    tolerance: float,
) -> tuple[FloatArray, float]:
    edge = clip_end - clip_start
    previous_distance = _cross2(edge, previous - clip_start)
    current_distance = _cross2(edge, current - clip_start)
    denominator = previous_distance - current_distance
    if abs(denominator) <= tolerance:
        raise ClippingTopologyError(
            "clip intersection is parallel or numerically singular"
        )
    fraction = previous_distance / denominator
    return previous + fraction * (current - previous), fraction


def _line_intersection_linearized(
    previous: FloatArray,
    current: FloatArray,
    clip_start: FloatArray,
    clip_end: FloatArray,
    derivative_previous: FloatArray,
    derivative_current: FloatArray,
    derivative_clip_start: FloatArray,
    derivative_clip_end: FloatArray,
    *,
    tolerance: float,
) -> tuple[FloatArray, FloatArray]:
    edge = clip_end - clip_start
    derivative_edge = derivative_clip_end - derivative_clip_start
    previous_relative = previous - clip_start
    current_relative = current - clip_start
    derivative_previous_relative = derivative_previous - derivative_clip_start
    derivative_current_relative = derivative_current - derivative_clip_start

    previous_distance = _cross2(edge, previous_relative)
    current_distance = _cross2(edge, current_relative)
    derivative_previous_distance = _cross2_derivative(
        edge,
        previous_relative,
        derivative_edge,
        derivative_previous_relative,
    )
    derivative_current_distance = _cross2_derivative(
        edge,
        current_relative,
        derivative_edge,
        derivative_current_relative,
    )

    denominator = previous_distance - current_distance
    if abs(denominator) <= tolerance:
        raise ClippingTopologyError(
            "clip intersection is parallel or numerically singular"
        )
    derivative_denominator = (
        derivative_previous_distance - derivative_current_distance
    )
    fraction = previous_distance / denominator
    derivative_fraction = (
        derivative_previous_distance * denominator
        - previous_distance * derivative_denominator
    ) / denominator**2

    segment = current - previous
    derivative_segment = derivative_current - derivative_previous
    point = previous + fraction * segment
    derivative_point = (
        derivative_previous
        + fraction * derivative_segment
        + segment[:, None] * derivative_fraction[None, :]
    )
    return point, derivative_point


def trace_clipping_topology(
    subject: FloatArray,
    clipper: FloatArray,
    *,
    tolerance: float = 1.0e-12,
    event_tolerance: float | None = None,
) -> tuple[FloatArray, ClippingTopology]:
    """Clip two convex polygons and record a smooth, replayable operation trace.

    Vertices whose signed distance to a clip edge is within ``event_tolerance``
    are rejected: an on-edge classification is a nonsmooth topology event and
    must be handled outside one Newton derivative.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    event = (
        10.0 * tolerance
        if event_tolerance is None
        else float(event_tolerance)
    )
    if event <= 0.0:
        raise ValueError("event_tolerance must be positive")

    output, _, subject_reversed = _oriented_polygon(subject)
    clip, _, clipper_reversed = _oriented_polygon(clipper)
    stages: list[ClippingStage] = []

    for clip_edge, (clip_start, clip_end) in enumerate(
        zip(clip, np.roll(clip, -1, axis=0), strict=True)
    ):
        if len(output) == 0:
            stages.append(ClippingStage(clip_edge, 0, (), ()))
            continue
        edge = clip_end - clip_start
        distances = np.array(
            [_cross2(edge, point - clip_start) for point in output]
        )
        if np.any(np.abs(distances) <= event):
            raise ClippingTopologyError(
                "polygon vertex lies on a clip edge within the topology event band"
            )
        inside = distances > 0.0
        operations: list[ClippingOperation] = []
        for current_index in range(len(output)):
            previous_index = (current_index - 1) % len(output)
            if bool(inside[current_index]) != bool(inside[previous_index]):
                operations.append(
                    ClippingOperation(
                        "intersection",
                        previous_index,
                        current_index,
                    )
                )
            if bool(inside[current_index]):
                operations.append(ClippingOperation("retain", current_index))

        stage = ClippingStage(
            clip_edge=clip_edge,
            input_count=len(output),
            inside=tuple(bool(value) for value in inside),
            operations=tuple(operations),
        )
        stages.append(stage)
        next_output: list[FloatArray] = []
        for operation in operations:
            if operation.kind == "retain":
                next_output.append(output[operation.first].copy())
            else:
                point, _ = _line_intersection(
                    output[operation.first],
                    output[operation.second],
                    clip_start,
                    clip_end,
                    tolerance=tolerance,
                )
                next_output.append(point)
        output = (
            np.asarray(next_output, dtype=float)
            if next_output
            else np.empty((0, 2), dtype=float)
        )

    if len(output) >= 3 and abs(polygon_signed_area(output)) <= tolerance:
        raise ClippingTopologyError("intersection polygon has a degenerate area")
    return output, ClippingTopology(
        subject_reversed,
        clipper_reversed,
        tuple(stages),
    )


def replay_clipping_topology(
    subject: FloatArray,
    clipper: FloatArray,
    topology: ClippingTopology,
    *,
    tolerance: float = 1.0e-14,
) -> FloatArray:
    """Rebuild an intersection without reclassifying inside/outside state."""

    output, _, _ = _oriented_polygon(
        subject,
        reversed_orientation=topology.subject_reversed,
    )
    clip, _, _ = _oriented_polygon(
        clipper,
        reversed_orientation=topology.clipper_reversed,
    )
    if len(topology.stages) != len(clip):
        raise ValueError("clipping topology does not match the clipper edge count")

    for stage in topology.stages:
        if stage.input_count != len(output):
            raise ValueError(
                "clipping topology input count does not match replay state"
            )
        clip_start = clip[stage.clip_edge]
        clip_end = clip[(stage.clip_edge + 1) % len(clip)]
        next_output: list[FloatArray] = []
        for operation in stage.operations:
            if operation.kind == "retain":
                next_output.append(output[operation.first].copy())
            else:
                point, _ = _line_intersection(
                    output[operation.first],
                    output[operation.second],
                    clip_start,
                    clip_end,
                    tolerance=tolerance,
                )
                next_output.append(point)
        output = (
            np.asarray(next_output, dtype=float)
            if next_output
            else np.empty((0, 2), dtype=float)
        )
    return output


def linearize_clipping_topology(
    subject: FloatArray,
    clipper: FloatArray,
    subject_jacobian: FloatArray,
    clipper_jacobian: FloatArray,
    topology: ClippingTopology,
    *,
    tolerance: float = 1.0e-14,
) -> ClippedPolygonLinearization:
    """Propagate coordinate Jacobians through a frozen clipping trace."""

    output, derivative_output, _ = _oriented_polygon(
        subject,
        subject_jacobian,
        reversed_orientation=topology.subject_reversed,
    )
    clip, derivative_clip, _ = _oriented_polygon(
        clipper,
        clipper_jacobian,
        reversed_orientation=topology.clipper_reversed,
    )
    assert derivative_output is not None
    assert derivative_clip is not None
    if derivative_output.shape[2] != derivative_clip.shape[2]:
        raise ValueError(
            "subject and clipper Jacobians must share the same DOF count"
        )
    if len(topology.stages) != len(clip):
        raise ValueError("clipping topology does not match the clipper edge count")

    for stage in topology.stages:
        if stage.input_count != len(output):
            raise ValueError(
                "clipping topology input count does not match linearization state"
            )
        clip_start_index = stage.clip_edge
        clip_end_index = (stage.clip_edge + 1) % len(clip)
        clip_start = clip[clip_start_index]
        clip_end = clip[clip_end_index]
        derivative_clip_start = derivative_clip[clip_start_index]
        derivative_clip_end = derivative_clip[clip_end_index]
        next_output: list[FloatArray] = []
        next_derivative: list[FloatArray] = []
        for operation in stage.operations:
            if operation.kind == "retain":
                next_output.append(output[operation.first].copy())
                next_derivative.append(
                    derivative_output[operation.first].copy()
                )
            else:
                point, derivative = _line_intersection_linearized(
                    output[operation.first],
                    output[operation.second],
                    clip_start,
                    clip_end,
                    derivative_output[operation.first],
                    derivative_output[operation.second],
                    derivative_clip_start,
                    derivative_clip_end,
                    tolerance=tolerance,
                )
                next_output.append(point)
                next_derivative.append(derivative)
        dof_count = derivative_output.shape[2]
        output = (
            np.asarray(next_output, dtype=float)
            if next_output
            else np.empty((0, 2), dtype=float)
        )
        derivative_output = (
            np.asarray(next_derivative, dtype=float)
            if next_derivative
            else np.empty((0, 2, dof_count), dtype=float)
        )

    return ClippedPolygonLinearization(output, derivative_output, topology)


def clip_convex_polygon_linearized(
    subject: FloatArray,
    clipper: FloatArray,
    subject_jacobian: FloatArray,
    clipper_jacobian: FloatArray,
    *,
    tolerance: float = 1.0e-12,
    event_tolerance: float | None = None,
) -> ClippedPolygonLinearization:
    """Trace and analytically linearize one smooth convex clipping branch."""

    _, topology = trace_clipping_topology(
        subject,
        clipper,
        tolerance=tolerance,
        event_tolerance=event_tolerance,
    )
    return linearize_clipping_topology(
        subject,
        clipper,
        subject_jacobian,
        clipper_jacobian,
        topology,
        tolerance=tolerance,
    )


def linearize_facet_intersection(
    slave_points: FloatArray,
    master_points: FloatArray,
    *,
    tolerance: float = 1.0e-12,
    event_tolerance: float | None = None,
) -> FacetIntersectionLinearization:
    """Differentiate the projected facet intersection through clipping.

    The projection frame is defined by the slave facet. The function combines
    plane and direct-coordinate terms for slave vertices, while retaining
    separate slave-plane and master-point contributions for master vertices.
    """

    slave = np.asarray(slave_points, dtype=float)
    master = np.asarray(master_points, dtype=float)
    slave_kind = infer_facet_kind(slave)
    infer_facet_kind(master)
    plane = facet_projection_plane(slave, slave_kind)
    plane_jacobian = facet_projection_plane_jacobian(slave, slave_kind)
    slave_polygon = project_to_plane(slave, plane)
    master_polygon = project_to_plane(master, plane)
    slave_projected_jacobian = project_to_plane_jacobian(
        slave,
        plane,
        plane_jacobian,
    ).combined_shared_coordinates()
    master_projected_jacobian = project_to_plane_jacobian(
        master,
        plane,
        plane_jacobian,
    )

    slave_count = len(slave)
    master_count = len(master)
    total_dofs = 3 * (slave_count + master_count)
    slave_jacobian = np.zeros((slave_count, 2, total_dofs), dtype=float)
    master_jacobian = np.zeros((master_count, 2, total_dofs), dtype=float)
    slave_jacobian[:, :, : 3 * slave_count] = (
        slave_projected_jacobian.reshape(
            (slave_count, 2, 3 * slave_count)
        )
    )
    master_jacobian[:, :, : 3 * slave_count] = (
        master_projected_jacobian.plane.reshape(
            (master_count, 2, 3 * slave_count)
        )
    )
    master_jacobian[:, :, 3 * slave_count :] = (
        master_projected_jacobian.points.reshape(
            (master_count, 2, 3 * master_count)
        )
    )

    clipped = clip_convex_polygon_linearized(
        slave_polygon,
        master_polygon,
        slave_jacobian,
        master_jacobian,
        tolerance=tolerance,
        event_tolerance=event_tolerance,
    )
    return FacetIntersectionLinearization(
        plane=plane,
        slave_polygon=slave_polygon,
        master_polygon=master_polygon,
        intersection_polygon=clipped.polygon,
        intersection_jacobian=clipped.jacobian,
        topology=clipped.topology,
    )
