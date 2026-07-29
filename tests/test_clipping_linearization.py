from __future__ import annotations

import numpy as np
import pytest

from contact3d import (
    ClippingTopologyError,
    clip_convex_polygon_linearized,
    facet_projection_plane,
    infer_facet_kind,
    linearize_facet_intersection,
    project_to_plane,
    replay_clipping_topology,
    trace_clipping_topology,
)


def _identity_jacobians(
    subject: np.ndarray,
    clipper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    subject_dofs = 2 * len(subject)
    total_dofs = subject_dofs + 2 * len(clipper)
    subject_jacobian = np.zeros((*subject.shape, total_dofs))
    clipper_jacobian = np.zeros((*clipper.shape, total_dofs))
    for node in range(len(subject)):
        for component in range(2):
            subject_jacobian[node, component, 2 * node + component] = 1.0
    for node in range(len(clipper)):
        for component in range(2):
            clipper_jacobian[
                node,
                component,
                subject_dofs + 2 * node + component,
            ] = 1.0
    return subject_jacobian, clipper_jacobian


def test_clipping_vertex_jacobian_matches_frozen_topology_difference() -> None:
    subject = np.array(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
    )
    clipper = np.array(
        [[0.5, -0.3], [2.3, 0.2], [1.7, 1.8], [0.2, 1.5]]
    )
    subject_jacobian, clipper_jacobian = _identity_jacobians(
        subject,
        clipper,
    )
    result = clip_convex_polygon_linearized(
        subject,
        clipper,
        subject_jacobian,
        clipper_jacobian,
    )

    step = 1.0e-7
    numerical = np.zeros_like(result.jacobian)
    for column in range(result.jacobian.shape[2]):
        plus_subject = subject.copy()
        minus_subject = subject.copy()
        plus_clipper = clipper.copy()
        minus_clipper = clipper.copy()
        if column < 2 * len(subject):
            node, component = divmod(column, 2)
            plus_subject[node, component] += step
            minus_subject[node, component] -= step
        else:
            node, component = divmod(column - 2 * len(subject), 2)
            plus_clipper[node, component] += step
            minus_clipper[node, component] -= step
        numerical[:, :, column] = (
            replay_clipping_topology(
                plus_subject,
                plus_clipper,
                result.topology,
            )
            - replay_clipping_topology(
                minus_subject,
                minus_clipper,
                result.topology,
            )
        ) / (2.0 * step)

    assert result.polygon.shape == (6, 2)
    assert np.max(np.abs(result.jacobian - numerical)) < 5.0e-8


def _warped_pair() -> tuple[np.ndarray, np.ndarray]:
    slave = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.03, 0.04],
            [1.1, 1.05, -0.02],
            [-0.04, 0.96, 0.03],
        ]
    )
    master = np.array(
        [
            [0.15, -0.12, -0.11],
            [1.35, -0.03, -0.09],
            [1.27, 0.87, -0.12],
            [0.10, 0.91, -0.14],
        ]
    )
    return slave, master


def _projected_polygons(
    slave: np.ndarray,
    master: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    plane = facet_projection_plane(slave, infer_facet_kind(slave))
    return project_to_plane(slave, plane), project_to_plane(master, plane)


def test_facet_intersection_jacobian_matches_frozen_topology_difference() -> None:
    slave, master = _warped_pair()
    result = linearize_facet_intersection(slave, master)
    total_dofs = 3 * (len(slave) + len(master))
    step = 2.0e-7
    numerical = np.zeros_like(result.intersection_jacobian)

    for column in range(total_dofs):
        plus_slave = slave.copy()
        minus_slave = slave.copy()
        plus_master = master.copy()
        minus_master = master.copy()
        if column < 3 * len(slave):
            node, component = divmod(column, 3)
            plus_slave[node, component] += step
            minus_slave[node, component] -= step
        else:
            node, component = divmod(column - 3 * len(slave), 3)
            plus_master[node, component] += step
            minus_master[node, component] -= step
        plus_subject, plus_clipper = _projected_polygons(
            plus_slave,
            plus_master,
        )
        minus_subject, minus_clipper = _projected_polygons(
            minus_slave,
            minus_master,
        )
        numerical[:, :, column] = (
            replay_clipping_topology(
                plus_subject,
                plus_clipper,
                result.topology,
            )
            - replay_clipping_topology(
                minus_subject,
                minus_clipper,
                result.topology,
            )
        ) / (2.0 * step)

    assert np.max(
        np.abs(result.intersection_jacobian - numerical)
    ) < 8.0e-8


def test_facet_intersection_has_common_translation_nullspace() -> None:
    slave, master = _warped_pair()
    result = linearize_facet_intersection(slave, master)
    total_nodes = len(slave) + len(master)
    for component in range(3):
        direction = np.zeros(3 * total_nodes)
        direction[component::3] = 1.0
        derivative = np.tensordot(
            result.intersection_jacobian,
            direction,
            axes=(2, 0),
        )
        assert np.linalg.norm(derivative) < 2.0e-12


def test_trace_rejects_vertex_on_clipping_edge() -> None:
    subject = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    clipper = np.array(
        [[0.5, 0.0], [1.5, 0.0], [1.5, 1.5], [0.5, 1.5]]
    )
    with pytest.raises(ClippingTopologyError, match="topology event band"):
        trace_clipping_topology(subject, clipper)
