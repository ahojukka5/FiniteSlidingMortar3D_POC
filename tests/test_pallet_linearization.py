from __future__ import annotations

import numpy as np
import pytest

from contact3d import (
    PalletTopologyError,
    facet_projection_plane,
    infer_facet_kind,
    linearize_centroid_fan,
    linearize_facet_pallets,
    polygon_signed_area_linearized,
    project_to_plane,
    replay_clipping_topology,
    triangulate_convex_polygon,
)


def _coordinate_jacobian(vertex_count: int) -> np.ndarray:
    jacobian = np.zeros((vertex_count, 2, 2 * vertex_count))
    for vertex in range(vertex_count):
        jacobian[vertex, 0, 2 * vertex] = 1.0
        jacobian[vertex, 1, 2 * vertex + 1] = 1.0
    return jacobian


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


def _replayed_pallets(
    slave: np.ndarray,
    master: np.ndarray,
    topology: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plane = facet_projection_plane(slave, infer_facet_kind(slave))
    slave_polygon = project_to_plane(slave, plane)
    master_polygon = project_to_plane(master, plane)
    polygon = replay_clipping_topology(slave_polygon, master_polygon, topology)
    center = np.mean(polygon, axis=0)
    pallets = triangulate_convex_polygon(polygon)
    vertices = np.stack([pallet.vertices for pallet in pallets])
    areas = np.asarray([pallet.area for pallet in pallets])
    return center, vertices, areas


def test_signed_area_jacobian_matches_centered_difference() -> None:
    polygon = np.array(
        [[-0.2, 0.1], [1.1, -0.1], [1.3, 0.8], [0.4, 1.4], [-0.3, 0.9]]
    )
    jacobian = _coordinate_jacobian(len(polygon))
    analytical = polygon_signed_area_linearized(polygon, jacobian)
    numerical = np.zeros(2 * len(polygon))
    step = 1.0e-7
    for column in range(len(numerical)):
        plus = polygon.copy().reshape(-1)
        minus = polygon.copy().reshape(-1)
        plus[column] += step
        minus[column] -= step
        plus = plus.reshape((-1, 2))
        minus = minus.reshape((-1, 2))
        plus_area = 0.5 * np.sum(
            plus[:, 0] * np.roll(plus[:, 1], -1)
            - plus[:, 1] * np.roll(plus[:, 0], -1)
        )
        minus_area = 0.5 * np.sum(
            minus[:, 0] * np.roll(minus[:, 1], -1)
            - minus[:, 1] * np.roll(minus[:, 0], -1)
        )
        numerical[column] = (plus_area - minus_area) / (2.0 * step)

    assert analytical.jacobian == pytest.approx(numerical, abs=2.0e-9)


def test_centroid_fan_area_decomposition_is_exact_at_derivative_level() -> None:
    polygon = np.array(
        [[0.0, 0.0], [1.2, -0.1], [1.4, 0.7], [0.7, 1.3], [-0.2, 0.8]]
    )
    fan = linearize_centroid_fan(polygon, _coordinate_jacobian(len(polygon)))

    assert fan.area_consistency_error <= 2.0e-15
    assert fan.area_jacobian_consistency_error <= 2.0e-15


def test_facet_pallet_jacobians_match_frozen_topology_difference() -> None:
    slave, master = _warped_pair()
    result = linearize_facet_pallets(slave, master)
    analytical_vertices = np.stack(
        [pallet.vertex_jacobian for pallet in result.fan.pallets]
    )
    analytical_areas = np.stack(
        [pallet.area_jacobian for pallet in result.fan.pallets]
    )
    numerical_center = np.zeros_like(result.fan.center_jacobian)
    numerical_vertices = np.zeros_like(analytical_vertices)
    numerical_areas = np.zeros_like(analytical_areas)
    total_dofs = 3 * (len(slave) + len(master))
    step = 2.0e-7

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
        plus = _replayed_pallets(
            plus_slave,
            plus_master,
            result.intersection.topology,
        )
        minus = _replayed_pallets(
            minus_slave,
            minus_master,
            result.intersection.topology,
        )
        numerical_center[:, column] = (plus[0] - minus[0]) / (2.0 * step)
        numerical_vertices[:, :, :, column] = (
            plus[1] - minus[1]
        ) / (2.0 * step)
        numerical_areas[:, column] = (plus[2] - minus[2]) / (2.0 * step)

    assert np.max(np.abs(result.fan.center_jacobian - numerical_center)) < 1.0e-8
    assert np.max(np.abs(analytical_vertices - numerical_vertices)) < 1.0e-7
    assert np.max(np.abs(analytical_areas - numerical_areas)) < 1.0e-7


def test_facet_pallets_have_common_translation_nullspace() -> None:
    slave, master = _warped_pair()
    result = linearize_facet_pallets(slave, master)
    total_nodes = len(slave) + len(master)

    for component in range(3):
        direction = np.zeros(3 * total_nodes)
        direction[component::3] = 1.0
        assert np.linalg.norm(result.fan.center_jacobian @ direction) < 2.0e-12
        assert abs(float(result.fan.total_area_jacobian @ direction)) < 2.0e-12
        for pallet in result.fan.pallets:
            derivative_vertices = np.tensordot(
                pallet.vertex_jacobian,
                direction,
                axes=(2, 0),
            )
            assert np.linalg.norm(derivative_vertices) < 2.0e-12
            assert abs(float(pallet.area_jacobian @ direction)) < 2.0e-12


def test_empty_branch_and_degenerate_events() -> None:
    empty = linearize_centroid_fan(
        np.empty((0, 2)),
        np.empty((0, 2, 6)),
    )
    assert empty.pallets == ()
    assert empty.total_area == 0.0

    with pytest.raises(PalletTopologyError, match="fewer than three"):
        linearize_centroid_fan(
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.zeros((2, 2, 1)),
        )
    with pytest.raises(PalletTopologyError, match="inverted or has degenerate"):
        linearize_centroid_fan(
            np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            np.zeros((3, 2, 1)),
        )
