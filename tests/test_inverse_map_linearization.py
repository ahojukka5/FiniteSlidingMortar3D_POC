from __future__ import annotations

import numpy as np
import pytest

from contact3d import (
    InverseMapTopologyError,
    facet_projection_plane,
    infer_facet_kind,
    inverse_map_2d,
    inverse_map_2d_linearized,
    linearize_facet_quadrature,
    project_to_plane,
    replay_clipping_topology,
    shape_values,
    triangulate_convex_polygon,
)
from contact3d.quadrature import triangle_rule


def _identity_input_jacobians(
    polygon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    polygon_dofs = 2 * len(polygon)
    total_dofs = polygon_dofs + 2
    polygon_jacobian = np.zeros((*polygon.shape, total_dofs))
    point_jacobian = np.zeros((2, total_dofs))
    for node in range(len(polygon)):
        for component in range(2):
            polygon_jacobian[node, component, 2 * node + component] = 1.0
    point_jacobian[0, -2] = 1.0
    point_jacobian[1, -1] = 1.0
    return polygon_jacobian, point_jacobian


@pytest.mark.parametrize(
    ("kind", "polygon", "parent"),
    [
        (
            "tri3",
            np.array([[0.0, 0.0], [1.3, 0.1], [-0.2, 1.1]]),
            np.array([0.27, 0.31]),
        ),
        (
            "quad4",
            np.array([[-1.0, -0.8], [1.2, -1.0], [1.0, 1.1], [-0.9, 0.9]]),
            np.array([0.31, -0.27]),
        ),
    ],
)
def test_inverse_map_jacobian_matches_centered_difference(
    kind: str,
    polygon: np.ndarray,
    parent: np.ndarray,
) -> None:
    point = shape_values(kind, parent) @ polygon
    polygon_jacobian, point_jacobian = _identity_input_jacobians(polygon)
    result = inverse_map_2d_linearized(
        polygon,
        kind,
        point,
        polygon_jacobian,
        point_jacobian,
    )

    step = 2.0e-7
    numerical_parent = np.zeros_like(result.parent_jacobian)
    numerical_shape = np.zeros_like(result.shape_jacobian)
    for column in range(result.parent_jacobian.shape[1]):
        plus_polygon = polygon + step * polygon_jacobian[:, :, column]
        minus_polygon = polygon - step * polygon_jacobian[:, :, column]
        plus_point = point + step * point_jacobian[:, column]
        minus_point = point - step * point_jacobian[:, column]
        plus_parent = inverse_map_2d(plus_polygon, kind, plus_point)
        minus_parent = inverse_map_2d(minus_polygon, kind, minus_point)
        numerical_parent[:, column] = (plus_parent - minus_parent) / (2.0 * step)
        numerical_shape[:, column] = (
            shape_values(kind, plus_parent) - shape_values(kind, minus_parent)
        ) / (2.0 * step)

    assert np.max(np.abs(result.parent_jacobian - numerical_parent)) < 2.0e-8
    assert np.max(np.abs(result.shape_jacobian - numerical_shape)) < 2.0e-8
    assert result.mapping_residual < 1.0e-12
    assert result.mapping_jacobian_residual < 2.0e-12
    assert result.partition_error < 1.0e-14
    assert result.partition_jacobian_error < 2.0e-14


def test_inverse_map_rejects_singular_projected_facet() -> None:
    polygon = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    polygon_jacobian, point_jacobian = _identity_input_jacobians(polygon)

    with pytest.raises(InverseMapTopologyError, match="singular"):
        inverse_map_2d_linearized(
            polygon,
            "tri3",
            np.array([0.0, 0.0]),
            polygon_jacobian,
            point_jacobian,
        )


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


def _frozen_quadrature_values(
    slave: np.ndarray,
    master: np.ndarray,
    topology: object,
    quadrature_points: int,
) -> tuple[np.ndarray, ...]:
    slave_kind = infer_facet_kind(slave)
    master_kind = infer_facet_kind(master)
    plane = facet_projection_plane(slave, slave_kind)
    slave_polygon = project_to_plane(slave, plane)
    master_polygon = project_to_plane(master, plane)
    intersection = replay_clipping_topology(
        slave_polygon,
        master_polygon,
        topology,
    )
    pallets = triangulate_convex_polygon(intersection)
    barycentric_points, rule_weights = triangle_rule(quadrature_points)

    points: list[np.ndarray] = []
    slave_parents: list[np.ndarray] = []
    master_parents: list[np.ndarray] = []
    slave_shapes: list[np.ndarray] = []
    master_shapes: list[np.ndarray] = []
    integration_weights: list[float] = []
    for pallet in pallets:
        for barycentric, rule_weight in zip(
            barycentric_points,
            rule_weights,
            strict=True,
        ):
            point = barycentric @ pallet.vertices
            slave_parent = inverse_map_2d(slave_polygon, slave_kind, point)
            master_parent = inverse_map_2d(master_polygon, master_kind, point)
            points.append(point)
            slave_parents.append(slave_parent)
            master_parents.append(master_parent)
            slave_shapes.append(shape_values(slave_kind, slave_parent))
            master_shapes.append(shape_values(master_kind, master_parent))
            integration_weights.append(pallet.area * float(rule_weight))

    return (
        np.asarray(points),
        np.asarray(slave_parents),
        np.asarray(master_parents),
        np.asarray(slave_shapes),
        np.asarray(master_shapes),
        np.asarray(integration_weights),
    )


def test_full_facet_quadrature_chain_matches_frozen_topology_difference() -> None:
    slave, master = _warped_pair()
    quadrature_points = 3
    result = linearize_facet_quadrature(
        slave,
        master,
        quadrature_points=quadrature_points,
    )
    assert result.points

    base_values = _frozen_quadrature_values(
        slave,
        master,
        result.geometry.intersection.topology,
        quadrature_points,
    )
    assert len(result.points) == len(base_values[0])

    analytical = (
        np.stack([point.point_jacobian for point in result.points]),
        np.stack([point.slave.parent_jacobian for point in result.points]),
        np.stack([point.master.parent_jacobian for point in result.points]),
        np.stack([point.slave.shape_jacobian for point in result.points]),
        np.stack([point.master.shape_jacobian for point in result.points]),
        np.stack([point.integration_weight_jacobian for point in result.points]),
    )
    numerical = tuple(np.zeros_like(value) for value in analytical)
    coordinates = np.vstack([slave, master])
    total_dofs = coordinates.size

    for column in range(total_dofs):
        node, component = divmod(column, 3)
        step = 2.0e-7 * max(1.0, abs(float(coordinates[node, component])))
        plus_slave = slave.copy()
        minus_slave = slave.copy()
        plus_master = master.copy()
        minus_master = master.copy()
        if node < len(slave):
            plus_slave[node, component] += step
            minus_slave[node, component] -= step
        else:
            master_node = node - len(slave)
            plus_master[master_node, component] += step
            minus_master[master_node, component] -= step

        plus = _frozen_quadrature_values(
            plus_slave,
            plus_master,
            result.geometry.intersection.topology,
            quadrature_points,
        )
        minus = _frozen_quadrature_values(
            minus_slave,
            minus_master,
            result.geometry.intersection.topology,
            quadrature_points,
        )
        for target, plus_value, minus_value in zip(
            numerical,
            plus,
            minus,
            strict=True,
        ):
            target[..., column] = (plus_value - minus_value) / (2.0 * step)

    tolerances = (2.0e-7, 3.0e-7, 3.0e-7, 3.0e-7, 3.0e-7, 2.0e-7)
    for analytical_value, numerical_value, tolerance in zip(
        analytical,
        numerical,
        tolerances,
        strict=True,
    ):
        assert np.max(np.abs(analytical_value - numerical_value)) < tolerance

    assert result.weight_consistency_error < 2.0e-14
    assert result.weight_jacobian_consistency_error < 2.0e-12
    assert max(
        point.slave.partition_jacobian_error for point in result.points
    ) < 2.0e-13
    assert max(
        point.master.partition_jacobian_error for point in result.points
    ) < 2.0e-13


def test_facet_quadrature_has_common_translation_nullspace() -> None:
    slave, master = _warped_pair()
    result = linearize_facet_quadrature(slave, master, quadrature_points=3)
    total_nodes = len(slave) + len(master)

    for component in range(3):
        direction = np.zeros(3 * total_nodes)
        direction[component::3] = 1.0
        for point in result.points:
            assert np.linalg.norm(point.point_jacobian @ direction) < 2.0e-12
            assert np.linalg.norm(point.slave.parent_jacobian @ direction) < 2.0e-12
            assert np.linalg.norm(point.master.parent_jacobian @ direction) < 2.0e-12
            assert np.linalg.norm(point.slave.shape_jacobian @ direction) < 2.0e-12
            assert np.linalg.norm(point.master.shape_jacobian @ direction) < 2.0e-12
            assert abs(
                float(point.integration_weight_jacobian @ direction)
            ) < 2.0e-12
