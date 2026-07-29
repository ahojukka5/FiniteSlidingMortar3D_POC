from __future__ import annotations

import numpy as np
import pytest

from contact3d import (
    facet_projection_plane,
    facet_projection_plane_jacobian,
    project_to_plane,
    project_to_plane_jacobian,
)


def _facet(kind: str) -> np.ndarray:
    if kind == "tri3":
        return np.array(
            [[0.1, -0.2, 0.05], [1.3, 0.1, 0.18], [-0.1, 1.1, -0.12]],
            dtype=float,
        )
    return np.array(
        [
            [0.0, 0.0, 0.02],
            [1.4, -0.1, 0.17],
            [1.2, 1.1, 0.31],
            [-0.2, 0.9, -0.16],
        ],
        dtype=float,
    )


@pytest.mark.parametrize("kind", ["tri3", "quad4"])
def test_projection_plane_jacobian_matches_centered_differences(kind: str) -> None:
    points = _facet(kind)
    analytical = facet_projection_plane_jacobian(points, kind)
    fields = ("origin", "tangent_u", "tangent_v", "normal")
    step = 1.0e-7

    for node in range(len(points)):
        for component in range(3):
            plus = points.copy()
            minus = points.copy()
            plus[node, component] += step
            minus[node, component] -= step
            plus_plane = facet_projection_plane(plus, kind)
            minus_plane = facet_projection_plane(minus, kind)
            for field in fields:
                numerical = (
                    getattr(plus_plane, field) - getattr(minus_plane, field)
                ) / (2.0 * step)
                expected = getattr(analytical, field)[:, node, component]
                assert numerical == pytest.approx(expected, abs=2.0e-9)


@pytest.mark.parametrize("kind", ["tri3", "quad4"])
def test_self_projected_vertex_jacobian_matches_centered_differences(kind: str) -> None:
    points = _facet(kind)
    plane = facet_projection_plane(points, kind)
    plane_jacobian = facet_projection_plane_jacobian(points, kind)
    analytical = project_to_plane_jacobian(
        points,
        plane,
        plane_jacobian,
    ).combined_shared_coordinates()
    step = 1.0e-7

    for node in range(len(points)):
        for component in range(3):
            plus = points.copy()
            minus = points.copy()
            plus[node, component] += step
            minus[node, component] -= step
            numerical = (
                project_to_plane(plus, facet_projection_plane(plus, kind))
                - project_to_plane(minus, facet_projection_plane(minus, kind))
            ) / (2.0 * step)
            assert numerical == pytest.approx(
                analytical[:, :, node, component],
                abs=3.0e-9,
            )


def test_master_projection_jacobian_separates_slave_and_master_coordinates() -> None:
    slave = _facet("quad4")
    master = np.array(
        [
            [0.2, -0.1, -0.20],
            [1.1, 0.0, -0.14],
            [1.3, 0.8, -0.08],
            [0.0, 1.0, -0.21],
        ],
        dtype=float,
    )
    plane = facet_projection_plane(slave, "quad4")
    plane_jacobian = facet_projection_plane_jacobian(slave, "quad4")
    analytical = project_to_plane_jacobian(master, plane, plane_jacobian)
    step = 1.0e-7

    for node in range(len(slave)):
        for component in range(3):
            plus = slave.copy()
            minus = slave.copy()
            plus[node, component] += step
            minus[node, component] -= step
            numerical = (
                project_to_plane(master, facet_projection_plane(plus, "quad4"))
                - project_to_plane(master, facet_projection_plane(minus, "quad4"))
            ) / (2.0 * step)
            assert numerical == pytest.approx(
                analytical.plane[:, :, node, component],
                abs=3.0e-9,
            )

    for node in range(len(master)):
        for component in range(3):
            plus = master.copy()
            minus = master.copy()
            plus[node, component] += step
            minus[node, component] -= step
            numerical = (
                project_to_plane(plus, plane) - project_to_plane(minus, plane)
            ) / (2.0 * step)
            assert numerical == pytest.approx(
                analytical.points[:, :, node, component],
                abs=2.0e-9,
            )


def test_self_projection_jacobian_annihilates_common_translation() -> None:
    points = _facet("quad4")
    plane = facet_projection_plane(points, "quad4")
    jacobian = project_to_plane_jacobian(
        points,
        plane,
        facet_projection_plane_jacobian(points, "quad4"),
    ).combined_shared_coordinates()
    translation = np.tile([0.3, -0.4, 0.8], (len(points), 1))

    projected_increment = np.einsum("ijab,ab->ij", jacobian, translation)

    assert np.linalg.norm(projected_increment) <= 2.0e-15
