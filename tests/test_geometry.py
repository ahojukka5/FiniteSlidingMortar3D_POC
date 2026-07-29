from __future__ import annotations

import numpy as np
import pytest

from contact3d import (
    build_facet_overlap,
    clip_convex_polygon,
    facet_projection_plane,
    inverse_map_2d,
    project_to_plane,
    shape_values,
)


def test_projection_plane_flattens_warped_quad_center() -> None:
    points = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.1], [2.0, 1.0, 0.2], [0.0, 1.0, -0.1]]
    )
    plane = facet_projection_plane(points, "quad4")
    projected = project_to_plane(points, plane)

    assert np.linalg.norm(plane.normal) == pytest.approx(1.0)
    assert np.dot(plane.normal, plane.tangent_u) == pytest.approx(0.0, abs=1.0e-14)
    assert np.dot(plane.normal, plane.tangent_v) == pytest.approx(0.0, abs=1.0e-14)
    assert projected.shape == (4, 2)


def test_quad_inverse_map_round_trip() -> None:
    polygon = np.array([[-1.0, -0.8], [1.2, -1.0], [1.0, 1.1], [-0.9, 0.9]])
    parent = np.array([0.31, -0.27])
    point = shape_values("quad4", parent) @ polygon

    recovered = inverse_map_2d(polygon, "quad4", point)

    assert recovered == pytest.approx(parent, abs=1.0e-12)


def test_convex_clipping_returns_half_square() -> None:
    first = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    second = np.array([[0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [0.5, 1.5]])

    overlap = clip_convex_polygon(first, second)

    assert len(overlap) == 4
    assert np.min(overlap[:, 0]) == pytest.approx(0.5)
    assert np.max(overlap[:, 0]) == pytest.approx(1.0)


def test_disjoint_facets_have_no_pallets() -> None:
    slave = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    master = slave + np.array([2.0, 0.0, 0.1])

    overlap = build_facet_overlap(slave, master)

    assert overlap.area == 0.0
    assert overlap.pallets == ()
