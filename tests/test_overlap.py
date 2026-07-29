from __future__ import annotations

import numpy as np
import pytest

from contact3d import integrate_facet_pair


def _unit_quad(z: float = 0.0) -> np.ndarray:
    return np.array(
        [[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]],
        dtype=float,
    )


def test_identical_quad_recovers_consistent_mass_matrix() -> None:
    points = _unit_quad()
    result = integrate_facet_pair(points, points.copy(), quadrature_points=7)
    expected = np.array(
        [[4.0, 2.0, 1.0, 2.0], [2.0, 4.0, 2.0, 1.0], [1.0, 2.0, 4.0, 2.0], [2.0, 1.0, 2.0, 4.0]]
    ) / 36.0

    assert result.overlap.area == pytest.approx(1.0)
    assert result.d == pytest.approx(expected, abs=2.0e-14)
    assert result.m == pytest.approx(expected, abs=2.0e-14)
    assert result.consistency_error <= 2.0e-15


def test_partial_quad_overlap_conserves_linear_momentum_rowwise() -> None:
    slave = _unit_quad()
    master = _unit_quad(z=0.2) + np.array([0.5, 0.0, 0.0])

    result = integrate_facet_pair(slave, master, quadrature_points=7)

    assert result.overlap.area == pytest.approx(0.5)
    assert np.sum(result.d) == pytest.approx(0.5)
    assert np.sum(result.m) == pytest.approx(0.5)
    assert result.consistency_error <= 2.0e-15


def test_triangle_over_quad_uses_same_physical_quadrature_points() -> None:
    slave = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    master = _unit_quad(z=0.25)

    result = integrate_facet_pair(slave, master, quadrature_points=7)

    assert result.overlap.area == pytest.approx(0.5)
    assert result.d.shape == (3, 3)
    assert result.m.shape == (3, 4)
    assert result.consistency_error <= 2.0e-15


def test_touching_only_at_edge_has_zero_integrated_support() -> None:
    slave = _unit_quad()
    master = _unit_quad(z=0.1) + np.array([1.0, 0.0, 0.0])

    result = integrate_facet_pair(slave, master)

    assert result.overlap.area == 0.0
    assert np.count_nonzero(result.d) == 0
    assert np.count_nonzero(result.m) == 0
