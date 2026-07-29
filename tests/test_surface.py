from __future__ import annotations

import numpy as np
import pytest

from contact3d import ContactSurface, averaged_nodal_normals, discover_facet_pairs


def _unit_quad(z: float = 0.0) -> np.ndarray:
    return np.array(
        [[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]],
        dtype=float,
    )


def test_flat_quad_has_constant_appendix_a_normal() -> None:
    nodes = _unit_quad()
    surface = ContactSurface(nodes, (np.array([0, 1, 2, 3]),))

    normals = averaged_nodal_normals(surface, nodes)

    assert normals == pytest.approx(np.tile([0.0, 0.0, 1.0], (4, 1)))


def test_appendix_a_normal_rotates_objectively() -> None:
    nodes = _unit_quad()
    surface = ContactSurface(nodes, (np.array([0, 1, 2, 3]),))
    angle = 0.37
    rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )
    current = nodes @ rotation.T

    normals = averaged_nodal_normals(surface, current)

    expected = np.tile(rotation @ np.array([0.0, 0.0, 1.0]), (4, 1))
    assert normals == pytest.approx(expected, abs=2.0e-15)


def test_broad_phase_keeps_both_master_facets_at_shared_edge() -> None:
    slave_nodes = _unit_quad()
    slave = ContactSurface(slave_nodes, (np.array([0, 1, 2, 3]),))
    master_nodes = np.array(
        [
            [0.0, 0.0, -0.1],
            [0.5, 0.0, -0.1],
            [1.0, 0.0, -0.1],
            [0.0, 1.0, -0.1],
            [0.5, 1.0, -0.1],
            [1.0, 1.0, -0.1],
        ]
    )
    master = ContactSurface(
        master_nodes,
        (np.array([0, 1, 4, 3]), np.array([1, 2, 5, 4])),
    )

    pairs = discover_facet_pairs(
        slave,
        master,
        slave_nodes,
        master_nodes,
        search_distance=0.2,
    )

    assert pairs == ((0, 0), (0, 1))


def test_surface_rejects_duplicate_facet_nodes() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ContactSurface(_unit_quad(), (np.array([0, 1, 1, 3]),))
