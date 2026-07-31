from __future__ import annotations

import numpy as np
import pytest

from contact3d import ContactSurface, averaged_nodal_normals, discover_facet_pairs
from contact3d.broad_phase import FacetAABBTree, facet_aabbs
from contact3d.surface import (
    discover_facet_pairs_brute_force,
    discover_facet_pairs_with_diagnostics,
)


def _unit_quad(z: float = 0.0) -> np.ndarray:
    return np.array(
        [[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]],
        dtype=float,
    )


def _quad_grid(
    nx: int,
    ny: int,
    *,
    z: float = 0.0,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    nodes = np.array(
        [[i / nx, j / ny, z] for j in range(ny + 1) for i in range(nx + 1)],
        dtype=float,
    )
    facets = []
    for j in range(ny):
        for i in range(nx):
            first = j * (nx + 1) + i
            facets.append(
                np.array(
                    [first, first + 1, first + nx + 2, first + nx + 1],
                    dtype=np.int64,
                )
            )
    return nodes, tuple(facets)


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


def test_bvh_matches_quadratic_oracle_on_randomized_deformation() -> None:
    slave_reference, slave_facets = _quad_grid(7, 5)
    master_reference, master_facets = _quad_grid(6, 8, z=-0.08)
    slave = ContactSurface(slave_reference, slave_facets)
    master = ContactSurface(master_reference, master_facets)
    generator = np.random.default_rng(119)
    slave_current = slave_reference + 0.015 * generator.standard_normal(slave_reference.shape)
    master_current = master_reference + 0.015 * generator.standard_normal(master_reference.shape)

    expected = discover_facet_pairs_brute_force(
        slave,
        master,
        slave_current,
        master_current,
        search_distance=0.12,
    )
    result = discover_facet_pairs_with_diagnostics(
        slave,
        master,
        slave_current,
        master_current,
        search_distance=0.12,
    )

    assert result.pairs == expected
    assert result.pairs == tuple(sorted(result.pairs))
    assert result.diagnostics.accepted_pairs == len(expected)
    assert result.diagnostics.facet_tests < result.diagnostics.brute_force_tests


def test_bvh_pair_order_is_independent_of_leaf_size() -> None:
    slave_nodes, slave_facets = _quad_grid(8, 6)
    master_nodes, master_facets = _quad_grid(7, 9, z=-0.05)
    slave_minimums, slave_maximums = facet_aabbs(slave_nodes, slave_facets)
    expected = None

    for leaf_size in (1, 2, 5, 16):
        tree = FacetAABBTree.build(master_nodes, master_facets, leaf_size=leaf_size)
        result = tree.refit(master_nodes).query(
            slave_minimums,
            slave_maximums,
            search_distance=0.08,
        )
        if expected is None:
            expected = result.pairs
        assert result.pairs == expected
        assert result.pairs == tuple(sorted(result.pairs))


def test_refit_reuses_reference_topology_and_updates_cached_bounds() -> None:
    nodes, facets = _quad_grid(8, 8)
    tree = FacetAABBTree.build(nodes, facets, leaf_size=4)
    current = nodes.copy()
    current[:, 2] = 0.1 * np.sin(np.pi * current[:, 0])

    refitted = tree.refit(current)

    assert refitted.topology is tree
    assert refitted.facet_minimums.shape == (64, 3)
    assert refitted.node_minimums.shape == (len(tree.nodes), 3)
    assert np.max(refitted.node_maximums[tree.root]) == pytest.approx(1.0)
    assert refitted.node_maximums[tree.root, 2] == pytest.approx(0.1)


def test_surface_rejects_duplicate_facet_nodes() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ContactSurface(_unit_quad(), (np.array([0, 1, 1, 3]),))
