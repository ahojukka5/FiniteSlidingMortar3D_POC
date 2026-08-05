from __future__ import annotations

import numpy as np
import pytest

from contact3d.geometry import ContactSurface, discover_facet_pairs_brute_force
from contact3d.mortar import (
    ContactPair,
    assemble_mortar_weights,
    evaluate_contact,
    numerical_contact_tangent,
)


def _unit_quad(z: float = 0.0) -> np.ndarray:
    return np.array(
        [[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]],
        dtype=float,
    )


def _pair(master_z: float, *, penalty: float = 100.0) -> ContactPair:
    slave_nodes = _unit_quad(0.0)
    master_nodes = _unit_quad(master_z)
    slave = ContactSurface(slave_nodes, (np.array([0, 1, 2, 3]),))
    master = ContactSurface(master_nodes, (np.array([0, 1, 2, 3]),))
    return ContactPair(slave, master, penalty, search_distance=0.5)


def test_uniform_penetration_produces_uniform_pressure_and_balanced_force() -> None:
    pair = _pair(-0.1)

    evaluation = evaluate_contact(pair)

    assert evaluation.normal_gaps == pytest.approx(np.full(4, 0.1), abs=2.0e-15)
    assert evaluation.pressure == pytest.approx(np.full(4, 10.0), abs=2.0e-13)
    assert np.sum(evaluation.slave_force, axis=0) == pytest.approx([0.0, 0.0, 10.0])
    assert np.sum(evaluation.master_force, axis=0) == pytest.approx([0.0, 0.0, -10.0])
    assert evaluation.force_balance_norm <= 2.0e-14
    assert evaluation.moment_balance_norm <= 2.0e-14
    assert evaluation.weights.consistency_error <= 2.0e-15


def test_open_interface_has_zero_contact_force() -> None:
    evaluation = evaluate_contact(_pair(0.1))

    assert not np.any(evaluation.active_rows)
    assert np.count_nonzero(evaluation.pressure) == 0
    assert np.count_nonzero(evaluation.residual) == 0


def test_common_rigid_translation_leaves_contact_residual_unchanged() -> None:
    pair = _pair(-0.1)
    translation = np.tile([0.3, -0.2, 0.5], (4, 1))

    reference = evaluate_contact(pair)
    translated = evaluate_contact(pair, translation, translation)

    assert translated.residual == pytest.approx(reference.residual, abs=5.0e-13)
    assert translated.normal_gaps == pytest.approx(reference.normal_gaps, abs=5.0e-15)


def test_global_assembly_integrates_both_halves_of_split_master() -> None:
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
    pair = ContactPair(slave, master, 100.0, search_distance=0.2)

    weights = assemble_mortar_weights(pair, slave_nodes, master_nodes)
    evaluation = evaluate_contact(pair)

    assert weights.facet_pairs == ((0, 0), (0, 1))
    assert weights.total_area == pytest.approx(1.0)
    assert np.sum(weights.d) == pytest.approx(1.0)
    assert np.sum(weights.m) == pytest.approx(1.0)
    assert evaluation.force_balance_norm <= 2.0e-14


def test_bvh_and_oracle_pairs_produce_identical_contact_operators() -> None:
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
    pair = ContactPair(slave, master, 100.0, search_distance=0.2)
    oracle_pairs = discover_facet_pairs_brute_force(
        slave,
        master,
        slave_nodes,
        master_nodes,
        search_distance=pair.search_distance,
    )

    default_weights = assemble_mortar_weights(pair, slave_nodes, master_nodes)
    oracle_weights = assemble_mortar_weights(
        pair,
        slave_nodes,
        master_nodes,
        facet_pairs=oracle_pairs,
    )
    default_evaluation = evaluate_contact(pair)
    oracle_evaluation = evaluate_contact(pair, facet_pairs=oracle_pairs)

    assert default_weights.facet_pairs == oracle_weights.facet_pairs
    np.testing.assert_array_equal(default_weights.overlap_areas, oracle_weights.overlap_areas)
    np.testing.assert_array_equal(default_weights.d, oracle_weights.d)
    np.testing.assert_array_equal(default_weights.m, oracle_weights.m)
    np.testing.assert_array_equal(default_evaluation.residual, oracle_evaluation.residual)


def test_numerical_tangent_has_common_translation_nullspace() -> None:
    pair = _pair(-0.1)

    tangent = numerical_contact_tangent(pair, relative_step=1.0e-7)
    translation = np.tile([0.4, -0.7, 0.2], pair.slave.node_count + pair.master.node_count)

    assert np.linalg.norm(tangent @ translation) <= 2.0e-6
