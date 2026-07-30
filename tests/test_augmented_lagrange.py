from __future__ import annotations

import numpy as np
import pytest

from contact3d import (
    AugmentedLagrangeState,
    ContactPair,
    ContactSurface,
    augment_multipliers,
    augmented_lagrange_contact_tangent,
    augmented_pressure_projection,
    evaluate_augmented_lagrange,
    kkt_diagnostics,
    numerical_augmented_lagrange_tangent,
)


def _pair() -> ContactPair:
    slave = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.05, 0.03],
            [1.1, 1.05, -0.02],
            [-0.05, 0.95, 0.04],
        ]
    )
    master = np.array(
        [
            [0.18, -0.12, -0.13],
            [1.35, -0.05, -0.08],
            [1.28, 0.86, -0.11],
            [0.12, 0.92, -0.16],
        ]
    )
    facet = (np.array([0, 1, 2, 3]),)
    return ContactPair(
        ContactSurface(slave, facet),
        ContactSurface(master, facet),
        normal_penalty=37.0,
        search_distance=0.5,
    )


def test_projection_accumulates_penetration_and_releases_open_rows() -> None:
    trial, pressure, active = augmented_pressure_projection(
        np.array([2.0, 2.0, 0.0, 7.0]),
        np.array([0.25, -0.10, -0.20, 0.50]),
        penalty=10.0,
        supported_rows=np.array([True, True, True, False]),
    )
    np.testing.assert_allclose(trial, [4.5, 1.0, -2.0, 12.0])
    np.testing.assert_allclose(pressure, [4.5, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(active, [True, True, False, False])


def test_kkt_projection_residual_vanishes_at_complementary_state() -> None:
    result = kkt_diagnostics(
        np.array([3.0, 0.0, 0.0]),
        np.array([0.0, -0.2, 0.0]),
        20.0,
        np.array([True, True, False]),
    )
    assert result.l2_residual == pytest.approx(0.0)
    assert result.converged(
        gap_tolerance=0.0,
        complementarity_tolerance=0.0,
        projection_tolerance=0.0,
    )


def test_zero_state_reproduces_penalty_contact() -> None:
    pair = _pair()
    state = AugmentedLagrangeState.zeros(pair.slave.node_count)
    augmented = evaluate_augmented_lagrange(pair, state)
    expected = pair.normal_penalty * np.maximum(augmented.contact.normal_gaps, 0.0)
    np.testing.assert_allclose(augmented.contact.pressure, expected)


def test_multiplier_update_is_projected_and_clears_unsupported_rows() -> None:
    pair = _pair()
    initial = AugmentedLagrangeState(np.full(pair.slave.node_count, 1.5))
    evaluation = evaluate_augmented_lagrange(pair, initial)
    update = augment_multipliers(pair, evaluation)
    supported = evaluation.contact.weights.row_areas > 1.0e-12
    expected = np.where(
        supported,
        np.maximum(
            initial.multipliers
            + pair.normal_penalty * evaluation.contact.normal_gaps,
            0.0,
        ),
        0.0,
    )
    np.testing.assert_allclose(update.state.multipliers, expected)
    assert update.state.augmentation == 1
    assert np.all(update.state.multipliers >= 0.0)


def test_augmented_tangent_matches_centered_difference() -> None:
    pair = _pair()
    state = AugmentedLagrangeState(np.array([1.2, 0.7, 1.5, 0.9]))
    base = evaluate_augmented_lagrange(pair, state)
    analytical = augmented_lagrange_contact_tangent(
        pair,
        state,
        active_rows=base.contact.active_rows,
    )
    numerical = numerical_augmented_lagrange_tangent(
        pair,
        state,
        relative_step=5.0e-7,
        freeze_facet_pairs=True,
        freeze_active_rows=True,
    )
    relative_error = np.linalg.norm(analytical - numerical) / np.linalg.norm(numerical)
    assert relative_error < 5.0e-5
