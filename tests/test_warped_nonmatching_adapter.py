from __future__ import annotations

import numpy as np

from contact3d import (
    AugmentedLagrangeState,
    ContactPair,
    ContactSurface,
    MortarContactInterface,
)


def warped_interface() -> MortarContactInterface:
    slave_nodes = np.array(
        [
            [0.14, -0.08, -0.012],
            [1.14, 0.18, -0.006],
            [0.84, 1.18, -0.014],
            [-0.16, 0.86, -0.008],
        ]
    )
    master_nodes = np.array(
        [
            [0.00, 0.00, 0.000],
            [1.00, 0.00, 0.004],
            [1.00, 1.00, -0.002],
            [0.00, 1.00, 0.003],
        ]
    )
    slave = ContactSurface(
        slave_nodes,
        (np.array([0, 1, 2, 3], dtype=np.int64),),
        normal_sign=-1.0,
    )
    master = ContactSurface(
        master_nodes,
        (
            np.array([0, 1, 2], dtype=np.int64),
            np.array([0, 2, 3], dtype=np.int64),
        ),
    )
    pair = ContactPair(
        slave,
        master,
        normal_penalty=2400.0,
        search_distance=0.2,
        quadrature_points=7,
    )
    return MortarContactInterface(
        pair,
        np.arange(4, dtype=np.int64),
        np.arange(4, 8, dtype=np.int64),
    )


def test_production_adapter_warped_nonmatching_tangent() -> None:
    interface = warped_interface()
    displacement = np.zeros((8, 3), dtype=float)
    displacement[:4] = np.array(
        [
            [0.006, -0.003, -0.005],
            [0.010, 0.002, -0.004],
            [0.004, 0.007, -0.006],
            [-0.005, 0.003, -0.003],
        ]
    )
    displacement[4:] = np.array(
        [
            [-0.002, 0.001, 0.000],
            [0.001, -0.002, 0.001],
            [0.003, 0.002, -0.001],
            [-0.001, 0.003, 0.000],
        ]
    )
    state = AugmentedLagrangeState.zeros(4)
    base = interface.evaluate(displacement.ravel(), state, tolerance=1.0e-12)
    tangent = interface.tangent(
        displacement.ravel(),
        state,
        base,
        tolerance=1.0e-12,
    )

    assert set(base.signature.facet_pairs) == {(0, 0), (0, 1)}
    assert np.count_nonzero(base.signature.supported_rows) >= 3
    assert np.count_nonzero(base.signature.active_rows) >= 3
    assert tangent.shape == (24, 24)
    assert np.linalg.norm(base.residual.reshape((-1, 3)).sum(axis=0)) < 1.0e-11

    rng = np.random.default_rng(90317)
    direction = rng.normal(size=24)
    direction /= np.linalg.norm(direction)
    step = 2.0e-7
    plus = interface.evaluate(
        displacement.ravel() + step * direction,
        state,
        tolerance=1.0e-12,
    )
    minus = interface.evaluate(
        displacement.ravel() - step * direction,
        state,
        tolerance=1.0e-12,
    )
    assert plus.signature == base.signature == minus.signature
    numerical = (plus.residual - minus.residual) / (2.0 * step)
    analytical = tangent @ direction
    relative_error = np.linalg.norm(analytical - numerical) / np.linalg.norm(numerical)
    assert relative_error < 2.0e-7
