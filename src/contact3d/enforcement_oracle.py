"""Centered-difference oracle for augmented-Lagrange contact."""

from __future__ import annotations

import numpy as np

from .contact import ContactPair
from .enforcement_evaluation import evaluate_augmented_lagrange
from .enforcement_state import AugmentedLagrangeState
from .model import FloatArray


def numerical_augmented_lagrange_tangent(
    pair: ContactPair,
    state: AugmentedLagrangeState,
    slave_displacement: FloatArray | None = None,
    master_displacement: FloatArray | None = None,
    *,
    relative_step: float = 2.0e-7,
    freeze_facet_pairs: bool = True,
    freeze_active_rows: bool = True,
    freeze_weights: bool = False,
) -> FloatArray:
    """Return an independent centered-difference tangent with fixed multipliers."""

    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    state.validate_for(pair)
    slave_u = (
        np.zeros_like(pair.slave.reference_nodes)
        if slave_displacement is None
        else np.asarray(slave_displacement, dtype=float)
        .reshape(pair.slave.reference_nodes.shape)
        .copy()
    )
    master_u = (
        np.zeros_like(pair.master.reference_nodes)
        if master_displacement is None
        else np.asarray(master_displacement, dtype=float)
        .reshape(pair.master.reference_nodes.shape)
        .copy()
    )
    base = evaluate_augmented_lagrange(pair, state, slave_u, master_u)
    frozen_weights = base.contact.weights if freeze_weights else None
    frozen_pairs = (
        base.contact.weights.facet_pairs
        if freeze_facet_pairs and not freeze_weights
        else None
    )
    frozen_active = base.contact.active_rows if freeze_active_rows else None
    coordinates = np.vstack(
        [pair.slave.reference_nodes + slave_u, pair.master.reference_nodes + master_u]
    )
    ndof = 3 * (pair.slave.node_count + pair.master.node_count)
    tangent = np.zeros((ndof, ndof), dtype=float)

    for column in range(ndof):
        node, component = divmod(column, 3)
        step = relative_step * max(1.0, abs(float(coordinates[node, component])))
        plus_slave = slave_u.copy()
        minus_slave = slave_u.copy()
        plus_master = master_u.copy()
        minus_master = master_u.copy()
        if node < pair.slave.node_count:
            plus_slave[node, component] += step
            minus_slave[node, component] -= step
        else:
            master_node = node - pair.slave.node_count
            plus_master[master_node, component] += step
            minus_master[master_node, component] -= step
        plus = evaluate_augmented_lagrange(
            pair,
            state,
            plus_slave,
            plus_master,
            facet_pairs=frozen_pairs,
            active_rows=frozen_active,
            frozen_weights=frozen_weights,
        ).contact.residual
        minus = evaluate_augmented_lagrange(
            pair,
            state,
            minus_slave,
            minus_master,
            facet_pairs=frozen_pairs,
            active_rows=frozen_active,
            frozen_weights=frozen_weights,
        ).contact.residual
        tangent[:, column] = (plus - minus) / (2.0 * step)
    return tangent
