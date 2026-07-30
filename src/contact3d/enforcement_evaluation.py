"""Augmented-Lagrange residual evaluation and accepted multiplier updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contact import ContactEvaluation, ContactPair, GlobalMortarWeights, evaluate_contact
from .enforcement_state import (
    AugmentedLagrangeState,
    KKTDiagnostics,
    augmented_pressure_projection,
    kkt_diagnostics,
    supported_rows,
)
from .model import FloatArray
from .surface import FacetPair


@dataclass(frozen=True, slots=True)
class AugmentedLagrangeEvaluation:
    """Contact residual evaluated with an accepted multiplier state held fixed."""

    state: AugmentedLagrangeState
    contact: ContactEvaluation
    trial_pressure: FloatArray
    diagnostics: KKTDiagnostics


@dataclass(frozen=True, slots=True)
class AugmentedLagrangeUpdate:
    """One explicit multiplier augmentation after an equilibrium solve."""

    previous: AugmentedLagrangeState
    state: AugmentedLagrangeState
    increment: FloatArray
    diagnostics_before: KKTDiagnostics
    diagnostics_after: KKTDiagnostics


def evaluate_augmented_lagrange(
    pair: ContactPair,
    state: AugmentedLagrangeState,
    slave_displacement: FloatArray | None = None,
    master_displacement: FloatArray | None = None,
    *,
    facet_pairs: tuple[FacetPair, ...] | None = None,
    active_rows: np.ndarray | None = None,
    frozen_weights: GlobalMortarWeights | None = None,
    tolerance: float = 1.0e-12,
) -> AugmentedLagrangeEvaluation:
    """Evaluate one inner Newton state with accepted multipliers held fixed."""

    state.validate_for(pair)
    geometry = evaluate_contact(
        pair,
        slave_displacement,
        master_displacement,
        facet_pairs=facet_pairs,
        frozen_weights=frozen_weights,
        tolerance=tolerance,
    )
    supported = supported_rows(geometry.weights, tolerance)
    trial, projected, projected_active = augmented_pressure_projection(
        state.multipliers,
        geometry.normal_gaps,
        pair.normal_penalty,
        supported,
    )
    if active_rows is None:
        active = projected_active
        pressure = projected
    else:
        active = np.asarray(active_rows, dtype=bool)
        if active.shape != (pair.slave.node_count,):
            raise ValueError("active_rows must match the slave-node count")
        active = active & supported
        pressure = np.where(active, trial, 0.0)

    traction = pressure[:, None] * geometry.nodal_normals
    slave_force = geometry.weights.d.T @ traction
    master_force = -(geometry.weights.m.T @ traction)
    contact = ContactEvaluation(
        residual=np.concatenate([slave_force.ravel(), master_force.ravel()]),
        slave_nodes=geometry.slave_nodes,
        master_nodes=geometry.master_nodes,
        slave_force=slave_force,
        master_force=master_force,
        nodal_normals=geometry.nodal_normals,
        weighted_gap_vectors=geometry.weighted_gap_vectors,
        weighted_normal_gaps=geometry.weighted_normal_gaps,
        normal_gaps=geometry.normal_gaps,
        pressure=pressure,
        active_rows=active,
        weights=geometry.weights,
    )
    diagnostics = kkt_diagnostics(
        state.multipliers,
        contact.normal_gaps,
        pair.normal_penalty,
        supported,
    )
    return AugmentedLagrangeEvaluation(state, contact, trial, diagnostics)


def augment_multipliers(
    pair: ContactPair,
    evaluation: AugmentedLagrangeEvaluation,
    *,
    tolerance: float = 1.0e-12,
) -> AugmentedLagrangeUpdate:
    """Project accepted multipliers after the current equilibrium solve."""

    evaluation.state.validate_for(pair)
    supported = supported_rows(evaluation.contact.weights, tolerance)
    _, projected, _ = augmented_pressure_projection(
        evaluation.state.multipliers,
        evaluation.contact.normal_gaps,
        pair.normal_penalty,
        supported,
    )
    next_state = AugmentedLagrangeState(
        projected,
        augmentation=evaluation.state.augmentation + 1,
    )
    diagnostics_after = kkt_diagnostics(
        next_state.multipliers,
        evaluation.contact.normal_gaps,
        pair.normal_penalty,
        supported,
    )
    return AugmentedLagrangeUpdate(
        previous=evaluation.state,
        state=next_state,
        increment=next_state.multipliers - evaluation.state.multipliers,
        diagnostics_before=evaluation.diagnostics,
        diagnostics_after=diagnostics_after,
    )
