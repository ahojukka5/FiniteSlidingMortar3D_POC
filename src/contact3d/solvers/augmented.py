"""Projected augmented-Lagrange orchestration for coupled contact."""

from __future__ import annotations

import numpy as np

from ..coupling import ContactInterfaceEvaluation, CoupledEquilibriumProblem
from ..mechanics import FloatArray
from ..mortar.enforcement import AugmentedLagrangeState
from .newton import solve_coupled_equilibrium
from .results import (
    AugmentationIteration,
    AugmentedContactOptions,
    AugmentedContactResult,
)


def _all_kkt_converged(
    contacts: tuple[ContactInterfaceEvaluation, ...],
    options: AugmentedContactOptions,
) -> bool:
    return all(
        contact.diagnostics.converged(
            gap_tolerance=options.gap_tolerance,
            complementarity_tolerance=options.complementarity_tolerance,
            projection_tolerance=options.projection_tolerance,
            multiplier_tolerance=options.multiplier_tolerance,
        )
        for contact in contacts
    )


def _augmentation_record(
    augmentation: int,
    equilibrium,
    *,
    multiplier_increment: float,
) -> AugmentationIteration:
    contacts = equilibrium.evaluation.contacts
    return AugmentationIteration(
        augmentation=augmentation,
        newton_iterations=equilibrium.iteration_count,
        contact_event_restarts=equilibrium.contact_event_restarts,
        equilibrium_residual=equilibrium.evaluation.free_residual_norm,
        maximum_penetration=max(
            (contact.diagnostics.maximum_penetration for contact in contacts),
            default=0.0,
        ),
        maximum_complementarity=max(
            (contact.diagnostics.maximum_complementarity for contact in contacts),
            default=0.0,
        ),
        maximum_projection_residual=max(
            (
                contact.diagnostics.maximum_projection_residual
                for contact in contacts
            ),
            default=0.0,
        ),
        maximum_multiplier_increment=multiplier_increment,
        active_rows=sum(
            int(np.count_nonzero(contact.signature.active_rows))
            for contact in contacts
        ),
        maximum_pressure=max(
            (
                float(np.max(contact.pressure, initial=0.0))
                for contact in contacts
            ),
            default=0.0,
        ),
    )


def solve_augmented_contact(
    problem: CoupledEquilibriumProblem,
    initial_displacement: FloatArray | None = None,
    initial_states: tuple[AugmentedLagrangeState, ...] | None = None,
    *,
    load_factor: float = 1.0,
    options: AugmentedContactOptions | None = None,
    tolerance: float = 1.0e-12,
) -> AugmentedContactResult:
    """Alternate fixed-multiplier equilibrium solves and accepted AL updates."""

    settings = AugmentedContactOptions() if options is None else options
    states = problem.validate_states(initial_states)
    displacement = initial_displacement
    history: list[AugmentationIteration] = []
    last_equilibrium = None
    equilibria = []
    for augmentation in range(settings.maximum_augmentations):
        equilibrium = solve_coupled_equilibrium(
            problem,
            states,
            displacement,
            load_factor=load_factor,
            options=settings.newton,
            event_policy=settings.event_policy,
            tolerance=tolerance,
        )
        last_equilibrium = equilibrium
        equilibria.append(equilibrium)
        if not equilibrium.converged:
            return AugmentedContactResult(
                equilibrium.displacement,
                states,
                False,
                "inner_equilibrium_failed",
                equilibrium,
                tuple(equilibria),
                tuple(history),
            )
        displacement = equilibrium.displacement
        contacts = equilibrium.evaluation.contacts
        if _all_kkt_converged(contacts, settings):
            history.append(
                _augmentation_record(
                    augmentation,
                    equilibrium,
                    multiplier_increment=0.0,
                )
            )
            return AugmentedContactResult(
                displacement,
                states,
                True,
                "converged",
                equilibrium,
                tuple(equilibria),
                tuple(history),
            )
        updates = tuple(
            interface.augment(contact, tolerance=tolerance)
            for interface, contact in zip(problem.interfaces, contacts, strict=True)
        )
        increment = max(
            (
                float(np.max(np.abs(update.increment), initial=0.0))
                for update in updates
            ),
            default=0.0,
        )
        history.append(
            _augmentation_record(
                augmentation,
                equilibrium,
                multiplier_increment=increment,
            )
        )
        if augmentation + 1 == settings.maximum_augmentations:
            break
        states = tuple(update.state for update in updates)
    assert last_equilibrium is not None
    return AugmentedContactResult(
        last_equilibrium.displacement,
        states,
        False,
        "maximum_augmentations",
        last_equilibrium,
        tuple(equilibria),
        tuple(history),
    )


__all__ = ["solve_augmented_contact"]
