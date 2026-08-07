"""Event-aware projected augmented-Lagrange contact iteration."""

from __future__ import annotations

import numpy as np

from ...coupling import CoupledEquilibriumProblem
from ...mechanics import FloatArray
from ...mortar.enforcement import AugmentedLagrangeState
from ...topology_events import TopologyEventLocalizationOptions
from ..results import AugmentationIteration, AugmentedContactOptions
from .newton import solve_event_aware_coupled_equilibrium
from .results import EventAwareAugmentedContactResult, EventAwareCoupledNewtonResult


def _all_kkt_converged(evaluation, options: AugmentedContactOptions) -> bool:
    return all(
        contact.diagnostics.converged(
            gap_tolerance=options.gap_tolerance,
            complementarity_tolerance=options.complementarity_tolerance,
            projection_tolerance=options.projection_tolerance,
            multiplier_tolerance=options.multiplier_tolerance,
        )
        for contact in evaluation.contacts
    )


def _augmentation_row(
    augmentation: int,
    equilibrium: EventAwareCoupledNewtonResult,
    increment: float,
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
            (contact.diagnostics.maximum_projection_residual for contact in contacts),
            default=0.0,
        ),
        maximum_multiplier_increment=increment,
        active_rows=sum(
            int(np.count_nonzero(contact.signature.active_rows)) for contact in contacts
        ),
        maximum_pressure=max(
            (float(np.max(contact.pressure, initial=0.0)) for contact in contacts),
            default=0.0,
        ),
    )


def solve_event_aware_augmented_contact(
    problem: CoupledEquilibriumProblem,
    initial_displacement: FloatArray | None = None,
    initial_states: tuple[AugmentedLagrangeState, ...] | None = None,
    *,
    load_factor: float = 1.0,
    options: AugmentedContactOptions | None = None,
    event_options: TopologyEventLocalizationOptions | None = None,
    tolerance: float = 1.0e-12,
) -> EventAwareAugmentedContactResult:
    """Run projected augmentation with event-localized inner Newton solves."""

    settings = AugmentedContactOptions() if options is None else options
    states = problem.validate_states(initial_states)
    displacement = initial_displacement
    history: list[AugmentationIteration] = []
    equilibria: list[EventAwareCoupledNewtonResult] = []
    last: EventAwareCoupledNewtonResult | None = None
    for augmentation in range(settings.maximum_augmentations):
        equilibrium = solve_event_aware_coupled_equilibrium(
            problem,
            states,
            displacement,
            load_factor=load_factor,
            options=settings.newton,
            event_policy=settings.event_policy,
            event_options=event_options,
            tolerance=tolerance,
        )
        equilibria.append(equilibrium)
        last = equilibrium
        returned_states = tuple(getattr(equilibrium, "states", ()))
        if returned_states:
            states = returned_states
        if not equilibrium.converged:
            return EventAwareAugmentedContactResult(
                equilibrium.displacement,
                states,
                False,
                "inner_equilibrium_failed",
                equilibrium,
                tuple(equilibria),
                tuple(history),
            )
        displacement = equilibrium.displacement
        if _all_kkt_converged(equilibrium.evaluation, settings):
            history.append(_augmentation_row(augmentation, equilibrium, 0.0))
            return EventAwareAugmentedContactResult(
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
            for interface, contact in zip(
                problem.interfaces,
                equilibrium.evaluation.contacts,
                strict=True,
            )
        )
        increment = max(
            (
                float(np.max(np.abs(update.increment), initial=0.0))
                for update in updates
            ),
            default=0.0,
        )
        history.append(_augmentation_row(augmentation, equilibrium, increment))
        if augmentation + 1 == settings.maximum_augmentations:
            break
        states = tuple(update.state for update in updates)
    assert last is not None
    return EventAwareAugmentedContactResult(
        last.displacement,
        states,
        False,
        "maximum_augmentations",
        last,
        tuple(equilibria),
        tuple(history),
    )


__all__ = ["solve_event_aware_augmented_contact"]
