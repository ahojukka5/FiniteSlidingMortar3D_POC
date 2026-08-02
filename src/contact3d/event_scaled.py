"""Scale-aware augmented contact with event-localized inner Newton solves."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .coupled import AugmentedContactOptions, CoupledEquilibriumProblem
from .enforcement_state import AugmentedLagrangeState
from .event_model import EventAwareCoupledNewtonResult
from .event_newton import solve_event_aware_coupled_equilibrium
from .model import FloatArray
from .multiplier_transport import MultiplierTransportRecord
from .scaled_solver import (
    ScaleAwareAugmentationIteration,
    ScaleAwareNewtonIteration,
    _all_kkt_converged,
    _augmentation_row,
    _scaled_newton_history,
    _validated_states,
)
from .scaling import (
    CoupledProblemScales,
    ScaleAwareConvergenceOptions,
    coupled_problem_scales,
)
from .topology_events import (
    ContactTopologyEventBatch,
    TopologyEventLocalizationOptions,
)


@dataclass(frozen=True, slots=True)
class EventAwareScaleAwareAugmentedContactResult:
    """Scale-aware augmented result retaining every localized event batch."""

    displacement: FloatArray
    states: tuple[AugmentedLagrangeState, ...]
    converged: bool
    termination_reason: str
    equilibrium: EventAwareCoupledNewtonResult
    equilibria: tuple[EventAwareCoupledNewtonResult, ...]
    history: tuple[ScaleAwareAugmentationIteration, ...]
    scales: CoupledProblemScales
    newton_histories: tuple[tuple[ScaleAwareNewtonIteration, ...], ...]

    @property
    def events(self) -> tuple[ContactTopologyEventBatch, ...]:
        """Return localized event batches in augmentation and Newton order."""

        return tuple(batch for equilibrium in self.equilibria for batch in equilibrium.events)

    @property
    def multiplier_transports(self) -> tuple[MultiplierTransportRecord, ...]:
        """Return support transports in augmentation and Newton order."""

        return tuple(
            record
            for equilibrium in self.equilibria
            for record in getattr(equilibrium, "multiplier_transports", ())
        )


def solve_event_aware_scale_aware_augmented_contact(
    problem: CoupledEquilibriumProblem,
    initial_displacement: FloatArray | None = None,
    initial_states: tuple[AugmentedLagrangeState, ...] | None = None,
    *,
    load_factor: float = 1.0,
    options: AugmentedContactOptions | None = None,
    scaling: ScaleAwareConvergenceOptions | None = None,
    event_options: TopologyEventLocalizationOptions | None = None,
    tolerance: float = 1.0e-12,
) -> EventAwareScaleAwareAugmentedContactResult:
    """Solve normalized augmented contact and localize every recoverable event."""

    settings = AugmentedContactOptions() if options is None else options
    scale_settings = (
        ScaleAwareConvergenceOptions(enabled=True) if scaling is None else scaling
    )
    if not scale_settings.enabled:
        raise ValueError("scale-aware augmented solve requires enabled scaling options")

    scales = coupled_problem_scales(problem)
    newton = replace(
        settings.newton,
        absolute_tolerance=scale_settings.equilibrium_tolerance * scales.force,
    )
    states = _validated_states(problem, initial_states)
    displacement = initial_displacement
    history: list[ScaleAwareAugmentationIteration] = []
    equilibria: list[EventAwareCoupledNewtonResult] = []
    newton_histories: list[tuple[ScaleAwareNewtonIteration, ...]] = []
    last_equilibrium: EventAwareCoupledNewtonResult | None = None

    for augmentation in range(settings.maximum_augmentations):
        equilibrium = solve_event_aware_coupled_equilibrium(
            problem,
            states,
            displacement,
            load_factor=load_factor,
            options=newton,
            event_policy=settings.event_policy,
            event_options=event_options,
            tolerance=tolerance,
        )
        last_equilibrium = equilibrium
        equilibria.append(equilibrium)
        newton_histories.append(_scaled_newton_history(equilibrium, scales))
        returned_states = tuple(getattr(equilibrium, "states", ()))
        if returned_states:
            states = returned_states
        if not equilibrium.converged:
            return EventAwareScaleAwareAugmentedContactResult(
                equilibrium.displacement,
                states,
                False,
                "inner_equilibrium_failed",
                equilibrium,
                tuple(equilibria),
                tuple(history),
                scales,
                tuple(newton_histories),
            )

        displacement = equilibrium.displacement
        contacts = equilibrium.evaluation.contacts
        if _all_kkt_converged(contacts, scales, scale_settings):
            history.append(
                _augmentation_row(
                    augmentation=augmentation,
                    equilibrium=equilibrium,
                    problem=problem,
                    scales=scales,
                    increments=None,
                )
            )
            return EventAwareScaleAwareAugmentedContactResult(
                displacement,
                states,
                True,
                "converged",
                equilibrium,
                tuple(equilibria),
                tuple(history),
                scales,
                tuple(newton_histories),
            )

        updates = tuple(
            interface.augment(contact, tolerance=tolerance)
            for interface, contact in zip(problem.interfaces, contacts, strict=True)
        )
        history.append(
            _augmentation_row(
                augmentation=augmentation,
                equilibrium=equilibrium,
                problem=problem,
                scales=scales,
                increments=tuple(update.increment for update in updates),
            )
        )
        if augmentation + 1 == settings.maximum_augmentations:
            break
        states = tuple(update.state for update in updates)

    assert last_equilibrium is not None
    return EventAwareScaleAwareAugmentedContactResult(
        last_equilibrium.displacement,
        states,
        False,
        "maximum_augmentations",
        last_equilibrium,
        tuple(equilibria),
        tuple(history),
        scales,
        tuple(newton_histories),
    )
