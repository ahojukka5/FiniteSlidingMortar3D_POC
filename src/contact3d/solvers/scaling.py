"""Scale-aware augmented-Lagrange solution for coupled contact."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..coupling import (
    ContactInterfaceEvaluation,
    CoupledEquilibriumProblem,
)
from ..mechanics import FloatArray
from ..mortar.enforcement import AugmentedLagrangeState
from ..scaling import (
    ContactScaleIndicators,
    CoupledProblemScales,
    NormalizedKKTDiagnostics,
    ScaleAwareConvergenceOptions,
    contact_scale_indicators,
    coupled_problem_scales,
    interface_normal_penalty,
)
from .newton import solve_coupled_equilibrium
from .results import AugmentedContactOptions, CoupledNewtonResult


@dataclass(frozen=True, slots=True)
class ScaleAwareNewtonIteration:
    """One Newton row with dimensional and normalized equilibrium residuals."""

    iteration: int
    residual_norm: float
    normalized_residual: float
    relative_residual: float
    bulk_potential: float
    normalized_bulk_potential: float
    minimum_jacobian: float
    maximum_penetration: float
    normalized_maximum_penetration: float
    step_norm: float
    normalized_step_norm: float
    accepted_step: float
    line_search_iterations: int
    contact_branch_changed: bool


@dataclass(frozen=True, slots=True)
class ScaleAwareAugmentationIteration:
    """One augmentation row with dimensional and normalized Newton/KKT data."""

    augmentation: int
    newton_iterations: int
    contact_event_restarts: int
    equilibrium_residual: float
    normalized_equilibrium_residual: float
    maximum_penetration: float
    normalized_maximum_penetration: float
    maximum_complementarity: float
    normalized_maximum_complementarity: float
    maximum_projection_residual: float
    normalized_maximum_projection_residual: float
    maximum_multiplier_increment: float
    normalized_maximum_multiplier_increment: float
    active_rows: int
    maximum_pressure: float
    normalized_maximum_pressure: float
    interfaces: tuple[ContactScaleIndicators, ...]


@dataclass(frozen=True, slots=True)
class ScaleAwareAugmentedContactResult:
    """Augmented result preserving dimensional and normalized histories."""

    displacement: FloatArray
    states: tuple[AugmentedLagrangeState, ...]
    converged: bool
    termination_reason: str
    equilibrium: CoupledNewtonResult
    equilibria: tuple[CoupledNewtonResult, ...]
    history: tuple[ScaleAwareAugmentationIteration, ...]
    scales: CoupledProblemScales
    newton_histories: tuple[tuple[ScaleAwareNewtonIteration, ...], ...]


def _normalized_contacts(
    contacts: tuple[ContactInterfaceEvaluation, ...],
    scales: CoupledProblemScales,
) -> tuple[NormalizedKKTDiagnostics, ...]:
    return tuple(
        scale.normalize_kkt(contact.diagnostics)
        for contact, scale in zip(contacts, scales.interfaces, strict=True)
    )


def _all_kkt_converged(
    contacts: tuple[ContactInterfaceEvaluation, ...],
    scales: CoupledProblemScales,
    options: ScaleAwareConvergenceOptions,
) -> bool:
    return all(value.converged(options) for value in _normalized_contacts(contacts, scales))


def _scaled_newton_history(
    equilibrium: CoupledNewtonResult,
    scales: CoupledProblemScales,
) -> tuple[ScaleAwareNewtonIteration, ...]:
    maximum_length = max(scale.length for scale in scales.interfaces)
    return tuple(
        ScaleAwareNewtonIteration(
            iteration=row.iteration,
            residual_norm=row.residual_norm,
            normalized_residual=row.residual_norm / scales.force,
            relative_residual=row.relative_residual,
            bulk_potential=row.bulk_potential,
            normalized_bulk_potential=row.bulk_potential / scales.energy,
            minimum_jacobian=row.minimum_jacobian,
            maximum_penetration=row.maximum_penetration,
            normalized_maximum_penetration=row.maximum_penetration / maximum_length,
            step_norm=row.step_norm,
            normalized_step_norm=row.step_norm / scales.length,
            accepted_step=row.accepted_step,
            line_search_iterations=row.line_search_iterations,
            contact_branch_changed=row.contact_branch_changed,
        )
        for row in equilibrium.history
    )


def _interface_indicators(
    problem: CoupledEquilibriumProblem,
    contacts: tuple[ContactInterfaceEvaluation, ...],
    scales: CoupledProblemScales,
) -> tuple[ContactScaleIndicators, ...]:
    return tuple(
        contact_scale_indicators(
            contact,
            scale,
            interface_normal_penalty(interface),
        )
        for interface, contact, scale in zip(
            problem.interfaces,
            contacts,
            scales.interfaces,
            strict=True,
        )
    )


def _augmentation_row(
    *,
    augmentation: int,
    equilibrium: CoupledNewtonResult,
    problem: CoupledEquilibriumProblem,
    scales: CoupledProblemScales,
    increments: tuple[np.ndarray, ...] | None,
) -> ScaleAwareAugmentationIteration:
    contacts = equilibrium.evaluation.contacts
    normalized = _normalized_contacts(contacts, scales)
    indicators = _interface_indicators(problem, contacts, scales)
    dimensional_increment = 0.0
    normalized_increment = 0.0
    if increments is not None:
        dimensional_increment = max(
            (float(np.max(np.abs(value), initial=0.0)) for value in increments),
            default=0.0,
        )
        normalized_increment = max(
            (
                float(np.max(np.abs(value), initial=0.0)) / scale.pressure
                for value, scale in zip(increments, scales.interfaces, strict=True)
            ),
            default=0.0,
        )
    return ScaleAwareAugmentationIteration(
        augmentation=augmentation,
        newton_iterations=equilibrium.iteration_count,
        contact_event_restarts=equilibrium.contact_event_restarts,
        equilibrium_residual=equilibrium.evaluation.free_residual_norm,
        normalized_equilibrium_residual=(
            equilibrium.evaluation.free_residual_norm / scales.force
        ),
        maximum_penetration=max(
            (contact.diagnostics.maximum_penetration for contact in contacts),
            default=0.0,
        ),
        normalized_maximum_penetration=max(
            (value.maximum_penetration for value in normalized),
            default=0.0,
        ),
        maximum_complementarity=max(
            (contact.diagnostics.maximum_complementarity for contact in contacts),
            default=0.0,
        ),
        normalized_maximum_complementarity=max(
            (value.maximum_complementarity for value in normalized),
            default=0.0,
        ),
        maximum_projection_residual=max(
            (contact.diagnostics.maximum_projection_residual for contact in contacts),
            default=0.0,
        ),
        normalized_maximum_projection_residual=max(
            (value.maximum_projection_residual for value in normalized),
            default=0.0,
        ),
        maximum_multiplier_increment=dimensional_increment,
        normalized_maximum_multiplier_increment=normalized_increment,
        active_rows=sum(value.active_rows for value in indicators),
        maximum_pressure=max((value.pressure for value in indicators), default=0.0),
        normalized_maximum_pressure=max(
            (value.normalized_pressure for value in indicators),
            default=0.0,
        ),
        interfaces=indicators,
    )


def solve_scale_aware_augmented_contact(
    problem: CoupledEquilibriumProblem,
    initial_displacement: FloatArray | None = None,
    initial_states: tuple[AugmentedLagrangeState, ...] | None = None,
    *,
    load_factor: float = 1.0,
    options: AugmentedContactOptions | None = None,
    scaling: ScaleAwareConvergenceOptions | None = None,
    tolerance: float = 1.0e-12,
) -> ScaleAwareAugmentedContactResult:
    """Solve augmented contact with unit-consistent Newton and KKT thresholds."""

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
    states = problem.validate_states(initial_states)
    displacement = initial_displacement
    history: list[ScaleAwareAugmentationIteration] = []
    equilibria: list[CoupledNewtonResult] = []
    newton_histories: list[tuple[ScaleAwareNewtonIteration, ...]] = []
    last_equilibrium: CoupledNewtonResult | None = None

    for augmentation in range(settings.maximum_augmentations):
        equilibrium = solve_coupled_equilibrium(
            problem,
            states,
            displacement,
            load_factor=load_factor,
            options=newton,
            event_policy=settings.event_policy,
            tolerance=tolerance,
        )
        last_equilibrium = equilibrium
        equilibria.append(equilibrium)
        newton_histories.append(_scaled_newton_history(equilibrium, scales))
        if not equilibrium.converged:
            return ScaleAwareAugmentedContactResult(
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
            return ScaleAwareAugmentedContactResult(
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
    return ScaleAwareAugmentedContactResult(
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


__all__ = [
    "ScaleAwareAugmentationIteration",
    "ScaleAwareAugmentedContactResult",
    "ScaleAwareNewtonIteration",
    "solve_scale_aware_augmented_contact",
]
