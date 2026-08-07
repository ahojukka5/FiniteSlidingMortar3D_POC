"""Legacy solver entry points built on the coupling subsystem.

The coupling contracts, adapter, problem model, and assembly implementation
live in :mod:`contact3d.coupling`. Nonlinear solver ownership remains here
temporarily and moves to :mod:`contact3d.solvers` in issue #131.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .clipping import ClippingTopologyError
from .coupling import (
    ContactBranchSignature,
    ContactInterfaceEvaluation,
    ContactInterfaceUpdate,
    CoupledContactInterface,
    CoupledEquilibriumEvaluation,
    CoupledEquilibriumProblem,
    MortarContactInterface,
    evaluate_coupled_equilibrium,
)
from .equilibrium import NewtonOptions
from .linear_solver import LinearSolveDiagnostics, solve_reduced_system
from .mechanics import BulkGeometryError, FloatArray
from .mortar.enforcement import AugmentedLagrangeState
from .pallets import PalletTopologyError
from .parametric import InverseMapTopologyError

__all__ = [
    "AugmentationIteration",
    "AugmentedContactOptions",
    "AugmentedContactResult",
    "AugmentedTerminationReason",
    "ContactBranchSignature",
    "ContactEventPolicy",
    "ContactInterfaceEvaluation",
    "ContactInterfaceUpdate",
    "CoupledContactInterface",
    "CoupledEquilibriumEvaluation",
    "CoupledEquilibriumProblem",
    "CoupledNewtonIteration",
    "CoupledNewtonResult",
    "CoupledTerminationReason",
    "MortarContactInterface",
    "evaluate_coupled_equilibrium",
    "solve_augmented_contact",
    "solve_coupled_equilibrium",
]


@dataclass(frozen=True, slots=True)
class CoupledNewtonIteration:
    iteration: int
    residual_norm: float
    relative_residual: float
    bulk_potential: float
    minimum_jacobian: float
    maximum_penetration: float
    step_norm: float
    accepted_step: float
    line_search_iterations: int
    contact_branch_changed: bool
    linear_solve: LinearSolveDiagnostics


CoupledTerminationReason = Literal[
    "converged",
    "maximum_iterations",
    "line_search_failed",
    "singular_tangent",
    "linear_solve_failed",
    "contact_linearization_event",
]
ContactEventPolicy = Literal["restart", "reject"]


@dataclass(frozen=True, slots=True)
class CoupledNewtonResult:
    displacement: FloatArray
    load_factor: float
    converged: bool
    termination_reason: CoupledTerminationReason
    evaluation: CoupledEquilibriumEvaluation
    history: tuple[CoupledNewtonIteration, ...]
    contact_event_restarts: int
    linear_solve_failure: LinearSolveDiagnostics | None = None

    @property
    def iteration_count(self) -> int:
        return len(self.history)


@dataclass(frozen=True, slots=True)
class AugmentedContactOptions:
    maximum_augmentations: int = 12
    gap_tolerance: float = 1.0e-8
    complementarity_tolerance: float = 1.0e-8
    projection_tolerance: float = 1.0e-8
    multiplier_tolerance: float = 1.0e-8
    event_policy: ContactEventPolicy = "restart"
    newton: NewtonOptions = field(default_factory=NewtonOptions)

    def __post_init__(self) -> None:
        if self.maximum_augmentations <= 0:
            raise ValueError("maximum_augmentations must be positive")
        for value in (
            self.gap_tolerance,
            self.complementarity_tolerance,
            self.projection_tolerance,
            self.multiplier_tolerance,
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    "augmented-contact tolerances must be finite and nonnegative"
                )
        if self.event_policy not in ("restart", "reject"):
            raise ValueError("event_policy must be 'restart' or 'reject'")


@dataclass(frozen=True, slots=True)
class AugmentationIteration:
    augmentation: int
    newton_iterations: int
    contact_event_restarts: int
    equilibrium_residual: float
    maximum_penetration: float
    maximum_complementarity: float
    maximum_projection_residual: float
    maximum_multiplier_increment: float
    active_rows: int
    maximum_pressure: float


AugmentedTerminationReason = Literal[
    "converged",
    "maximum_augmentations",
    "inner_equilibrium_failed",
]


@dataclass(frozen=True, slots=True)
class AugmentedContactResult:
    displacement: FloatArray
    states: tuple[AugmentedLagrangeState, ...]
    converged: bool
    termination_reason: AugmentedTerminationReason
    equilibrium: CoupledNewtonResult
    equilibria: tuple[CoupledNewtonResult, ...]
    history: tuple[AugmentationIteration, ...]


_CONTACT_EVENT_ERRORS = (
    ClippingTopologyError,
    PalletTopologyError,
    InverseMapTopologyError,
)


def _relative_residual(norm: float, initial_norm: float) -> float:
    return norm / max(initial_norm, np.finfo(float).tiny)


def _linear_failure_reason(
    diagnostics: LinearSolveDiagnostics,
) -> CoupledTerminationReason:
    if diagnostics.failure_reason in {"singular_matrix", "factorization_failed"}:
        return "singular_tangent"
    return "linear_solve_failed"


def solve_coupled_equilibrium(
    problem: CoupledEquilibriumProblem,
    states: tuple[AugmentedLagrangeState, ...],
    initial_displacement: FloatArray | None = None,
    *,
    load_factor: float = 1.0,
    options: NewtonOptions | None = None,
    event_policy: ContactEventPolicy = "restart",
    tolerance: float = 1.0e-12,
) -> CoupledNewtonResult:
    """Solve one fixed-multiplier equilibrium state with contact-event restarts."""

    if event_policy not in ("restart", "reject"):
        raise ValueError("event_policy must be 'restart' or 'reject'")
    settings = NewtonOptions() if options is None else options
    states = problem.validate_states(states)
    total_dofs = 3 * problem.mesh.node_count
    displacement = (
        np.zeros(total_dofs, dtype=float)
        if initial_displacement is None
        else np.asarray(initial_displacement, dtype=float).reshape(-1).copy()
    )
    if displacement.shape != (total_dofs,):
        raise ValueError("initial_displacement must match the mesh DOF count")
    displacement = problem.constraints.apply(displacement)
    try:
        evaluation = evaluate_coupled_equilibrium(
            problem,
            displacement,
            states,
            load_factor=load_factor,
            tolerance=tolerance,
        )
    except _CONTACT_EVENT_ERRORS:
        residual_only = evaluate_coupled_equilibrium(
            problem,
            displacement,
            states,
            load_factor=load_factor,
            assemble_tangent=False,
            tolerance=tolerance,
        )
        return CoupledNewtonResult(
            displacement,
            load_factor,
            False,
            "contact_linearization_event",
            residual_only,
            (),
            0,
        )
    initial_norm = evaluation.free_residual_norm
    threshold = max(
        settings.absolute_tolerance,
        settings.relative_tolerance * initial_norm,
    )
    history: list[CoupledNewtonIteration] = []
    event_restarts = 0
    if evaluation.free_residual_norm <= threshold:
        return CoupledNewtonResult(
            displacement, load_factor, True, "converged", evaluation, (), 0
        )

    for iteration in range(settings.maximum_iterations):
        free = evaluation.free_dofs
        assert evaluation.tangent is not None
        linear_result = solve_reduced_system(
            evaluation.tangent,
            free,
            -evaluation.residual[free],
            options=settings.linear_solver,
        )
        if linear_result.solution is None:
            return CoupledNewtonResult(
                displacement,
                load_factor,
                False,
                _linear_failure_reason(linear_result.diagnostics),
                evaluation,
                tuple(history),
                event_restarts,
                linear_result.diagnostics,
            )
        step_free = linear_result.solution
        step = np.zeros(total_dofs, dtype=float)
        step[free] = step_free
        merit = 0.5 * evaluation.free_residual_norm**2
        slope = -evaluation.free_residual_norm**2
        accepted: CoupledEquilibriumEvaluation | None = None
        branch_changed = False
        alpha = 1.0
        line_iteration = 0
        for line_iteration in range(settings.maximum_line_search_iterations):  # noqa: B007
            try:
                trial = evaluate_coupled_equilibrium(
                    problem,
                    displacement + alpha * step,
                    states,
                    load_factor=load_factor,
                    assemble_tangent=False,
                    tolerance=tolerance,
                )
            except (BulkGeometryError, *_CONTACT_EVENT_ERRORS):
                trial = None
            if trial is not None:
                changed = trial.signatures != evaluation.signatures
                trial_merit = 0.5 * trial.free_residual_norm**2
                armijo = merit + settings.armijo_coefficient * alpha * slope
                acceptable = (
                    trial.free_residual_norm <= threshold or trial_merit <= armijo
                )
                if changed and event_policy == "reject":
                    acceptable = False
                if acceptable:
                    accepted = trial
                    branch_changed = changed
                    break
            alpha *= settings.line_search_reduction
            if alpha < settings.minimum_step:
                break
        if accepted is None:
            return CoupledNewtonResult(
                displacement,
                load_factor,
                False,
                "line_search_failed",
                evaluation,
                tuple(history),
                event_restarts,
            )
        displacement = accepted.displacement.copy()
        if branch_changed:
            event_restarts += 1
        try:
            evaluation = evaluate_coupled_equilibrium(
                problem,
                displacement,
                states,
                load_factor=load_factor,
                tolerance=tolerance,
            )
        except _CONTACT_EVENT_ERRORS:
            return CoupledNewtonResult(
                displacement,
                load_factor,
                False,
                "contact_linearization_event",
                accepted,
                tuple(history),
                event_restarts,
            )
        history.append(
            CoupledNewtonIteration(
                iteration=iteration + 1,
                residual_norm=evaluation.free_residual_norm,
                relative_residual=_relative_residual(
                    evaluation.free_residual_norm, initial_norm
                ),
                bulk_potential=evaluation.bulk_potential,
                minimum_jacobian=evaluation.bulk.minimum_jacobian,
                maximum_penetration=evaluation.maximum_penetration,
                step_norm=float(np.linalg.norm(step_free)),
                accepted_step=alpha,
                line_search_iterations=line_iteration,
                contact_branch_changed=branch_changed,
                linear_solve=linear_result.diagnostics,
            )
        )
        if evaluation.free_residual_norm <= threshold:
            return CoupledNewtonResult(
                displacement,
                load_factor,
                True,
                "converged",
                evaluation,
                tuple(history),
                event_restarts,
            )
    return CoupledNewtonResult(
        displacement,
        load_factor,
        False,
        "maximum_iterations",
        evaluation,
        tuple(history),
        event_restarts,
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
    last_equilibrium: CoupledNewtonResult | None = None
    equilibria: list[CoupledNewtonResult] = []
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
                AugmentationIteration(
                    augmentation=augmentation,
                    newton_iterations=equilibrium.iteration_count,
                    contact_event_restarts=equilibrium.contact_event_restarts,
                    equilibrium_residual=equilibrium.evaluation.free_residual_norm,
                    maximum_penetration=max(
                        (c.diagnostics.maximum_penetration for c in contacts),
                        default=0.0,
                    ),
                    maximum_complementarity=max(
                        (c.diagnostics.maximum_complementarity for c in contacts),
                        default=0.0,
                    ),
                    maximum_projection_residual=max(
                        (c.diagnostics.maximum_projection_residual for c in contacts),
                        default=0.0,
                    ),
                    maximum_multiplier_increment=0.0,
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
            AugmentationIteration(
                augmentation=augmentation,
                newton_iterations=equilibrium.iteration_count,
                contact_event_restarts=equilibrium.contact_event_restarts,
                equilibrium_residual=equilibrium.evaluation.free_residual_norm,
                maximum_penetration=max(
                    (c.diagnostics.maximum_penetration for c in contacts),
                    default=0.0,
                ),
                maximum_complementarity=max(
                    (c.diagnostics.maximum_complementarity for c in contacts),
                    default=0.0,
                ),
                maximum_projection_residual=max(
                    (c.diagnostics.maximum_projection_residual for c in contacts),
                    default=0.0,
                ),
                maximum_multiplier_increment=increment,
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
