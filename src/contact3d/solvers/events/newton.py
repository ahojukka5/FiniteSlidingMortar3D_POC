"""Event-localized fixed-multiplier Newton equilibrium."""

from __future__ import annotations

import numpy as np

from ...coupling import (
    CoupledEquilibriumEvaluation,
    CoupledEquilibriumProblem,
    evaluate_coupled_equilibrium,
)
from ...equilibrium import NewtonOptions
from ...mechanics import BulkGeometryError, FloatArray
from ...mortar.enforcement import AugmentedLagrangeState
from ...topology_events import (
    ContactTopologyEventBatch,
    ContactTopologyStateMachine,
    TopologyEventLocalizationOptions,
    TopologyObservation,
)
from ..linear import LinearSolveDiagnostics, solve_reduced_system
from ..results import (
    ContactEventPolicy,
    CoupledNewtonIteration,
    CoupledTerminationReason,
)
from .localization import (
    RECOVERABLE_CONTACT_EVENT_ERRORS as _RECOVERABLE_ERRORS,
)
from .localization import (
    event_signatures as _event_signatures,
)
from .localization import (
    observe_event_trial as _observation,
)
from .multiplier_transport import (
    MultiplierTransportRecord,
    transport_multiplier_states,
)
from .results import EventAwareCoupledNewtonResult

_RESTART_CLEARANCE_WEIGHTS = tuple(2.0**-exponent for exponent in range(24, -1, -1))


def _relative_residual(norm: float, initial_norm: float) -> float:
    return norm / max(initial_norm, np.finfo(float).tiny)


def _linear_failure_reason(
    diagnostics: LinearSolveDiagnostics,
) -> CoupledTerminationReason:
    if diagnostics.failure_reason in {"singular_matrix", "factorization_failed"}:
        return "singular_tangent"
    return "linear_solve_failed"


def _result(
    displacement: FloatArray,
    load_factor: float,
    converged: bool,
    reason: CoupledTerminationReason,
    evaluation: CoupledEquilibriumEvaluation,
    history: list[CoupledNewtonIteration],
    events: list[ContactTopologyEventBatch],
    states: tuple[AugmentedLagrangeState, ...],
    transports: list[MultiplierTransportRecord],
    linear_solve_failure: LinearSolveDiagnostics | None = None,
) -> EventAwareCoupledNewtonResult:
    return EventAwareCoupledNewtonResult(
        displacement.copy(),
        float(load_factor),
        converged,
        reason,
        evaluation,
        tuple(history),
        tuple(events),
        linear_solve_failure,
        tuple(states),
        tuple(transports),
    )


def _restart_evaluation_with_clearance(
    problem: CoupledEquilibriumProblem,
    states: tuple[AugmentedLagrangeState, ...],
    displacement: FloatArray,
    step: FloatArray,
    localized: ContactTopologyEventBatch,
    trial_fraction: float,
    *,
    load_factor: float,
    tolerance: float,
) -> tuple[float, CoupledEquilibriumEvaluation] | None:
    """Return the nearest tangent-capable point inside the selected branch."""

    selected = localized.selected
    signatures = selected.signatures
    start = float(selected.fraction)
    stop = float(trial_fraction)
    if signatures is None or stop <= start:
        return None
    span = stop - start
    for weight in _RESTART_CLEARANCE_WEIGHTS:
        fraction = start + weight * span
        try:
            observation = _observation(
                problem,
                states,
                displacement,
                step,
                fraction,
                load_factor=load_factor,
                tolerance=tolerance,
            )
        except BulkGeometryError:
            continue
        if not observation.is_valid or observation.signatures != signatures:
            continue
        try:
            evaluation = evaluate_coupled_equilibrium(
                problem,
                displacement + fraction * step,
                states,
                load_factor=load_factor,
                tolerance=tolerance,
            )
        except (BulkGeometryError, *_RECOVERABLE_ERRORS):
            continue
        return fraction, evaluation
    return None


def solve_event_aware_coupled_equilibrium(
    problem: CoupledEquilibriumProblem,
    states: tuple[AugmentedLagrangeState, ...],
    initial_displacement: FloatArray | None = None,
    *,
    load_factor: float = 1.0,
    options: NewtonOptions | None = None,
    event_policy: ContactEventPolicy = "restart",
    event_options: TopologyEventLocalizationOptions | None = None,
    tolerance: float = 1.0e-12,
) -> EventAwareCoupledNewtonResult:
    """Solve one equilibrium state and restart exactly after localized events."""

    if event_policy not in ("restart", "reject"):
        raise ValueError("event_policy must be 'restart' or 'reject'")
    settings = NewtonOptions() if options is None else options
    states = problem.validate_states(states)
    machine = ContactTopologyStateMachine(event_options)
    total_dofs = 3 * problem.mesh.node_count
    displacement = (
        np.zeros(total_dofs, dtype=float)
        if initial_displacement is None
        else np.asarray(initial_displacement, dtype=float).reshape(-1).copy()
    )
    if displacement.shape != (total_dofs,):
        raise ValueError("initial_displacement must match the mesh DOF count")
    displacement = problem.constraints.apply(displacement)
    history: list[CoupledNewtonIteration] = []
    events: list[ContactTopologyEventBatch] = []
    transports: list[MultiplierTransportRecord] = []
    try:
        evaluation = evaluate_coupled_equilibrium(
            problem,
            displacement,
            states,
            load_factor=load_factor,
            tolerance=tolerance,
        )
    except _RECOVERABLE_ERRORS:
        residual_only = evaluate_coupled_equilibrium(
            problem,
            displacement,
            states,
            load_factor=load_factor,
            assemble_tangent=False,
            tolerance=tolerance,
        )
        return _result(
            displacement,
            load_factor,
            False,
            "contact_linearization_event",
            residual_only,
            history,
            events,
            states,
            transports,
        )

    initial_norm = evaluation.free_residual_norm
    threshold = max(
        settings.absolute_tolerance,
        settings.relative_tolerance * initial_norm,
    )
    if evaluation.free_residual_norm <= threshold:
        return _result(
            displacement,
            load_factor,
            True,
            "converged",
            evaluation,
            history,
            events,
            states,
            transports,
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
            return _result(
                displacement,
                load_factor,
                False,
                _linear_failure_reason(linear_result.diagnostics),
                evaluation,
                history,
                events,
                states,
                transports,
                linear_result.diagnostics,
            )
        step_free = linear_result.solution
        step = np.zeros(total_dofs, dtype=float)
        step[free] = step_free
        merit = 0.5 * evaluation.free_residual_norm**2
        slope = -evaluation.free_residual_norm**2
        accepted: CoupledEquilibriumEvaluation | None = None
        accepted_alpha = 0.0
        branch_changed = False
        localized: ContactTopologyEventBatch | None = None
        alpha = 1.0
        line_iteration = 0
        for line_iteration in range(settings.maximum_line_search_iterations):  # noqa: B007
            try:
                trial_observation = _observation(
                    problem,
                    states,
                    displacement,
                    step,
                    alpha,
                    load_factor=load_factor,
                    tolerance=tolerance,
                )
            except BulkGeometryError:
                trial_observation = None
            if trial_observation is not None and trial_observation.is_valid:
                trial = trial_observation.payload
                current_signatures = _event_signatures(
                    problem,
                    evaluation,
                    tolerance=tolerance,
                )
                changed = trial_observation.signatures != current_signatures
                if changed and event_policy == "restart":
                    left = TopologyObservation.valid(0.0, current_signatures, evaluation)
                    try:
                        localized = machine.localize(
                            left,
                            trial_observation,
                            lambda fraction: _observation(
                                problem,
                                states,  # noqa: B023
                                displacement,  # noqa: B023
                                step,  # noqa: B023
                                fraction,
                                load_factor=load_factor,
                                tolerance=tolerance,
                            ),
                        ).restarted()
                    except BulkGeometryError:
                        localized = None
                    if localized is not None:
                        assert localized.selected.signatures is not None
                        states, updates = transport_multiplier_states(
                            states,
                            current_signatures,
                            localized.selected.signatures,
                        )
                        transports.extend(updates)
                        accepted = localized.selected.payload
                        accepted_alpha = localized.selected_fraction
                        branch_changed = True
                        break
                trial_merit = 0.5 * trial.free_residual_norm**2
                armijo = merit + settings.armijo_coefficient * alpha * slope
                acceptable = (
                    trial.free_residual_norm <= threshold or trial_merit <= armijo
                )
                if changed and event_policy == "reject":
                    acceptable = False
                if acceptable:
                    accepted = trial
                    accepted_alpha = alpha
                    branch_changed = changed
                    break
            alpha *= settings.line_search_reduction
            if alpha < settings.minimum_step:
                break
        if accepted is None:
            return _result(
                displacement,
                load_factor,
                False,
                "line_search_failed",
                evaluation,
                history,
                events,
                states,
                transports,
            )
        candidate_displacement = accepted.displacement.copy()
        if localized is not None:
            events.append(localized)
        try:
            evaluation = evaluate_coupled_equilibrium(
                problem,
                candidate_displacement,
                states,
                load_factor=load_factor,
                tolerance=tolerance,
            )
        except _RECOVERABLE_ERRORS:
            recovered = (
                None
                if localized is None
                else _restart_evaluation_with_clearance(
                    problem,
                    states,
                    displacement,
                    step,
                    localized,
                    alpha,
                    load_factor=load_factor,
                    tolerance=tolerance,
                )
            )
            if recovered is None:
                return _result(
                    candidate_displacement,
                    load_factor,
                    False,
                    "contact_linearization_event",
                    accepted,
                    history,
                    events,
                    states,
                    transports,
                )
            accepted_alpha, evaluation = recovered
            candidate_displacement = evaluation.displacement.copy()
        displacement = candidate_displacement
        history.append(
            CoupledNewtonIteration(
                iteration=iteration + 1,
                residual_norm=evaluation.free_residual_norm,
                relative_residual=_relative_residual(
                    evaluation.free_residual_norm,
                    initial_norm,
                ),
                bulk_potential=evaluation.bulk_potential,
                minimum_jacobian=evaluation.bulk.minimum_jacobian,
                maximum_penetration=evaluation.maximum_penetration,
                step_norm=float(np.linalg.norm(step_free)),
                accepted_step=accepted_alpha,
                line_search_iterations=line_iteration,
                contact_branch_changed=branch_changed,
                linear_solve=linear_result.diagnostics,
            )
        )
        if evaluation.free_residual_norm <= threshold:
            return _result(
                displacement,
                load_factor,
                True,
                "converged",
                evaluation,
                history,
                events,
                states,
                transports,
            )
    return _result(
        displacement,
        load_factor,
        False,
        "maximum_iterations",
        evaluation,
        history,
        events,
        states,
        transports,
    )


__all__ = ["solve_event_aware_coupled_equilibrium"]
