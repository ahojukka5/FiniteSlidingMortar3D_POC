"""Event-localized fixed-multiplier Newton equilibrium."""

from __future__ import annotations

import numpy as np

from .bulk_material import BulkGeometryError
from .coupled import (
    ContactEventPolicy,
    CoupledEquilibriumEvaluation,
    CoupledEquilibriumProblem,
    CoupledNewtonIteration,
    CoupledTerminationReason,
    evaluate_coupled_equilibrium,
)
from .equilibrium import NewtonOptions
from .event_geometry import (
    _RECOVERABLE_ERRORS,
    _event_signatures,
    _observation,
    _relative_residual,
    _validated_states,
)
from .enforcement_state import AugmentedLagrangeState
from .event_model import EventAwareCoupledNewtonResult
from .model import FloatArray
from .topology_events import (
    ContactTopologyEventBatch,
    ContactTopologyStateMachine,
    TopologyEventLocalizationOptions,
    TopologyObservation,
)


def _result(
    displacement: FloatArray,
    load_factor: float,
    converged: bool,
    reason: CoupledTerminationReason,
    evaluation: CoupledEquilibriumEvaluation,
    history: list[CoupledNewtonIteration],
    events: list[ContactTopologyEventBatch],
) -> EventAwareCoupledNewtonResult:
    return EventAwareCoupledNewtonResult(
        displacement.copy(),
        float(load_factor),
        converged,
        reason,
        evaluation,
        tuple(history),
        tuple(events),
    )


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
    states = _validated_states(problem, states)
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
        )

    for iteration in range(settings.maximum_iterations):
        free = evaluation.free_dofs
        assert evaluation.tangent is not None
        tangent = evaluation.tangent.extract_dense(free, free)
        try:
            step_free = np.linalg.solve(tangent, -evaluation.residual[free])
        except np.linalg.LinAlgError:
            return _result(
                displacement,
                load_factor,
                False,
                "singular_tangent",
                evaluation,
                history,
                events,
            )
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
        for line_iteration in range(settings.maximum_line_search_iterations):
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
                                states,
                                displacement,
                                step,
                                fraction,
                                load_factor=load_factor,
                                tolerance=tolerance,
                            ),
                        ).restarted()
                    except BulkGeometryError:
                        localized = None
                    if localized is not None:
                        accepted = localized.selected.payload
                        accepted_alpha = localized.selected_fraction
                        branch_changed = True
                        break
                trial_merit = 0.5 * trial.free_residual_norm**2
                armijo = merit + settings.armijo_coefficient * alpha * slope
                acceptable = trial.free_residual_norm <= threshold or trial_merit <= armijo
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
            )
        displacement = accepted.displacement.copy()
        if localized is not None:
            events.append(localized)
        try:
            evaluation = evaluate_coupled_equilibrium(
                problem,
                displacement,
                states,
                load_factor=load_factor,
                tolerance=tolerance,
            )
        except _RECOVERABLE_ERRORS:
            return _result(
                displacement,
                load_factor,
                False,
                "contact_linearization_event",
                accepted,
                history,
                events,
            )
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
            )
    return _result(
        displacement,
        load_factor,
        False,
        "maximum_iterations",
        evaluation,
        history,
        events,
    )


