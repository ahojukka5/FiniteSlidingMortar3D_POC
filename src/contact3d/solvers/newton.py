"""Fixed-multiplier Newton solution for coupled bulk-contact equilibrium."""

from __future__ import annotations

import numpy as np

from ..clipping import ClippingTopologyError
from ..coupling import (
    CoupledEquilibriumEvaluation,
    CoupledEquilibriumProblem,
    evaluate_coupled_equilibrium,
)
from ..equilibrium import NewtonOptions
from ..mechanics import BulkGeometryError, FloatArray
from ..mortar.enforcement import AugmentedLagrangeState
from ..pallets import PalletTopologyError
from ..parametric import InverseMapTopologyError
from .linear import LinearSolveDiagnostics, solve_reduced_system
from .results import (
    ContactEventPolicy,
    CoupledNewtonIteration,
    CoupledNewtonResult,
    CoupledTerminationReason,
)

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
            displacement,
            load_factor,
            True,
            "converged",
            evaluation,
            (),
            0,
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
                    evaluation.free_residual_norm,
                    initial_norm,
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


__all__ = ["solve_coupled_equilibrium"]
