"""Transactional adaptive continuation and normal-penalty control."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np

from .adaptive_model import (
    AdaptiveAcceptedStep,
    AdaptiveAttemptAction,
    AdaptiveContactAttempt,
    AdaptiveContactResult,
)
from .adaptive_options import (
    AdaptiveContactOptions,
    AdaptivePenaltyOptions,
)
from .coupled import (
    AugmentedContactResult,
    CoupledEquilibriumProblem,
    solve_augmented_contact,
)
from .enforcement_state import AugmentedLagrangeState
from .load_path import CoupledLoadPath, CoupledPathState, LoadFactorPath
from .model import FloatArray

AdaptiveSolver = Callable[..., AugmentedContactResult]


def _interface_penalty(interface: object) -> float:
    pair = getattr(interface, "pair", None)
    if pair is not None and hasattr(pair, "normal_penalty"):
        value = float(pair.normal_penalty)
    elif hasattr(interface, "penalty"):
        value = float(getattr(interface, "penalty"))
    else:
        raise TypeError(
            "adaptive penalty control requires an interface with pair.normal_penalty "
            "or penalty"
        )
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("contact-interface penalty must be finite and positive")
    return value


def _with_interface_penalty(interface: object, penalty: float) -> object:
    pair = getattr(interface, "pair", None)
    if pair is not None and hasattr(pair, "normal_penalty"):
        return replace(interface, pair=replace(pair, normal_penalty=penalty))
    if hasattr(interface, "penalty"):
        return replace(interface, penalty=penalty)
    raise TypeError("contact interface does not support normal-penalty replacement")


def contact_penalties(problem: CoupledEquilibriumProblem) -> tuple[float, ...]:
    """Return every interface normal penalty in problem order."""

    return tuple(_interface_penalty(interface) for interface in problem.interfaces)


def with_contact_penalties(
    problem: CoupledEquilibriumProblem,
    penalties: tuple[float, ...],
) -> CoupledEquilibriumProblem:
    """Return an equivalent coupled problem with replaced interface penalties."""

    values = tuple(float(value) for value in penalties)
    if len(values) != len(problem.interfaces):
        raise ValueError("one penalty is required for every contact interface")
    interfaces = tuple(
        _with_interface_penalty(interface, penalty)
        for interface, penalty in zip(problem.interfaces, values, strict=True)
    )
    return replace(problem, interfaces=interfaces)


def _maximum_penetration(result: AugmentedContactResult) -> float:
    return float(result.equilibrium.evaluation.maximum_penetration)


def _equilibrium_residual(result: AugmentedContactResult) -> float:
    return float(result.equilibrium.evaluation.free_residual_norm)


def _newton_iterations(result: AugmentedContactResult) -> int:
    return sum(equilibrium.iteration_count for equilibrium in result.equilibria)


def _contact_event_restarts(result: AugmentedContactResult) -> int:
    return sum(equilibrium.contact_event_restarts for equilibrium in result.equilibria)


def _constrained_reaction(result: AugmentedContactResult) -> FloatArray:
    evaluation = result.equilibrium.evaluation
    residual = getattr(evaluation, "residual", None)
    free_dofs = getattr(evaluation, "free_dofs", None)
    if residual is None or free_dofs is None:
        return np.empty(0, dtype=float)
    values = np.asarray(residual, dtype=float).reshape(-1)
    free = np.asarray(free_dofs, dtype=np.int64).reshape(-1)
    reaction = np.zeros_like(values)
    constrained = np.ones(len(values), dtype=bool)
    constrained[free] = False
    reaction[constrained] = values[constrained]
    return reaction


def _attempt(
    *,
    attempt: int,
    start: float,
    target: float,
    action: AdaptiveAttemptAction,
    result: AugmentedContactResult,
    penalties_before: tuple[float, ...],
    penalties_after: tuple[float, ...],
    path_state: CoupledPathState,
    reaction: FloatArray,
) -> AdaptiveContactAttempt:
    return AdaptiveContactAttempt(
        attempt=attempt,
        start_load_factor=start,
        target_load_factor=target,
        step_size=target - start,
        action=action,
        inner_termination_reason=str(result.termination_reason),
        augmentations=len(result.history),
        newton_iterations=_newton_iterations(result),
        contact_event_restarts=_contact_event_restarts(result),
        equilibrium_residual=_equilibrium_residual(result),
        maximum_penetration=_maximum_penetration(result),
        penalties_before=penalties_before,
        penalties_after=penalties_after,
        path_values=path_state.values,
        prescribed_dofs=tuple(int(value) for value in path_state.prescribed_dofs),
        prescribed_values=tuple(float(value) for value in path_state.prescribed_values),
        effective_load_norm=path_state.effective_load_norm,
        reaction_norm=float(np.linalg.norm(reaction)),
    )


def _increased_penalties(
    penalties: tuple[float, ...],
    options: AdaptivePenaltyOptions,
) -> tuple[float, ...]:
    return tuple(
        min(options.maximum_penalty, options.increase_factor * penalty)
        for penalty in penalties
    )


def _apply_path_constraints(
    displacement: FloatArray,
    path_state: CoupledPathState,
) -> FloatArray:
    values = np.asarray(displacement, dtype=float).reshape(-1).copy()
    if len(path_state.prescribed_dofs):
        if np.any(path_state.prescribed_dofs >= len(values)):
            raise ValueError("path-prescribed dof is outside the displacement vector")
        values[path_state.prescribed_dofs] = path_state.prescribed_values
    return values


def _result(
    *,
    problem: CoupledEquilibriumProblem,
    displacement: FloatArray,
    states: tuple[AugmentedLagrangeState, ...],
    load_factor: float,
    converged: bool,
    termination_reason: str,
    accepted_results: list[AugmentedContactResult],
    attempts: list[AdaptiveContactAttempt],
    accepted_steps: list[AdaptiveAcceptedStep],
) -> AdaptiveContactResult:
    return AdaptiveContactResult(
        problem,
        displacement,
        states,
        load_factor,
        converged,
        termination_reason,
        tuple(accepted_results),
        tuple(attempts),
        tuple(accepted_steps),
    )


def solve_adaptive_contact_path(
    problem: CoupledEquilibriumProblem,
    target_load_factor: float,
    initial_displacement: FloatArray | None = None,
    initial_states: tuple[AugmentedLagrangeState, ...] | None = None,
    *,
    initial_load_factor: float = 0.0,
    path: CoupledLoadPath | None = None,
    options: AdaptiveContactOptions | None = None,
    tolerance: float = 1.0e-12,
    _solver: AdaptiveSolver = solve_augmented_contact,
) -> AdaptiveContactResult:
    """Advance an immutable boundary/load path with cutback and penalty control.

    With ``path=None`` this is backward compatible with the original scalar dead-load
    continuation: the supplied problem stays fixed and ``load_factor`` is passed to the
    inner solver. A custom path may instead replace prescribed values and the dead-load
    vector at every candidate while returning its own inner-solver load factor.

    A failed candidate rolls displacement, multipliers, penalties, boundary values, and
    named path parameters back to the last accepted state. A penalty increase is committed
    only after the retried candidate reaches coupled equilibrium and the requested KKT
    tolerances.
    """

    settings = AdaptiveContactOptions() if options is None else options
    if not np.isfinite(initial_load_factor) or initial_load_factor < 0.0:
        raise ValueError("initial_load_factor must be finite and nonnegative")
    if not np.isfinite(target_load_factor) or target_load_factor <= initial_load_factor:
        raise ValueError("target_load_factor must exceed the initial load factor")

    continuation = LoadFactorPath() if path is None else path
    accepted_path_state = continuation.evaluate(problem, float(initial_load_factor))
    accepted_problem = accepted_path_state.problem
    total_dofs = 3 * accepted_problem.mesh.node_count
    accepted_displacement = (
        np.zeros(total_dofs, dtype=float)
        if initial_displacement is None
        else np.asarray(initial_displacement, dtype=float).reshape(-1).copy()
    )
    if accepted_displacement.shape != (total_dofs,):
        raise ValueError("initial_displacement must match the mesh DOF count")
    accepted_displacement = _apply_path_constraints(
        accepted_displacement,
        accepted_path_state,
    )
    accepted_states = (
        accepted_problem.initial_states() if initial_states is None else tuple(initial_states)
    )
    if len(accepted_states) != len(accepted_problem.interfaces):
        raise ValueError("one initial state is required for every contact interface")

    accepted_factor = float(initial_load_factor)
    accepted_results: list[AugmentedContactResult] = []
    accepted_steps: list[AdaptiveAcceptedStep] = []
    attempts: list[AdaptiveContactAttempt] = []
    step = settings.load.initial_step
    attempt_index = 0

    while accepted_factor < target_load_factor:
        if attempt_index >= settings.load.maximum_attempts:
            return _result(
                problem=accepted_problem,
                displacement=accepted_displacement,
                states=accepted_states,
                load_factor=accepted_factor,
                converged=False,
                termination_reason="maximum_attempts",
                accepted_results=accepted_results,
                attempts=attempts,
                accepted_steps=accepted_steps,
            )

        candidate = min(target_load_factor, accepted_factor + step)
        trial_path_state = continuation.evaluate(accepted_problem, candidate)
        trial_problem = trial_path_state.problem
        trial_displacement = accepted_displacement.copy()
        trial_states = accepted_states
        penalty_updates = 0

        while True:
            attempt_index += 1
            penalties_before = contact_penalties(trial_problem)
            result = _solver(
                trial_problem,
                trial_displacement,
                trial_states,
                load_factor=trial_path_state.solver_load_factor,
                options=settings.augmented,
                tolerance=tolerance,
            )
            reaction = _constrained_reaction(result)
            if result.converged:
                penalties_after = contact_penalties(trial_problem)
                attempts.append(
                    _attempt(
                        attempt=attempt_index,
                        start=accepted_factor,
                        target=candidate,
                        action="accepted",
                        result=result,
                        penalties_before=penalties_before,
                        penalties_after=penalties_after,
                        path_state=trial_path_state,
                        reaction=reaction,
                    )
                )
                accepted_problem = trial_problem
                accepted_path_state = trial_path_state
                accepted_displacement = result.displacement.copy()
                accepted_states = tuple(result.states)
                accepted_factor = candidate
                accepted_results.append(result)
                accepted_steps.append(
                    AdaptiveAcceptedStep(accepted_path_state, result, reaction)
                )
                if (
                    _newton_iterations(result) <= settings.load.easy_newton_iterations
                    and penalty_updates == 0
                ):
                    step = min(settings.load.maximum_step, settings.load.growth_factor * step)
                break

            penetration_target = (
                settings.augmented.gap_tolerance
                if settings.penalty.penetration_target is None
                else settings.penalty.penetration_target
            )
            may_increase = (
                settings.penalty.enabled
                and result.termination_reason == "maximum_augmentations"
                and _maximum_penetration(result) > penetration_target
                and penalty_updates < settings.penalty.maximum_updates_per_step
            )
            increased = _increased_penalties(penalties_before, settings.penalty)
            may_increase = may_increase and increased != penalties_before
            if may_increase:
                attempts.append(
                    _attempt(
                        attempt=attempt_index,
                        start=accepted_factor,
                        target=candidate,
                        action="penalty_increase",
                        result=result,
                        penalties_before=penalties_before,
                        penalties_after=increased,
                        path_state=trial_path_state,
                        reaction=reaction,
                    )
                )
                trial_problem = with_contact_penalties(trial_problem, increased)
                trial_path_state = trial_path_state.with_problem(trial_problem)
                trial_displacement = result.displacement.copy()
                trial_states = tuple(result.states)
                penalty_updates += 1
                if attempt_index >= settings.load.maximum_attempts:
                    break
                continue

            attempts.append(
                _attempt(
                    attempt=attempt_index,
                    start=accepted_factor,
                    target=candidate,
                    action="cutback",
                    result=result,
                    penalties_before=penalties_before,
                    penalties_after=contact_penalties(accepted_problem),
                    path_state=trial_path_state,
                    reaction=reaction,
                )
            )
            step *= settings.load.cutback_factor
            if step < settings.load.minimum_step:
                return _result(
                    problem=accepted_problem,
                    displacement=accepted_displacement,
                    states=accepted_states,
                    load_factor=accepted_factor,
                    converged=False,
                    termination_reason="minimum_step",
                    accepted_results=accepted_results,
                    attempts=attempts,
                    accepted_steps=accepted_steps,
                )
            break

    return _result(
        problem=accepted_problem,
        displacement=accepted_displacement,
        states=accepted_states,
        load_factor=accepted_factor,
        converged=True,
        termination_reason="converged",
        accepted_results=accepted_results,
        attempts=attempts,
        accepted_steps=accepted_steps,
    )
