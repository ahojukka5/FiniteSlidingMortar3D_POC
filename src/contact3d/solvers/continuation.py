"""Adaptive continuation contracts and transactional solution driver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..coupling import CoupledEquilibriumProblem
from ..mechanics import FloatArray
from ..mortar.enforcement import AugmentedLagrangeState
from ..scaling import (
    CoupledProblemScales,
    PenaltyUpdateDecision,
    PenaltyUpdatePlan,
    ScaleAwareConvergenceOptions,
    coupled_problem_scales,
    interface_normal_penalty,
    propose_interface_penalties,
    with_interface_normal_penalty,
)
from .augmented import solve_augmented_contact
from .results import AugmentedContactOptions, AugmentedContactResult
from .scaling import solve_scale_aware_augmented_contact

if TYPE_CHECKING:
    from ..load_path import CoupledLoadPath, CoupledPathState

AdaptiveAttemptAction = Literal["accepted", "cutback", "penalty_increase"]
AdaptiveTerminationReason = Literal[
    "converged",
    "minimum_step",
    "maximum_attempts",
]
AdaptiveSolver = Callable[..., object]


@dataclass(frozen=True, slots=True)
class AdaptiveLoadOptions:
    """Step-size policy for monotone load-factor continuation."""

    initial_step: float = 0.25
    minimum_step: float = 1.0 / 1024.0
    maximum_step: float = 0.5
    cutback_factor: float = 0.5
    growth_factor: float = 1.5
    easy_newton_iterations: int = 8
    maximum_attempts: int = 100

    def __post_init__(self) -> None:
        for name, value in (
            ("initial_step", self.initial_step),
            ("minimum_step", self.minimum_step),
            ("maximum_step", self.maximum_step),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.minimum_step > self.initial_step
            or self.initial_step > self.maximum_step
        ):
            raise ValueError("load steps must satisfy minimum <= initial <= maximum")
        if not 0.0 < self.cutback_factor < 1.0:
            raise ValueError("cutback_factor must lie between zero and one")
        if not np.isfinite(self.growth_factor) or self.growth_factor <= 1.0:
            raise ValueError("growth_factor must be finite and greater than one")
        if self.easy_newton_iterations < 0:
            raise ValueError("easy_newton_iterations must be nonnegative")
        if self.maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")


@dataclass(frozen=True, slots=True)
class AdaptivePenaltyOptions:
    """Escalate only under-resolved interface penalties within physical bounds."""

    enabled: bool = True
    increase_factor: float = 4.0
    maximum_penalty: float = 1.0e9
    maximum_updates_per_step: int = 4
    penetration_target: float | None = None
    normalized_penetration_target: float | None = None
    interface_local: bool = True
    minimum_scale_factor: float = 0.25
    maximum_scale_factor: float = 1.0e4

    def __post_init__(self) -> None:
        if not np.isfinite(self.increase_factor) or self.increase_factor <= 1.0:
            raise ValueError("increase_factor must be finite and greater than one")
        if not np.isfinite(self.maximum_penalty) or self.maximum_penalty <= 0.0:
            raise ValueError("maximum_penalty must be finite and positive")
        if self.maximum_updates_per_step < 0:
            raise ValueError("maximum_updates_per_step must be nonnegative")
        if self.penetration_target is not None and (
            not np.isfinite(self.penetration_target) or self.penetration_target < 0.0
        ):
            raise ValueError("penetration_target must be finite and nonnegative")
        if self.normalized_penetration_target is not None and (
            not np.isfinite(self.normalized_penetration_target)
            or self.normalized_penetration_target < 0.0
        ):
            raise ValueError(
                "normalized_penetration_target must be finite and nonnegative"
            )
        if (
            not np.isfinite(self.minimum_scale_factor)
            or self.minimum_scale_factor <= 0.0
        ):
            raise ValueError("minimum_scale_factor must be finite and positive")
        if (
            not np.isfinite(self.maximum_scale_factor)
            or self.maximum_scale_factor < self.minimum_scale_factor
        ):
            raise ValueError(
                "maximum_scale_factor must be finite and no smaller than the minimum"
            )


@dataclass(frozen=True, slots=True)
class AdaptiveContactOptions:
    """Combined continuation, penalty, inner solve, and scaling settings."""

    load: AdaptiveLoadOptions = field(default_factory=AdaptiveLoadOptions)
    penalty: AdaptivePenaltyOptions = field(default_factory=AdaptivePenaltyOptions)
    augmented: AugmentedContactOptions = field(default_factory=AugmentedContactOptions)
    scaling: ScaleAwareConvergenceOptions = field(
        default_factory=ScaleAwareConvergenceOptions
    )


@dataclass(frozen=True, slots=True)
class AdaptiveContactAttempt:
    """One accepted, cut-back, or penalty-escalated continuation attempt."""

    attempt: int
    start_load_factor: float
    target_load_factor: float
    step_size: float
    action: AdaptiveAttemptAction
    inner_termination_reason: str
    augmentations: int
    newton_iterations: int
    contact_event_restarts: int
    equilibrium_residual: float
    maximum_penetration: float
    penalties_before: tuple[float, ...]
    penalties_after: tuple[float, ...]
    path_values: tuple[tuple[str, float], ...] = ()
    prescribed_dofs: tuple[int, ...] = ()
    prescribed_values: tuple[float, ...] = ()
    effective_load_norm: float = 0.0
    reaction_norm: float = 0.0
    normalized_equilibrium_residual: float = 0.0
    normalized_maximum_penetration: float = 0.0
    interface_penetrations: tuple[float, ...] = ()
    normalized_interface_penetrations: tuple[float, ...] = ()
    penalty_ratios_before: tuple[float, ...] = ()
    penalty_ratios_after: tuple[float, ...] = ()
    penalty_update_reasons: tuple[str, ...] = ()

    @property
    def start_parameter(self) -> float:
        return self.start_load_factor

    @property
    def target_parameter(self) -> float:
        return self.target_load_factor


@dataclass(frozen=True, slots=True)
class AdaptiveAcceptedStep:
    """One committed path state, equilibrium solution, and constrained reaction."""

    path_state: CoupledPathState
    result: AugmentedContactResult
    reaction: FloatArray

    def __post_init__(self) -> None:
        values = np.asarray(self.reaction, dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("accepted-step reaction must be a finite flat vector")
        force = self.path_state.effective_force
        if len(force) and values.shape != force.shape:
            raise ValueError("accepted-step reaction must match the global force vector")
        object.__setattr__(self, "reaction", values.copy())

    @property
    def parameter(self) -> float:
        return self.path_state.parameter

    @property
    def reaction_norm(self) -> float:
        return float(np.linalg.norm(self.reaction))


@dataclass(frozen=True, slots=True)
class AdaptiveContactResult:
    """Accepted adaptive path and all rejected or retried attempts."""

    problem: CoupledEquilibriumProblem
    displacement: FloatArray
    states: tuple[AugmentedLagrangeState, ...]
    load_factor: float
    converged: bool
    termination_reason: AdaptiveTerminationReason
    accepted_results: tuple[AugmentedContactResult, ...]
    attempts: tuple[AdaptiveContactAttempt, ...]
    accepted_steps: tuple[AdaptiveAcceptedStep, ...] = ()

    @property
    def accepted_step_count(self) -> int:
        return len(self.accepted_results)

    @property
    def cutback_count(self) -> int:
        return sum(attempt.action == "cutback" for attempt in self.attempts)

    @property
    def penalty_update_count(self) -> int:
        return sum(attempt.action == "penalty_increase" for attempt in self.attempts)

    @property
    def final_path_state(self) -> CoupledPathState | None:
        return self.accepted_steps[-1].path_state if self.accepted_steps else None


def contact_penalties(problem: CoupledEquilibriumProblem) -> tuple[float, ...]:
    """Return every interface normal penalty in problem order."""

    return tuple(interface_normal_penalty(interface) for interface in problem.interfaces)


def with_contact_penalties(
    problem: CoupledEquilibriumProblem,
    penalties: tuple[float, ...],
) -> CoupledEquilibriumProblem:
    """Return an equivalent problem with immutable interface-local penalties."""

    values = tuple(float(value) for value in penalties)
    if len(values) != len(problem.interfaces):
        raise ValueError("one penalty is required for every contact interface")
    interfaces = tuple(
        with_interface_normal_penalty(interface, penalty)
        for interface, penalty in zip(problem.interfaces, values, strict=True)
    )
    updated = replace(problem, interfaces=interfaces)
    old_sparsity = getattr(problem, "sparsity", None)
    if old_sparsity is not None:
        object.__setattr__(updated, "sparsity", old_sparsity)
    return updated


def _maximum_penetration(result: object) -> float:
    return float(result.equilibrium.evaluation.maximum_penetration)


def _equilibrium_residual(result: object) -> float:
    return float(result.equilibrium.evaluation.free_residual_norm)


def _newton_iterations(result: object) -> int:
    return sum(equilibrium.iteration_count for equilibrium in result.equilibria)


def _contact_event_restarts(result: object) -> int:
    return sum(equilibrium.contact_event_restarts for equilibrium in result.equilibria)


def _constrained_reaction(result: object) -> FloatArray:
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


def _problem_scales(problem: object) -> CoupledProblemScales | None:
    try:
        return coupled_problem_scales(problem)
    except (AttributeError, TypeError, ValueError):
        return None


def _contacts(result: object) -> tuple[object, ...]:
    values = getattr(result.equilibrium.evaluation, "contacts", ())
    return tuple(values)


def _scaled_attempt_values(
    problem: object,
    result: object,
    penalties_before: tuple[float, ...],
    penalties_after: tuple[float, ...],
) -> tuple[
    float,
    float,
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]:
    scales = _problem_scales(problem)
    contacts = _contacts(result)
    if scales is None:
        return 0.0, 0.0, (), (), (), ()
    normalized_residual = _equilibrium_residual(result) / scales.force
    dimensional_penetrations = tuple(
        float(contact.diagnostics.maximum_penetration) for contact in contacts
    )
    normalized_penetrations = tuple(
        penetration / scale.length
        for penetration, scale in zip(
            dimensional_penetrations,
            scales.interfaces,
            strict=len(dimensional_penetrations) == len(scales.interfaces),
        )
    )
    normalized_maximum = max(normalized_penetrations, default=0.0)
    ratios_before = tuple(
        scale.penalty_ratio(value)
        for value, scale in zip(penalties_before, scales.interfaces, strict=True)
    )
    ratios_after = tuple(
        scale.penalty_ratio(value)
        for value, scale in zip(penalties_after, scales.interfaces, strict=True)
    )
    return (
        normalized_residual,
        normalized_maximum,
        dimensional_penetrations,
        normalized_penetrations,
        ratios_before,
        ratios_after,
    )


def _attempt(
    *,
    attempt: int,
    start: float,
    target: float,
    action: AdaptiveAttemptAction,
    result: object,
    problem: object,
    penalties_before: tuple[float, ...],
    penalties_after: tuple[float, ...],
    path_state: CoupledPathState,
    reaction: FloatArray,
    penalty_update_reasons: tuple[str, ...] = (),
) -> AdaptiveContactAttempt:
    (
        normalized_residual,
        normalized_penetration,
        interface_penetrations,
        normalized_interface_penetrations,
        ratios_before,
        ratios_after,
    ) = _scaled_attempt_values(problem, result, penalties_before, penalties_after)
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
        normalized_equilibrium_residual=normalized_residual,
        normalized_maximum_penetration=normalized_penetration,
        interface_penetrations=interface_penetrations,
        normalized_interface_penetrations=normalized_interface_penetrations,
        penalty_ratios_before=ratios_before,
        penalty_ratios_after=ratios_after,
        penalty_update_reasons=penalty_update_reasons,
    )


def _legacy_penalty_plan(
    penalties: tuple[float, ...],
    options: AdaptivePenaltyOptions,
) -> PenaltyUpdatePlan:
    proposed = tuple(
        min(options.maximum_penalty, options.increase_factor * penalty)
        for penalty in penalties
    )
    decisions = tuple(
        PenaltyUpdateDecision(
            interface=index,
            old_penalty=old,
            new_penalty=new,
            penetration=0.0,
            normalized_penetration=0.0,
            old_ratio=0.0,
            new_ratio=0.0,
            reason=(
                f"interface[{index}] legacy global penetration retry; "
                f"penalty {old:.6e} -> {new:.6e}"
            ),
        )
        for index, (old, new) in enumerate(zip(penalties, proposed, strict=True))
        if new > old
    )
    return PenaltyUpdatePlan(proposed, decisions)


def _penalty_plan(
    problem: object,
    result: object,
    options: AdaptiveContactOptions,
) -> PenaltyUpdatePlan:
    penalties = contact_penalties(problem)
    contacts = _contacts(result)
    if not contacts:
        penetration_target = (
            options.augmented.gap_tolerance
            if options.penalty.penetration_target is None
            else options.penalty.penetration_target
        )
        if _maximum_penetration(result) <= penetration_target:
            return PenaltyUpdatePlan(penalties, ())
        return _legacy_penalty_plan(penalties, options.penalty)

    normalized_target = (
        options.scaling.gap_tolerance
        if options.penalty.normalized_penetration_target is None
        else options.penalty.normalized_penetration_target
    )
    dimensional_target = (
        options.augmented.gap_tolerance
        if options.penalty.penetration_target is None
        else options.penalty.penetration_target
    )
    return propose_interface_penalties(
        problem,
        contacts,
        increase_factor=options.penalty.increase_factor,
        absolute_maximum=(
            np.finfo(float).max
            if options.scaling.enabled
            else options.penalty.maximum_penalty
        ),
        minimum_scale_factor=options.penalty.minimum_scale_factor,
        maximum_scale_factor=options.penalty.maximum_scale_factor,
        dimensional_target=dimensional_target,
        normalized_target=normalized_target,
        use_normalized_target=options.scaling.enabled,
        interface_local=options.penalty.interface_local,
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
    termination_reason: AdaptiveTerminationReason,
    accepted_results: list[object],
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


def _solve_candidate(
    solver: AdaptiveSolver,
    problem: object,
    displacement: FloatArray,
    states: tuple[AugmentedLagrangeState, ...],
    *,
    load_factor: float,
    options: AdaptiveContactOptions,
    tolerance: float,
) -> object:
    if solver is solve_augmented_contact and options.scaling.enabled:
        return solve_scale_aware_augmented_contact(
            problem,
            displacement,
            states,
            load_factor=load_factor,
            options=options.augmented,
            scaling=options.scaling,
            tolerance=tolerance,
        )
    return solver(
        problem,
        displacement,
        states,
        load_factor=load_factor,
        options=options.augmented,
        tolerance=tolerance,
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
    """Advance an immutable boundary/load path with local penalty control."""

    from ..load_path import LoadFactorPath

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
        accepted_problem.initial_states()
        if initial_states is None
        else tuple(initial_states)
    )
    if len(accepted_states) != len(accepted_problem.interfaces):
        raise ValueError("one initial state is required for every contact interface")

    accepted_factor = float(initial_load_factor)
    accepted_results: list[object] = []
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
            result = _solve_candidate(
                _solver,
                trial_problem,
                trial_displacement,
                trial_states,
                load_factor=trial_path_state.solver_load_factor,
                options=settings,
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
                        problem=trial_problem,
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
                    step = min(
                        settings.load.maximum_step,
                        settings.load.growth_factor * step,
                    )
                break

            may_increase = (
                settings.penalty.enabled
                and result.termination_reason == "maximum_augmentations"
                and penalty_updates < settings.penalty.maximum_updates_per_step
            )
            plan = (
                _penalty_plan(trial_problem, result, settings)
                if may_increase
                else PenaltyUpdatePlan(penalties_before, ())
            )
            if may_increase and plan.changed:
                attempts.append(
                    _attempt(
                        attempt=attempt_index,
                        start=accepted_factor,
                        target=candidate,
                        action="penalty_increase",
                        result=result,
                        problem=trial_problem,
                        penalties_before=penalties_before,
                        penalties_after=plan.penalties,
                        path_state=trial_path_state,
                        reaction=reaction,
                        penalty_update_reasons=plan.reasons,
                    )
                )
                trial_problem = with_contact_penalties(trial_problem, plan.penalties)
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
                    problem=trial_problem,
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


__all__ = [
    "AdaptiveAcceptedStep",
    "AdaptiveAttemptAction",
    "AdaptiveContactAttempt",
    "AdaptiveContactOptions",
    "AdaptiveContactResult",
    "AdaptiveLoadOptions",
    "AdaptivePenaltyOptions",
    "AdaptiveSolver",
    "AdaptiveTerminationReason",
    "contact_penalties",
    "solve_adaptive_contact_path",
    "with_contact_penalties",
]
