from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from contact3d.adaptive import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    contact_penalties,
    solve_adaptive_contact_path,
    with_contact_penalties,
)
from contact3d.coupled import AugmentedContactOptions
from contact3d.enforcement_state import AugmentedLagrangeState


@dataclass(frozen=True, slots=True)
class FakeMesh:
    node_count: int


@dataclass(frozen=True, slots=True)
class FakePair:
    normal_penalty: float


@dataclass(frozen=True, slots=True)
class FakeInterface:
    pair: FakePair

    @property
    def normal_penalty(self) -> float:
        return self.pair.normal_penalty

    def with_normal_penalty(self, normal_penalty: float) -> FakeInterface:
        return replace(self, pair=replace(self.pair, normal_penalty=normal_penalty))

    def reference_tributary_areas(self) -> np.ndarray:
        return np.ones(1)

    def initial_state(self) -> AugmentedLagrangeState:
        return AugmentedLagrangeState.zeros(1)


@dataclass(frozen=True, slots=True)
class DirectPenaltyInterface:
    penalty: float

    @property
    def normal_penalty(self) -> float:
        return self.penalty

    def with_normal_penalty(self, normal_penalty: float) -> DirectPenaltyInterface:
        return replace(self, penalty=normal_penalty)

    def reference_tributary_areas(self) -> np.ndarray:
        return np.ones(1)

    def initial_state(self) -> AugmentedLagrangeState:
        return AugmentedLagrangeState.zeros(1)


@dataclass(frozen=True, slots=True)
class FakeProblem:
    mesh: FakeMesh
    interfaces: tuple[object, ...]

    def initial_states(self) -> tuple[AugmentedLagrangeState, ...]:
        return tuple(interface.initial_state() for interface in self.interfaces)


@dataclass(frozen=True, slots=True)
class FakeEquilibrium:
    displacement: np.ndarray
    iteration_count: int
    contact_event_restarts: int
    evaluation: object


@dataclass(frozen=True, slots=True)
class FakeAugmentedResult:
    displacement: np.ndarray
    states: tuple[AugmentedLagrangeState, ...]
    converged: bool
    termination_reason: str
    equilibrium: FakeEquilibrium
    equilibria: tuple[FakeEquilibrium, ...]
    history: tuple[object, ...]


def result(
    *,
    load_factor: float,
    penalty: float,
    converged: bool,
    reason: str,
    penetration: float,
    newton_iterations: int,
    state_value: float,
) -> FakeAugmentedResult:
    displacement = np.full(6, load_factor + penalty * 1.0e-8)
    state = AugmentedLagrangeState(np.array([state_value]))
    evaluation = SimpleNamespace(
        maximum_penetration=penetration,
        free_residual_norm=1.0e-12 if converged else 2.0e-4,
    )
    equilibrium = FakeEquilibrium(
        displacement=displacement,
        iteration_count=newton_iterations,
        contact_event_restarts=int(load_factor > 0.7),
        evaluation=evaluation,
    )
    history = tuple(range(2 if converged else 3))
    return FakeAugmentedResult(
        displacement=displacement,
        states=(state,),
        converged=converged,
        termination_reason=reason,
        equilibrium=equilibrium,
        equilibria=(equilibrium,),
        history=history,
    )


def problem(penalty: float = 100.0) -> FakeProblem:
    return FakeProblem(FakeMesh(2), (FakeInterface(FakePair(penalty)),))


def test_penalty_replacement_is_immutable() -> None:
    original = problem(125.0)
    changed = with_contact_penalties(original, (500.0,))
    assert contact_penalties(original) == (125.0,)
    assert contact_penalties(changed) == (500.0,)


def test_direct_penalty_interface_replacement() -> None:
    original = FakeProblem(FakeMesh(2), (DirectPenaltyInterface(75.0),))
    changed = with_contact_penalties(original, (300.0,))
    assert contact_penalties(original) == (75.0,)
    assert contact_penalties(changed) == (300.0,)


def test_cutback_penalty_retry_and_growth_are_transactional() -> None:
    calls: list[tuple[float, float, float]] = []

    def solver(coupled_problem, displacement, states, *, load_factor, options, tolerance):
        penalty = contact_penalties(coupled_problem)[0]
        state_value = states[0].multipliers[0]
        calls.append((load_factor, penalty, state_value))
        if np.isclose(load_factor, 0.8) and np.isclose(penalty, 100.0):
            return result(
                load_factor=load_factor,
                penalty=penalty,
                converged=False,
                reason="inner_equilibrium_failed",
                penetration=3.0e-3,
                newton_iterations=12,
                state_value=state_value,
            )
        if np.isclose(load_factor, 0.4) and np.isclose(penalty, 100.0):
            return result(
                load_factor=load_factor,
                penalty=penalty,
                converged=False,
                reason="maximum_augmentations",
                penetration=2.0e-3,
                newton_iterations=7,
                state_value=7.0,
            )
        return result(
            load_factor=load_factor,
            penalty=penalty,
            converged=True,
            reason="converged",
            penetration=4.0e-9,
            newton_iterations=3,
            state_value=11.0 + load_factor,
        )

    options = AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.8,
            minimum_step=0.05,
            maximum_step=0.8,
            growth_factor=2.0,
            easy_newton_iterations=5,
        ),
        penalty=AdaptivePenaltyOptions(
            increase_factor=4.0,
            maximum_penalty=1600.0,
            maximum_updates_per_step=2,
        ),
        augmented=AugmentedContactOptions(maximum_augmentations=3, gap_tolerance=1.0e-8),
    )
    solved = solve_adaptive_contact_path(
        problem(),
        1.0,
        options=options,
        _solver=solver,
    )

    assert solved.converged
    assert solved.load_factor == pytest.approx(1.0)
    assert solved.cutback_count == 1
    assert solved.penalty_update_count == 1
    assert solved.accepted_step_count == 3
    assert [attempt.action for attempt in solved.attempts] == [
        "cutback",
        "penalty_increase",
        "accepted",
        "accepted",
        "accepted",
    ]
    assert contact_penalties(solved.problem) == (400.0,)
    assert calls[:3] == [
        (0.8, 100.0, 0.0),
        (0.4, 100.0, 0.0),
        (0.4, 400.0, 7.0),
    ]
    assert calls[-1][0] == pytest.approx(1.0)
    assert calls[-1][1] == pytest.approx(400.0)


def test_failed_penalty_retry_rolls_back_problem_and_state_before_cutback() -> None:
    calls: list[tuple[float, float, float]] = []

    def solver(coupled_problem, displacement, states, *, load_factor, options, tolerance):
        penalty = contact_penalties(coupled_problem)[0]
        state_value = states[0].multipliers[0]
        calls.append((load_factor, penalty, state_value))
        if (
            np.isclose(load_factor, 0.5)
            and np.isclose(penalty, 100.0)
            and np.isclose(state_value, 0.0)
        ):
            return result(
                load_factor=load_factor,
                penalty=penalty,
                converged=False,
                reason="maximum_augmentations",
                penetration=1.0e-3,
                newton_iterations=4,
                state_value=3.0,
            )
        if np.isclose(load_factor, 0.5) and np.isclose(penalty, 400.0):
            return result(
                load_factor=load_factor,
                penalty=penalty,
                converged=False,
                reason="inner_equilibrium_failed",
                penetration=8.0e-4,
                newton_iterations=9,
                state_value=4.0,
            )
        return result(
            load_factor=load_factor,
            penalty=penalty,
            converged=True,
            reason="converged",
            penetration=1.0e-9,
            newton_iterations=3,
            state_value=8.0,
        )

    options = AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.5,
            minimum_step=0.1,
            maximum_step=0.5,
        ),
        penalty=AdaptivePenaltyOptions(maximum_updates_per_step=1),
        augmented=AugmentedContactOptions(maximum_augmentations=2),
    )
    solved = solve_adaptive_contact_path(problem(), 0.5, options=options, _solver=solver)
    assert solved.converged
    assert [attempt.action for attempt in solved.attempts] == [
        "penalty_increase",
        "cutback",
        "accepted",
        "accepted",
    ]
    assert calls[2] == (0.25, 100.0, 0.0)
    assert contact_penalties(solved.problem) == (100.0,)


def test_minimum_step_failure_returns_last_accepted_state() -> None:
    def solver(coupled_problem, displacement, states, *, load_factor, options, tolerance):
        return result(
            load_factor=load_factor,
            penalty=contact_penalties(coupled_problem)[0],
            converged=False,
            reason="inner_equilibrium_failed",
            penetration=1.0,
            newton_iterations=1,
            state_value=99.0,
        )

    initial_displacement = np.arange(6, dtype=float)
    initial_state = (AugmentedLagrangeState(np.array([2.0])),)
    options = AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.2,
            minimum_step=0.1,
            maximum_step=0.2,
            cutback_factor=0.5,
        ),
        penalty=AdaptivePenaltyOptions(enabled=False),
    )
    solved = solve_adaptive_contact_path(
        problem(),
        1.0,
        initial_displacement,
        initial_state,
        options=options,
        _solver=solver,
    )
    assert not solved.converged
    assert solved.termination_reason == "minimum_step"
    np.testing.assert_array_equal(solved.displacement, initial_displacement)
    np.testing.assert_array_equal(solved.states[0].multipliers, initial_state[0].multipliers)
    assert solved.load_factor == 0.0


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: AdaptiveLoadOptions(initial_step=0.1, minimum_step=0.2),
        lambda: AdaptiveLoadOptions(cutback_factor=1.0),
        lambda: AdaptivePenaltyOptions(increase_factor=1.0),
        lambda: AdaptivePenaltyOptions(penetration_target=-1.0),
    ],
)
def test_option_validation(constructor) -> None:
    with pytest.raises(ValueError):
        constructor()
