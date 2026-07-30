from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from contact3d.adaptive import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    LinearBoundaryPath,
    LinearPathValue,
    LoadFactorPath,
    contact_penalties,
    solve_adaptive_contact_path,
)
from contact3d.coupled import AugmentedContactOptions
from contact3d.enforcement_state import AugmentedLagrangeState
from contact3d.equilibrium import DeadLoad, DirichletConstraints


@dataclass(frozen=True, slots=True)
class FakeMesh:
    node_count: int


@dataclass(frozen=True, slots=True)
class Pair:
    normal_penalty: float


@dataclass(frozen=True, slots=True)
class Interface:
    pair: Pair

    def initial_state(self) -> AugmentedLagrangeState:
        return AugmentedLagrangeState.zeros(1)


@dataclass(frozen=True, slots=True)
class FakeProblem:
    mesh: FakeMesh
    constraints: DirichletConstraints
    load: DeadLoad
    interfaces: tuple[object, ...]
    sparsity: object

    def initial_states(self) -> tuple[AugmentedLagrangeState, ...]:
        return tuple(interface.initial_state() for interface in self.interfaces)


@dataclass(frozen=True, slots=True)
class Equilibrium:
    displacement: np.ndarray
    iteration_count: int
    contact_event_restarts: int
    evaluation: object


@dataclass(frozen=True, slots=True)
class Result:
    displacement: np.ndarray
    states: tuple[AugmentedLagrangeState, ...]
    converged: bool
    termination_reason: str
    equilibrium: Equilibrium
    equilibria: tuple[Equilibrium, ...]
    history: tuple[object, ...]


def problem() -> FakeProblem:
    constraints = DirichletConstraints(np.array([2, 4]), np.array([-0.12, 0.04]))
    load = DeadLoad(np.array([3.0, 0.0, 0.0, 0.0, -2.0, 0.0]))
    return FakeProblem(
        FakeMesh(2),
        constraints,
        load,
        (Interface(Pair(100.0)),),
        object(),
    )


def result(
    load_factor: float,
    penalty: float,
    *,
    converged: bool,
    reason: str,
    penetration: float,
    state: float,
) -> Result:
    displacement = np.full(6, load_factor + penalty * 1.0e-8)
    residual = np.array([0.0, 0.0, 10.0 + load_factor, 0.0, -5.0, 0.0])
    evaluation = SimpleNamespace(
        maximum_penetration=penetration,
        free_residual_norm=1.0e-12 if converged else 1.0e-3,
        residual=residual,
        free_dofs=np.array([0, 1, 3, 5], dtype=np.int64),
    )
    equilibrium = Equilibrium(displacement, 3, 0, evaluation)
    return Result(
        displacement,
        (AugmentedLagrangeState(np.array([state])),),
        converged,
        reason,
        equilibrium,
        (equilibrium,),
        (object(),),
    )


def test_load_factor_path_preserves_legacy_solver_scale() -> None:
    original = problem()
    state = LoadFactorPath().evaluate(original, 0.25)
    assert state.problem is original
    assert state.solver_load_factor == pytest.approx(0.25)
    np.testing.assert_allclose(state.effective_force, 0.25 * original.load.force)
    np.testing.assert_array_equal(state.prescribed_values, original.constraints.values)


def test_linear_path_modes_and_sparsity_identity() -> None:
    original = problem()
    marker = original.sparsity

    prescribed = LinearBoundaryPath.proportional_prescribed_displacement(original)
    state = prescribed.evaluate(original, 0.5)
    np.testing.assert_allclose(state.prescribed_values, [-0.06, 0.02])
    np.testing.assert_allclose(state.effective_force, original.load.force)
    assert state.problem.sparsity is marker

    dead = LinearBoundaryPath.proportional_dead_load(original)
    state = dead.evaluate(original, 0.5)
    np.testing.assert_allclose(state.prescribed_values, original.constraints.values)
    np.testing.assert_allclose(state.effective_force, 0.5 * original.load.force)
    assert state.problem.sparsity is marker

    mixed = LinearBoundaryPath.proportional_mixed(
        original,
        values=(LinearPathValue("tool_z", 0.0, -0.12),),
    )
    state = mixed.evaluate(original, 0.5)
    np.testing.assert_allclose(state.prescribed_values, [-0.06, 0.02])
    np.testing.assert_allclose(state.effective_force, 0.5 * original.load.force)
    assert state.value("tool_z") == pytest.approx(-0.06)
    assert state.problem.sparsity is marker


def test_mixed_path_cutback_rolls_back_complete_boundary_state() -> None:
    original = problem()
    path = LinearBoundaryPath.proportional_mixed(
        original,
        values=(LinearPathValue("tool_z", 0.0, -0.12),),
    )
    calls: list[tuple[float, np.ndarray, np.ndarray, float, float]] = []

    def solver(coupled_problem, displacement, states, *, load_factor, options, tolerance):
        penalty = contact_penalties(coupled_problem)[0]
        parameter = abs(coupled_problem.constraints.values[0]) / 0.12
        calls.append(
            (
                parameter,
                coupled_problem.constraints.values.copy(),
                coupled_problem.load.force.copy(),
                load_factor,
                states[0].multipliers[0],
            )
        )
        if np.isclose(parameter, 0.8):
            return result(
                load_factor,
                penalty,
                converged=False,
                reason="inner_equilibrium_failed",
                penetration=1.0e-3,
                state=99.0,
            )
        return result(
            load_factor,
            penalty,
            converged=True,
            reason="converged",
            penetration=1.0e-9,
            state=10.0 + parameter,
        )

    options = AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.8,
            minimum_step=0.1,
            maximum_step=0.8,
            cutback_factor=0.5,
            growth_factor=2.0,
            easy_newton_iterations=5,
        ),
        penalty=AdaptivePenaltyOptions(enabled=False),
        augmented=AugmentedContactOptions(maximum_augmentations=3),
    )
    solved = solve_adaptive_contact_path(
        original,
        1.0,
        path=path,
        options=options,
        _solver=solver,
    )

    assert solved.converged
    assert [attempt.action for attempt in solved.attempts] == [
        "cutback",
        "accepted",
        "accepted",
    ]
    assert [round(call[0], 8) for call in calls] == [0.8, 0.4, 1.0]
    np.testing.assert_allclose(calls[1][1], 0.4 * original.constraints.values)
    np.testing.assert_allclose(calls[1][2], 0.4 * original.load.force)
    assert calls[1][4] == pytest.approx(0.0)
    assert all(call[3] == pytest.approx(1.0) for call in calls)

    assert solved.accepted_step_count == 2
    assert [step.parameter for step in solved.accepted_steps] == pytest.approx([0.4, 1.0])
    np.testing.assert_allclose(
        solved.accepted_steps[-1].path_state.prescribed_values,
        original.constraints.values,
    )
    np.testing.assert_allclose(
        solved.accepted_steps[-1].path_state.effective_force,
        original.load.force,
    )
    np.testing.assert_allclose(
        solved.accepted_steps[-1].reaction,
        [0.0, 0.0, 11.0, 0.0, -5.0, 0.0],
    )
    assert solved.attempts[-1].path_values == (("tool_z", -0.12),)
    assert solved.attempts[-1].effective_load_norm == pytest.approx(
        np.linalg.norm(original.load.force)
    )


def test_initial_path_constraints_are_applied_to_rollback_state() -> None:
    original = problem()
    start = DirichletConstraints(original.constraints.dofs, np.array([0.01, -0.02]))
    path = LinearBoundaryPath(
        start,
        original.constraints,
        DeadLoad(np.zeros(6)),
        original.load,
    )

    def solver(coupled_problem, displacement, states, *, load_factor, options, tolerance):
        return result(
            load_factor,
            contact_penalties(coupled_problem)[0],
            converged=False,
            reason="inner_equilibrium_failed",
            penetration=1.0,
            state=9.0,
        )

    options = AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.2,
            minimum_step=0.1,
            maximum_step=0.2,
            cutback_factor=0.5,
        ),
        penalty=AdaptivePenaltyOptions(enabled=False),
    )
    initial = np.arange(6, dtype=float)
    solved = solve_adaptive_contact_path(
        original,
        1.0,
        initial,
        path=path,
        options=options,
        _solver=solver,
    )
    assert not solved.converged
    assert solved.load_factor == 0.0
    expected = initial.copy()
    expected[start.dofs] = start.values
    np.testing.assert_allclose(solved.displacement, expected)


def test_failed_penalty_retry_restores_mixed_boundary_state_before_cutback() -> None:
    original = problem()
    path = LinearBoundaryPath.proportional_mixed(original)
    calls: list[tuple[float, float, float, np.ndarray, np.ndarray]] = []

    def solver(coupled_problem, displacement, states, *, load_factor, options, tolerance):
        penalty = contact_penalties(coupled_problem)[0]
        parameter = abs(coupled_problem.constraints.values[0]) / 0.12
        state_value = states[0].multipliers[0]
        calls.append(
            (
                parameter,
                penalty,
                state_value,
                coupled_problem.constraints.values.copy(),
                coupled_problem.load.force.copy(),
            )
        )
        if (
            np.isclose(parameter, 0.5)
            and np.isclose(penalty, 100.0)
            and np.isclose(state_value, 0.0)
        ):
            return result(
                load_factor,
                penalty,
                converged=False,
                reason="maximum_augmentations",
                penetration=1.0e-3,
                state=3.0,
            )
        if np.isclose(parameter, 0.5) and np.isclose(penalty, 400.0):
            return result(
                load_factor,
                penalty,
                converged=False,
                reason="inner_equilibrium_failed",
                penetration=8.0e-4,
                state=4.0,
            )
        return result(
            load_factor,
            penalty,
            converged=True,
            reason="converged",
            penetration=1.0e-9,
            state=8.0,
        )

    options = AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.5,
            minimum_step=0.1,
            maximum_step=0.5,
        ),
        penalty=AdaptivePenaltyOptions(
            increase_factor=4.0,
            maximum_updates_per_step=1,
        ),
        augmented=AugmentedContactOptions(maximum_augmentations=2),
    )
    solved = solve_adaptive_contact_path(
        original,
        0.5,
        path=path,
        options=options,
        _solver=solver,
    )
    assert solved.converged
    assert [attempt.action for attempt in solved.attempts] == [
        "penalty_increase",
        "cutback",
        "accepted",
        "accepted",
    ]
    parameter, penalty, state_value, prescribed, force = calls[2]
    assert parameter == pytest.approx(0.25)
    assert penalty == pytest.approx(100.0)
    assert state_value == pytest.approx(0.0)
    np.testing.assert_allclose(prescribed, 0.25 * original.constraints.values)
    np.testing.assert_allclose(force, 0.25 * original.load.force)
