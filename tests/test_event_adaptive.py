from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from contact3d.adaptive import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    ScaleAwareConvergenceOptions,
)
from contact3d.enforcement_state import AugmentedLagrangeState
from contact3d.event_solver import solve_event_aware_adaptive_contact_path
from contact3d.load_path import CoupledPathState
from contact3d.topology_events import (
    ContactTopologyEvent,
    ContactTopologyEventBatch,
    TopologyEventLocalizationOptions,
    TopologyObservation,
)


@dataclass(frozen=True, slots=True)
class Mesh:
    node_count: int = 2


@dataclass(frozen=True, slots=True)
class Pair:
    normal_penalty: float = 100.0


@dataclass(frozen=True, slots=True)
class Interface:
    pair: Pair = Pair()

    @property
    def normal_penalty(self) -> float:
        return self.pair.normal_penalty

    def with_normal_penalty(self, normal_penalty: float) -> Interface:
        return replace(self, pair=replace(self.pair, normal_penalty=normal_penalty))

    def reference_tributary_areas(self) -> np.ndarray:
        return np.ones(1)

    def initial_state(self) -> AugmentedLagrangeState:
        return AugmentedLagrangeState.zeros(1)


@dataclass(frozen=True, slots=True)
class Problem:
    mesh: Mesh = Mesh()
    interfaces: tuple[Interface, ...] = (Interface(),)

    def initial_states(self) -> tuple[AugmentedLagrangeState, ...]:
        return tuple(interface.initial_state() for interface in self.interfaces)


@dataclass(frozen=True, slots=True)
class Signature:
    facet_pairs: tuple[tuple[int, int], ...] = ((0, 0),)
    active_rows: tuple[bool, ...] = (True,)
    supported_rows: tuple[bool, ...] = (True,)


def event_batch(fraction: float) -> ContactTopologyEventBatch:
    selected_fraction = min(1.0, fraction + 1.0e-8)
    selected = TopologyObservation.valid(selected_fraction, (Signature(),), selected_fraction)
    event = ContactTopologyEvent(
        "pair_entry",
        0,
        (0, 0),
        fraction,
        "right",
        "synthetic adaptive transition",
    )
    return ContactTopologyEventBatch(
        "restarted",
        max(0.0, fraction - 1.0e-8),
        fraction,
        selected_fraction,
        selected_fraction,
        "right",
        (event,),
        selected,
    )


def result(load_factor: float, *, converged: bool) -> object:
    displacement = np.full(6, load_factor)
    state = AugmentedLagrangeState(np.array([load_factor]))
    evaluation = SimpleNamespace(
        maximum_penetration=0.0,
        free_residual_norm=1.0e-12 if converged else 1.0e-3,
    )
    equilibrium = SimpleNamespace(
        displacement=displacement,
        load_factor=load_factor,
        iteration_count=2,
        contact_event_restarts=1,
        evaluation=evaluation,
        events=(event_batch(0.4),),
    )
    return SimpleNamespace(
        displacement=displacement,
        states=(state,),
        converged=converged,
        termination_reason="converged" if converged else "inner_equilibrium_failed",
        equilibrium=equilibrium,
        equilibria=(equilibrium,),
        history=(object(),),
    )


def options(*, scaling: bool = False) -> AdaptiveContactOptions:
    return AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.75,
            minimum_step=0.1,
            maximum_step=0.75,
            cutback_factor=0.5,
            easy_newton_iterations=0,
        ),
        penalty=AdaptivePenaltyOptions(enabled=False),
        scaling=ScaleAwareConvergenceOptions(enabled=scaling),
    )


def test_adaptive_events_include_cutback_and_absolute_parameters() -> None:
    calls: list[float] = []

    def solver(problem, displacement, states, *, load_factor, options, tolerance):
        del problem, displacement, states, options, tolerance
        calls.append(load_factor)
        return result(load_factor, converged=not (len(calls) == 1))

    solved = solve_event_aware_adaptive_contact_path(
        Problem(),
        1.0,
        options=options(),
        _solver=solver,
    )

    assert solved.converged
    assert [attempt.action for attempt in solved.attempts] == [
        "cutback",
        "accepted",
        "accepted",
        "accepted",
    ]
    assert [record.continuation_parameter for record in solved.event_batches] == pytest.approx(
        [0.75, 0.375, 0.75, 1.0]
    )
    assert [record.action for record in solved.event_batches] == [
        "cutback",
        "accepted",
        "accepted",
        "accepted",
    ]
    assert solved.contact_event_restarts == 4
    rows = solved.event_rows()
    assert [row["event_newton_fraction"] for row in rows] == pytest.approx([0.4] * 4)
    assert rows[0]["kind"] == "pair_entry"


@dataclass(frozen=True, slots=True)
class MixedPath:
    def evaluate(self, problem: Problem, parameter: float) -> CoupledPathState:
        return CoupledPathState(
            parameter,
            problem,
            1.0,
            np.empty(0, dtype=np.int64),
            np.empty(0),
            np.empty(0),
            (("tool_z", -0.2 * parameter),),
        )


def test_mixed_path_records_parameter_separately_from_solver_load() -> None:
    def solver(problem, displacement, states, *, load_factor, options, tolerance):
        del problem, displacement, states, options, tolerance
        return result(load_factor, converged=True)

    settings = replace(
        options(),
        load=AdaptiveLoadOptions(
            initial_step=0.5,
            minimum_step=0.1,
            maximum_step=0.5,
            easy_newton_iterations=0,
        ),
    )
    solved = solve_event_aware_adaptive_contact_path(
        Problem(),
        1.0,
        path=MixedPath(),
        options=settings,
        _solver=solver,
    )

    assert [record.continuation_parameter for record in solved.event_batches] == [0.5, 1.0]
    assert [record.solver_load_factor for record in solved.event_batches] == [1.0, 1.0]
    assert solved.event_rows()[0]["path_values"] == "tool_z=-0.10000000000000001"


def test_scaled_adaptive_path_uses_localized_scale_aware_solver(monkeypatch) -> None:
    import contact3d.event_adaptive as module

    calls: list[tuple[object, object]] = []
    localization = TopologyEventLocalizationOptions(fraction_tolerance=1.0e-8)

    def scaled_solver(
        problem,
        displacement,
        states,
        *,
        load_factor,
        options,
        scaling,
        event_options,
        tolerance,
    ):
        del problem, displacement, states, options, tolerance
        calls.append((scaling, event_options))
        return result(load_factor, converged=True)

    monkeypatch.setattr(
        module,
        "solve_event_aware_scale_aware_augmented_contact",
        scaled_solver,
    )
    settings = replace(
        options(scaling=True),
        load=AdaptiveLoadOptions(
            initial_step=1.0,
            minimum_step=0.1,
            maximum_step=1.0,
            easy_newton_iterations=0,
        ),
    )
    solved = solve_event_aware_adaptive_contact_path(
        Problem(),
        1.0,
        options=settings,
        event_options=localization,
    )

    assert solved.converged
    assert calls == [(settings.scaling, localization)]
