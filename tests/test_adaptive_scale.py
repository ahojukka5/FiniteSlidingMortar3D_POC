from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from contact3d.bulk_material import NeoHookeanMaterial
from contact3d.enforcement_state import AugmentedLagrangeState, KKTDiagnostics
from contact3d.scaling import ScaleAwareConvergenceOptions
from contact3d.solvers import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    contact_penalties,
    solve_adaptive_contact_path,
)


@dataclass(frozen=True, slots=True)
class Mesh:
    reference_nodes: np.ndarray

    @property
    def node_count(self):
        return len(self.reference_nodes)


@dataclass(frozen=True, slots=True)
class Interface:
    penalty: float
    areas: np.ndarray

    @property
    def normal_penalty(self):
        return self.penalty

    def with_normal_penalty(self, normal_penalty):
        return replace(self, penalty=normal_penalty)

    def reference_tributary_areas(self):
        return self.areas.copy()

    def initial_state(self):
        return AugmentedLagrangeState.zeros(1)


@dataclass(frozen=True, slots=True)
class Problem:
    mesh: Mesh
    material: NeoHookeanMaterial
    interfaces: tuple[Interface, ...]
    sparsity: object

    def initial_states(self):
        return tuple(interface.initial_state() for interface in self.interfaces)


@dataclass(frozen=True, slots=True)
class Signature:
    active_rows: tuple[bool, ...] = (True,)
    supported_rows: tuple[bool, ...] = (True,)


@dataclass(frozen=True, slots=True)
class Contact:
    diagnostics: KKTDiagnostics
    pressure: np.ndarray
    signature: Signature = Signature()


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


def contact(penetration: float) -> Contact:
    return Contact(
        KKTDiagnostics(
            np.array([penetration]),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
        ),
        np.array([1.0]),
    )


def result(problem, load_factor, converged):
    contacts = (contact(1e-10), contact(2e-4 if not converged else 1e-10))
    evaluation = SimpleNamespace(
        contacts=contacts,
        maximum_penetration=max(
            value.diagnostics.maximum_penetration for value in contacts
        ),
        free_residual_norm=1e-7 if not converged else 1e-12,
        residual=np.zeros(6),
        free_dofs=np.arange(6),
    )
    displacement = np.full(6, load_factor)
    equilibrium = Equilibrium(displacement, 3, 0, evaluation)
    return Result(
        displacement,
        tuple(
            AugmentedLagrangeState(np.array([load_factor]))
            for _ in problem.interfaces
        ),
        converged,
        "converged" if converged else "maximum_augmentations",
        equilibrium,
        (equilibrium,),
        (object(),),
    )


def test_interface_local_retry_and_histories_are_transactional():
    calls = []
    problem = Problem(
        Mesh(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])),
        NeoHookeanMaterial.from_young_poisson(210.0, 0.3),
        (
            Interface(100.0, np.array([0.25])),
            Interface(100.0, np.array([0.01])),
        ),
        object(),
    )
    original_sparsity = problem.sparsity

    def solver(problem, displacement, states, *, load_factor, options, tolerance):
        penalties = contact_penalties(problem)
        calls.append(penalties)
        return result(problem, load_factor, converged=penalties[1] > 100.0)

    options = AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=1.0,
            minimum_step=0.1,
            maximum_step=1.0,
        ),
        penalty=AdaptivePenaltyOptions(
            increase_factor=4.0,
            normalized_penetration_target=1e-6,
            maximum_scale_factor=100.0,
        ),
        scaling=ScaleAwareConvergenceOptions(enabled=True, gap_tolerance=1e-6),
    )
    solved = solve_adaptive_contact_path(problem, 1.0, options=options, _solver=solver)
    assert solved.converged
    assert calls[0] == (100.0, 100.0)
    assert calls[1][0] == 100.0
    assert calls[1][1] > 100.0
    assert contact_penalties(solved.problem)[0] == 100.0
    assert solved.problem.sparsity is original_sparsity
    assert [attempt.action for attempt in solved.attempts] == [
        "penalty_increase",
        "accepted",
    ]
    retry = solved.attempts[0]
    assert retry.normalized_interface_penetrations[0] < 1e-6
    assert retry.normalized_interface_penetrations[1] > 1e-6
    assert retry.penalty_ratios_after[0] == retry.penalty_ratios_before[0]
    assert retry.penalty_ratios_after[1] > retry.penalty_ratios_before[1]
    assert "interface[1]" in retry.penalty_update_reasons[0]
