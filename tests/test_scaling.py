from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from contact3d.bulk_material import NeoHookeanMaterial
from contact3d.enforcement_state import KKTDiagnostics
from contact3d.scaling import (
    ScaleAwareConvergenceOptions,
    contact_interface_scales,
    coupled_problem_scales,
    interface_normal_penalty,
    propose_interface_penalties,
    with_interface_normal_penalty,
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
    def normal_penalty(self) -> float:
        return self.penalty

    def with_normal_penalty(self, normal_penalty: float) -> Interface:
        return replace(self, penalty=normal_penalty)

    def reference_tributary_areas(self) -> np.ndarray:
        return self.areas.copy()


@dataclass(frozen=True, slots=True)
class Problem:
    mesh: Mesh
    material: NeoHookeanMaterial
    interfaces: tuple[Interface, ...]


@dataclass(frozen=True, slots=True)
class Signature:
    active_rows: tuple[bool, ...]
    supported_rows: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class Contact:
    diagnostics: KKTDiagnostics
    pressure: np.ndarray
    signature: Signature


def material(young: float) -> NeoHookeanMaterial:
    return NeoHookeanMaterial.from_young_poisson(young, 0.3)


def diagnostics(length_factor: float = 1.0, pressure_factor: float = 1.0):
    return KKTDiagnostics(
        penetration=np.array([2.0e-5 * length_factor]),
        multiplier_violation=np.array([3.0e-4 * pressure_factor]),
        complementarity=np.array([4.0e-9 * pressure_factor * length_factor]),
        projection_residual=np.array([5.0e-4 * pressure_factor]),
        unsupported_multiplier=np.array([6.0e-4 * pressure_factor]),
    )


def test_normalized_scales_are_invariant_under_unit_conversion() -> None:
    length_factor = 1000.0
    pressure_factor = 1.0e-6
    force_factor = pressure_factor * length_factor**2
    original = Problem(
        Mesh(np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])),
        material(200.0),
        (Interface(800.0, np.array([0.25, 0.75])),),
    )
    converted = Problem(
        Mesh(length_factor * original.mesh.reference_nodes),
        material(pressure_factor * 200.0),
        (
            Interface(
                pressure_factor / length_factor * 800.0,
                length_factor**2 * np.array([0.25, 0.75]),
            ),
        ),
    )
    first = coupled_problem_scales(original)
    second = coupled_problem_scales(converted)
    assert second.length == pytest.approx(length_factor * first.length)
    assert second.pressure == pytest.approx(pressure_factor * first.pressure)
    assert second.force == pytest.approx(force_factor * first.force)
    assert second.energy == pytest.approx(force_factor * length_factor * first.energy)
    assert second.interfaces[0].penalty == pytest.approx(
        pressure_factor / length_factor * first.interfaces[0].penalty
    )

    first_kkt = first.interfaces[0].normalize_kkt(diagnostics())
    second_kkt = second.interfaces[0].normalize_kkt(
        diagnostics(length_factor, pressure_factor)
    )
    for name in first_kkt.__dataclass_fields__:
        assert getattr(second_kkt, name) == pytest.approx(getattr(first_kkt, name))
    residual = 17.0
    assert second.normalized_equilibrium_residual(force_factor * residual) == pytest.approx(
        first.normalized_equilibrium_residual(residual)
    )


def test_only_under_resolved_interface_receives_bounded_penalty_update() -> None:
    problem = Problem(
        Mesh(np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])),
        material(210.0),
        (
            Interface(100.0, np.array([0.25, 0.25])),
            Interface(100.0, np.array([0.01, 0.01])),
        ),
    )
    contacts = (
        Contact(
            KKTDiagnostics(
                np.array([1.0e-10]),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
            ),
            np.array([0.0]),
            Signature((False,), (True,)),
        ),
        Contact(
            KKTDiagnostics(
                np.array([2.0e-4]),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
            ),
            np.array([1.0]),
            Signature((True,), (True,)),
        ),
    )
    plan = propose_interface_penalties(
        problem,
        contacts,
        increase_factor=4.0,
        absolute_maximum=1.0e9,
        minimum_scale_factor=0.25,
        maximum_scale_factor=100.0,
        dimensional_target=None,
        normalized_target=1.0e-6,
        use_normalized_target=True,
        interface_local=True,
    )
    assert plan.penalties[0] == 100.0
    assert plan.penalties[1] > 100.0
    assert [decision.interface for decision in plan.decisions] == [1]
    assert "normalized_penetration" in plan.reasons[0]

    scale = contact_interface_scales(problem.interfaces[1], problem.material)
    _, upper = scale.penalty_bounds(
        minimum_factor=0.25,
        maximum_factor=100.0,
        absolute_maximum=1.0e9,
    )
    assert plan.penalties[1] <= upper


def test_penalty_protocol_replaces_without_mutating_original() -> None:
    original = Interface(125.0, np.array([0.5]))
    changed = with_interface_normal_penalty(original, 500.0)
    assert interface_normal_penalty(original) == 125.0
    assert interface_normal_penalty(changed) == 500.0


def test_legacy_attribute_shape_is_not_accepted_as_a_contract() -> None:
    legacy = SimpleNamespace(pair=SimpleNamespace(normal_penalty=100.0))
    with pytest.raises(TypeError, match="PenaltyControlledContactInterface"):
        interface_normal_penalty(legacy)


def test_scale_option_validation() -> None:
    with pytest.raises(ValueError):
        ScaleAwareConvergenceOptions(equilibrium_tolerance=-1.0)


def test_scale_aware_augmented_solver_converts_newton_and_kkt_tolerances(monkeypatch) -> None:
    import contact3d.scaled_solver as module
    from contact3d.coupled import AugmentedContactOptions
    from contact3d.equilibrium import NewtonOptions
    from contact3d.scaled_solver import solve_scale_aware_augmented_contact

    @dataclass(frozen=True, slots=True)
    class SolverInterface(Interface):
        def initial_state(self):
            from contact3d.enforcement_state import AugmentedLagrangeState

            return AugmentedLagrangeState.zeros(1)

        def augment(self, contact, *, tolerance):
            raise AssertionError("already converged KKT state must not augment")

    @dataclass(frozen=True, slots=True)
    class SolverProblem(Problem):
        def initial_states(self):
            return tuple(interface.initial_state() for interface in self.interfaces)

    problem = SolverProblem(
        Mesh(np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])),
        material(200.0),
        (SolverInterface(800.0, np.array([0.25, 0.75])),),
    )
    scales = coupled_problem_scales(problem)
    observed = {}

    row = SimpleNamespace(
        iteration=1,
        residual_norm=2.0e-7 * scales.force,
        relative_residual=2.0e-7,
        bulk_potential=0.25 * scales.energy,
        minimum_jacobian=0.9,
        maximum_penetration=1.0e-9 * scales.interfaces[0].length,
        step_norm=0.1 * scales.length,
        accepted_step=1.0,
        line_search_iterations=0,
        contact_branch_changed=False,
    )
    scaled_diagnostics = KKTDiagnostics(
        penetration=np.array([1.0e-9 * scales.interfaces[0].length]),
        multiplier_violation=np.array([1.0e-9 * scales.interfaces[0].pressure]),
        complementarity=np.array(
            [1.0e-9 * scales.interfaces[0].pressure * scales.interfaces[0].length]
        ),
        projection_residual=np.array([1.0e-9 * scales.interfaces[0].pressure]),
        unsupported_multiplier=np.zeros(1),
    )
    contact_value = Contact(
        scaled_diagnostics,
        np.array([0.2 * scales.interfaces[0].pressure]),
        Signature((True,), (True,)),
    )

    def fake_solve(problem, states, displacement, *, load_factor, options, event_policy, tolerance):
        observed["absolute_tolerance"] = options.absolute_tolerance
        evaluation = SimpleNamespace(
            contacts=(contact_value,),
            free_residual_norm=2.0e-7 * scales.force,
        )
        return SimpleNamespace(
            displacement=np.zeros(6),
            converged=True,
            evaluation=evaluation,
            history=(row,),
            iteration_count=1,
            contact_event_restarts=0,
        )

    monkeypatch.setattr(module, "solve_coupled_equilibrium", fake_solve)
    options = AugmentedContactOptions(
        maximum_augmentations=2,
        newton=NewtonOptions(absolute_tolerance=1.0),
    )
    scaling = ScaleAwareConvergenceOptions(
        enabled=True,
        equilibrium_tolerance=3.0e-8,
        gap_tolerance=1.0e-8,
        complementarity_tolerance=1.0e-8,
        projection_tolerance=1.0e-8,
        multiplier_tolerance=1.0e-8,
    )
    solved = solve_scale_aware_augmented_contact(problem, options=options, scaling=scaling)
    assert solved.converged
    assert observed["absolute_tolerance"] == pytest.approx(3.0e-8 * scales.force)
    assert solved.history[0].normalized_equilibrium_residual == pytest.approx(2.0e-7)
    assert solved.history[0].normalized_maximum_penetration == pytest.approx(1.0e-9)
    assert solved.newton_histories[0][0].normalized_residual == pytest.approx(2.0e-7)


def test_production_mortar_adapter_uses_reference_tributary_area() -> None:
    from contact3d.coupled import MortarContactInterface
    from contact3d.scaling import interface_reference_tributary_areas

    @dataclass(frozen=True, slots=True)
    class Surface:
        reference_nodes: np.ndarray
        facets: tuple[np.ndarray, ...]

    @dataclass(frozen=True, slots=True)
    class Pair:
        normal_penalty: float
        slave: Surface

    surface = Surface(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        (np.array([0, 1, 2, 3]),),
    )
    interface = MortarContactInterface(
        Pair(100.0, surface),
        np.arange(4),
        np.arange(4, 8),
    )
    np.testing.assert_allclose(interface_reference_tributary_areas(interface), 0.25)
    changed = with_interface_normal_penalty(interface, 250.0)
    assert interface_normal_penalty(interface) == 100.0
    assert interface_normal_penalty(changed) == 250.0
