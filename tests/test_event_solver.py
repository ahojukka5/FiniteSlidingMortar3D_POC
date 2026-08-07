from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from contact3d.clipping import ClippingTopologyError
from contact3d.solvers.events.newton import solve_event_aware_coupled_equilibrium
from contact3d.sparse import CSRMatrix


@dataclass(frozen=True, slots=True)
class Signature:
    facet_pairs: tuple[tuple[int, int], ...]
    active_rows: tuple[bool, ...]
    supported_rows: tuple[bool, ...]


class Constraints:
    def apply(self, values):
        return np.asarray(values, dtype=float).copy()


class Problem:
    mesh = SimpleNamespace(node_count=1)
    constraints = Constraints()
    interfaces = (object(),)

    def initial_states(self):
        return (object(),)

    def validate_states(self, states):
        return self.initial_states() if states is None else tuple(states)


def fake_evaluation(value: float, *, tangent: bool = True):
    branch = Signature((), (False,), (False,))
    if value > 0.52:
        branch = Signature(((0, 0),), (True,), (True,))
    if 0.48 <= value <= 0.52:
        raise ClippingTopologyError("synthetic clipping coincidence")
    residual = np.array([value - 1.0, 0.0, 0.0])
    matrix = CSRMatrix(
        (1, 1),
        np.array([0, 1], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.ones(1),
    )
    return SimpleNamespace(
        displacement=np.array([value, 0.0, 0.0]),
        signatures=(branch,),
        contacts=(SimpleNamespace(signature=branch),),
        residual=residual,
        free_dofs=np.array([0]),
        tangent=matrix if tangent else None,
        free_residual_norm=abs(value - 1.0),
        bulk_potential=0.5 * (value - 1.0) ** 2,
        bulk=SimpleNamespace(minimum_jacobian=1.0),
        maximum_penetration=max(value - 0.52, 0.0),
    )


def test_event_aware_newton_restarts_on_right_branch(monkeypatch) -> None:
    import contact3d.event_geometry as geometry_module
    import contact3d.solvers.events.newton as newton_module

    def evaluate(problem, displacement, states, *, assemble_tangent=True, **kwargs):
        del problem, states, kwargs
        return fake_evaluation(float(np.asarray(displacement)[0]), tangent=assemble_tangent)

    monkeypatch.setattr(geometry_module, "evaluate_coupled_equilibrium", evaluate)
    monkeypatch.setattr(newton_module, "evaluate_coupled_equilibrium", evaluate)
    result = solve_event_aware_coupled_equilibrium(
        Problem(),
        (object(),),
        np.zeros(3),
    )
    assert result.converged
    assert result.displacement[0] == pytest.approx(1.0)
    assert result.contact_event_restarts == 1
    assert result.events[0].selected_fraction == pytest.approx(0.52, abs=2.0e-10)
    assert {event.kind for event in result.events[0].events} == {
        "clipping_vertex_edge",
        "pair_entry",
        "support_activation",
        "pressure_activation",
    }
    assert any(iteration.contact_branch_changed for iteration in result.history)
    assert all(row.linear_solve.backend == "dense" for row in result.history)
    assert all(row.linear_solve.materialized_dense for row in result.history)


def test_event_restart_clears_tangent_only_singularity(monkeypatch) -> None:
    import contact3d.event_geometry as geometry_module
    import contact3d.solvers.events.newton as newton_module

    def evaluate(problem, displacement, states, *, assemble_tangent=True, **kwargs):
        del problem, states, kwargs
        value = float(np.asarray(displacement)[0])
        if assemble_tangent and 0.52 < value <= 0.54:
            raise ClippingTopologyError("synthetic tangent-only coincidence")
        return fake_evaluation(value, tangent=assemble_tangent)

    monkeypatch.setattr(geometry_module, "evaluate_coupled_equilibrium", evaluate)
    monkeypatch.setattr(newton_module, "evaluate_coupled_equilibrium", evaluate)
    result = solve_event_aware_coupled_equilibrium(
        Problem(),
        (object(),),
        np.zeros(3),
    )

    assert result.converged
    assert result.contact_event_restarts == 1
    event_fraction = result.events[0].selected_fraction
    restart = next(row for row in result.history if row.contact_branch_changed)
    assert event_fraction == pytest.approx(0.52, abs=2.0e-10)
    assert restart.accepted_step > 0.54
    assert result.displacement[0] == pytest.approx(1.0)


def test_production_geometry_signature_records_polygon_and_pallet_counts(monkeypatch) -> None:
    import contact3d.event_geometry as module

    slave_surface = SimpleNamespace(
        reference_nodes=np.zeros((4, 3)),
        facets=(np.array([0, 1, 2, 3]),),
    )
    master_surface = SimpleNamespace(
        reference_nodes=np.zeros((3, 3)),
        facets=(np.array([0, 1, 2]),),
    )
    interface = SimpleNamespace(
        pair=SimpleNamespace(slave=slave_surface, master=master_surface),
        slave_nodes=np.array([0, 1, 2, 3]),
        master_nodes=np.array([4, 5, 6]),
    )
    branch = Signature(((0, 0),), (False, False, False, False), (True,) * 4)
    contact = SimpleNamespace(signature=branch)
    evaluation = SimpleNamespace(
        displacement=np.zeros(21),
        contacts=(contact,),
    )
    problem = SimpleNamespace(interfaces=(interface,))
    polygon = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.1, 0.5], [0.5, 1.0], [0.0, 0.5]]
    )
    monkeypatch.setattr(
        module,
        "build_facet_overlap",
        lambda *args, **kwargs: SimpleNamespace(
            intersection_polygon=polygon,
            pallets=tuple(range(5)),
        ),
    )
    monkeypatch.setattr(module, "polygon_signed_area", lambda values: 1.25)

    signatures = module._event_signatures(problem, evaluation, tolerance=1.0e-12)
    assert signatures[0].geometry_tokens == ((0, 0, 5, 5, 1),)
