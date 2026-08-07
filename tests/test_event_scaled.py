from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from contact3d.scaling import ScaleAwareConvergenceOptions
from contact3d.solvers import AugmentedContactOptions
from contact3d.solvers.events import solve_event_aware_scale_aware_augmented_contact
from contact3d.topology_events import (
    ContactTopologyEvent,
    ContactTopologyEventBatch,
    TopologyEventLocalizationOptions,
    TopologyObservation,
)


def _batch() -> ContactTopologyEventBatch:
    signature = SimpleNamespace(
        facet_pairs=((0, 0),),
        active_rows=(True,),
        supported_rows=(True,),
    )
    selected = TopologyObservation.valid(0.50000001, (signature,), None)
    event = ContactTopologyEvent(
        "pair_entry",
        0,
        (0, 0),
        0.5,
        "right",
        "scale-aware transition",
    )
    return ContactTopologyEventBatch(
        "restarted",
        0.49999999,
        0.5,
        0.50000001,
        0.50000001,
        "right",
        (event,),
        selected,
    )


def test_scale_aware_wrapper_retains_events_and_normalizes_newton(monkeypatch) -> None:
    import contact3d.solvers.events.scaling as module

    batch = _batch()
    scales = SimpleNamespace(force=250.0)
    captured: dict[str, object] = {}
    equilibrium = SimpleNamespace(
        displacement=np.zeros(3),
        converged=True,
        evaluation=SimpleNamespace(contacts=()),
        events=(batch,),
        history=(),
    )

    monkeypatch.setattr(module, "coupled_problem_scales", lambda problem: scales)
    monkeypatch.setattr(module, "_scaled_newton_history", lambda result, values: ())
    monkeypatch.setattr(module, "_all_kkt_converged", lambda *args: True)
    monkeypatch.setattr(
        module,
        "_augmentation_row",
        lambda **kwargs: SimpleNamespace(augmentation=kwargs["augmentation"]),
    )

    def solve(
        problem,
        states,
        displacement,
        *,
        load_factor,
        options,
        event_policy,
        event_options,
        tolerance,
    ):
        del problem, states, displacement, load_factor, event_policy, tolerance
        captured["absolute_tolerance"] = options.absolute_tolerance
        captured["event_options"] = event_options
        return equilibrium

    monkeypatch.setattr(module, "solve_event_aware_coupled_equilibrium", solve)
    localization = TopologyEventLocalizationOptions(fraction_tolerance=2.0e-9)
    scaling = ScaleAwareConvergenceOptions(
        enabled=True,
        equilibrium_tolerance=3.0e-8,
    )
    problem = SimpleNamespace(
        interfaces=(),
        validate_states=lambda states: (),
    )
    result = solve_event_aware_scale_aware_augmented_contact(
        problem,
        options=AugmentedContactOptions(maximum_augmentations=2),
        scaling=scaling,
        event_options=localization,
    )

    assert result.converged
    assert result.events == (batch,)
    assert len(result.history) == 1
    assert captured["absolute_tolerance"] == pytest.approx(7.5e-6)
    assert captured["event_options"] is localization
