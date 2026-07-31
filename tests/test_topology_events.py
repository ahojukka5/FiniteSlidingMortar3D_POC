from __future__ import annotations

from dataclasses import dataclass

import pytest

from contact3d.topology_events import (
    ContactTopologyStateMachine,
    TopologyEventLocalizationOptions,
    TopologyObservation,
)


@dataclass(frozen=True, slots=True)
class Signature:
    facet_pairs: tuple[tuple[int, int], ...]
    active_rows: tuple[bool, ...]
    supported_rows: tuple[bool, ...]


def signature(stage: int) -> tuple[Signature, ...]:
    return (
        Signature(
            () if stage == 0 else ((0, 0),),
            (stage >= 3, stage >= 3),
            (stage >= 2, stage >= 2),
        ),
    )


def staged_observation(fraction: float) -> TopologyObservation:
    if fraction < 0.2:
        stage = 0
    elif fraction < 0.4:
        stage = 1
    elif fraction < 0.6:
        stage = 2
    else:
        stage = 3
    return TopologyObservation.valid(fraction, signature(stage), fraction)


def test_signature_transitions_are_localized_and_ordered() -> None:
    machine = ContactTopologyStateMachine()
    batch = machine.localize(
        staged_observation(0.0),
        staged_observation(0.3),
        staged_observation,
    )
    assert batch.selected_branch == "right"
    assert batch.selected_fraction == pytest.approx(0.2, abs=2.0e-10)
    assert [event.kind for event in batch.events] == ["pair_entry"]
    assert batch.events[0].entity == (0, 0)


def test_event_location_is_invariant_under_segment_subdivision() -> None:
    machine = ContactTopologyStateMachine()
    coarse = machine.localize(
        staged_observation(0.3),
        staged_observation(0.5),
        staged_observation,
    )
    fine = machine.localize(
        staged_observation(0.35),
        staged_observation(0.45),
        staged_observation,
    )
    assert coarse.event_fraction == pytest.approx(fine.event_fraction, abs=2.0e-10)
    assert [event.kind for event in coarse.events] == [
        "support_activation",
        "support_activation",
    ]


def test_recoverable_singularity_selects_first_valid_right_branch() -> None:
    left = Signature((), (False,), (False,))
    right = Signature(((0, 0),), (False,), (True,))

    def observe(fraction: float) -> TopologyObservation:
        if 0.48 <= fraction <= 0.52:
            return TopologyObservation.recoverable(
                fraction,
                "clipping_vertex_edge",
                "synthetic edge-on-edge interval",
            )
        branch = left if fraction < 0.48 else right
        return TopologyObservation.valid(fraction, (branch,), fraction)

    machine = ContactTopologyStateMachine()
    batch = machine.localize(observe(0.0), observe(1.0), observe)
    assert batch.left_fraction == pytest.approx(0.48, abs=2.0e-10)
    assert batch.right_fraction == pytest.approx(0.52, abs=2.0e-10)
    assert batch.selected_fraction == pytest.approx(0.52, abs=2.0e-10)
    assert {event.kind for event in batch.events} == {
        "clipping_vertex_edge",
        "pair_entry",
        "support_activation",
    }


def test_left_branch_selection_is_explicit() -> None:
    machine = ContactTopologyStateMachine(
        TopologyEventLocalizationOptions(branch_selection="left")
    )
    batch = machine.localize(
        staged_observation(0.0),
        staged_observation(0.3),
        staged_observation,
    )
    assert batch.selected_branch == "left"
    assert batch.selected_fraction < 0.2
    assert batch.selected.payload == batch.selected_fraction


def test_geometry_tokens_classify_clipping_and_pallet_transitions() -> None:
    @dataclass(frozen=True, slots=True)
    class GeometrySignature:
        facet_pairs: tuple[tuple[int, int], ...]
        active_rows: tuple[bool, ...]
        supported_rows: tuple[bool, ...]
        geometry_tokens: tuple[tuple[int, int, int, int, int], ...]

    left = GeometrySignature(((0, 1),), (False,), (True,), ((0, 1, 4, 4, 1),))
    right = GeometrySignature(((0, 1),), (False,), (True,), ((0, 1, 5, 5, 1),))

    def observe(fraction: float) -> TopologyObservation:
        branch = left if fraction < 0.35 else right
        return TopologyObservation.valid(fraction, (branch,), fraction)

    batch = ContactTopologyStateMachine().localize(observe(0.0), observe(1.0), observe)
    assert [event.kind for event in batch.events] == [
        "clipping_vertex_edge",
        "pallet_transition",
    ]
    assert batch.events[0].entity == (0, 1)
