"""Deterministic decomposition of contact branch transitions."""

from __future__ import annotations

from .topology_model import (
    BranchSelection,
    BranchSignature,
    ContactTopologyEvent,
    EventKind,
    TopologyObservation,
)

_KIND_ORDER: dict[EventKind, int] = {
    "clipping_vertex_edge": 0,
    "pallet_transition": 1,
    "inverse_map_boundary": 2,
    "pair_exit": 3,
    "pair_entry": 4,
    "support_release": 5,
    "support_activation": 6,
    "pressure_release": 7,
    "pressure_activation": 8,
}


def signature_events(
    left: tuple[BranchSignature, ...],
    right: tuple[BranchSignature, ...],
    *,
    fraction: float,
    branch: BranchSelection,
) -> tuple[ContactTopologyEvent, ...]:
    """Return a stable sequence of atomic changes between two valid branches."""

    if len(left) != len(right):
        raise ValueError("left and right observations must contain equal interfaces")
    events: list[ContactTopologyEvent] = []
    for interface, (old, new) in enumerate(zip(left, right, strict=True)):
        old_pairs = set(old.facet_pairs)
        new_pairs = set(new.facet_pairs)
        for pair in sorted(old_pairs - new_pairs):
            events.append(
                ContactTopologyEvent(
                    "pair_exit", interface, tuple(pair), fraction, branch,
                    "facet pair removed",
                )
            )
        for pair in sorted(new_pairs - old_pairs):
            events.append(
                ContactTopologyEvent(
                    "pair_entry", interface, tuple(pair), fraction, branch,
                    "facet pair added",
                )
            )
        old_geometry = {
            (token[0], token[1]): token[2:]
            for token in getattr(old, "geometry_tokens", ())
        }
        new_geometry = {
            (token[0], token[1]): token[2:]
            for token in getattr(new, "geometry_tokens", ())
        }
        for pair in sorted(set(old_geometry) & set(new_geometry)):
            old_vertices, old_pallets, old_orientation = old_geometry[pair]
            new_vertices, new_pallets, new_orientation = new_geometry[pair]
            if old_vertices != new_vertices or old_orientation != new_orientation:
                events.append(
                    ContactTopologyEvent(
                        "clipping_vertex_edge",
                        interface,
                        tuple(pair),
                        fraction,
                        branch,
                        "intersection polygon topology changed",
                    )
                )
            if old_pallets != new_pallets:
                events.append(
                    ContactTopologyEvent(
                        "pallet_transition",
                        interface,
                        tuple(pair),
                        fraction,
                        branch,
                        "centroid-fan pallet count changed",
                    )
                )
        if len(old.supported_rows) != len(new.supported_rows):
            raise ValueError("support signatures must have equal row counts")
        if len(old.active_rows) != len(new.active_rows):
            raise ValueError("active signatures must have equal row counts")
        for row, (before, after) in enumerate(
            zip(old.supported_rows, new.supported_rows, strict=True)
        ):
            if before and not after:
                kind: EventKind = "support_release"
            elif not before and after:
                kind = "support_activation"
            else:
                continue
            events.append(
                ContactTopologyEvent(
                    kind,
                    interface,
                    (row,),
                    fraction,
                    branch,
                    "mortar-row support changed",
                )
            )
        for row, (before, after) in enumerate(
            zip(old.active_rows, new.active_rows, strict=True)
        ):
            if before and not after:
                kind = "pressure_release"
            elif not before and after:
                kind = "pressure_activation"
            else:
                continue
            events.append(
                ContactTopologyEvent(
                    kind,
                    interface,
                    (row,),
                    fraction,
                    branch,
                    "unilateral pressure branch changed",
                )
            )
    return tuple(
        sorted(
            events,
            key=lambda item: (
                item.interface,
                _KIND_ORDER[item.kind],
                item.entity,
            ),
        )
    )


def same_branch(
    observation: TopologyObservation,
    signatures: tuple[BranchSignature, ...],
) -> bool:
    """Return whether a valid observation remains on the supplied branch."""

    return observation.is_valid and observation.signatures == signatures
