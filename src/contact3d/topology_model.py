"""Typed data models for contact-topology event localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np

EventKind = Literal[
    "pair_entry",
    "pair_exit",
    "support_activation",
    "support_release",
    "pressure_activation",
    "pressure_release",
    "clipping_vertex_edge",
    "pallet_transition",
    "inverse_map_boundary",
]
BranchSelection = Literal["left", "right"]
MachineState = Literal["smooth", "bracketed", "localized", "restarted"]


class BranchSignature(Protocol):
    """Discrete interface branch required by the topology state machine."""

    facet_pairs: tuple[tuple[int, int], ...]
    active_rows: tuple[bool, ...]
    supported_rows: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class ContactTopologySignature:
    """Discrete branch plus projected-overlap topology for one interface."""

    facet_pairs: tuple[tuple[int, int], ...]
    active_rows: tuple[bool, ...]
    supported_rows: tuple[bool, ...]
    geometry_tokens: tuple[tuple[int, int, int, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class TopologyEventLocalizationOptions:
    """Deterministic controls for one-dimensional topology localization."""

    fraction_tolerance: float = 1.0e-10
    maximum_iterations: int = 80
    branch_selection: BranchSelection = "right"

    def __post_init__(self) -> None:
        if not np.isfinite(self.fraction_tolerance) or self.fraction_tolerance <= 0.0:
            raise ValueError("fraction_tolerance must be finite and positive")
        if self.maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be positive")
        if self.branch_selection not in ("left", "right"):
            raise ValueError("branch_selection must be 'left' or 'right'")


@dataclass(frozen=True, slots=True)
class TopologyObservation:
    """One valid branch sample or one recoverable singular event sample."""

    fraction: float
    signatures: tuple[BranchSignature, ...] | None
    payload: Any = None
    recoverable_kind: EventKind | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        value = float(self.fraction)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("observation fraction must lie in [0, 1]")
        valid = self.signatures is not None
        if valid == (self.recoverable_kind is not None):
            raise ValueError(
                "observation must be either a valid branch or one recoverable event"
            )
        object.__setattr__(self, "fraction", value)
        if self.signatures is not None:
            object.__setattr__(self, "signatures", tuple(self.signatures))

    @classmethod
    def valid(
        cls,
        fraction: float,
        signatures: tuple[BranchSignature, ...],
        payload: Any = None,
    ) -> TopologyObservation:
        return cls(fraction, tuple(signatures), payload)

    @classmethod
    def recoverable(
        cls,
        fraction: float,
        kind: EventKind,
        detail: str,
    ) -> TopologyObservation:
        return cls(fraction, None, None, kind, detail)

    @property
    def is_valid(self) -> bool:
        return self.signatures is not None


@dataclass(frozen=True, slots=True)
class ContactTopologyEvent:
    """One atomic interface transition at a localized segment fraction."""

    kind: EventKind
    interface: int
    entity: tuple[int, ...]
    fraction: float
    selected_branch: BranchSelection
    detail: str


@dataclass(frozen=True, slots=True)
class ContactTopologyEventBatch:
    """All atomic transitions crossed at one localized segment position."""

    state: MachineState
    left_fraction: float
    event_fraction: float
    right_fraction: float
    selected_fraction: float
    selected_branch: BranchSelection
    events: tuple[ContactTopologyEvent, ...]
    selected: TopologyObservation

    def __post_init__(self) -> None:
        if self.state not in ("localized", "restarted"):
            raise ValueError("an event batch must be localized or restarted")
        if not self.left_fraction <= self.event_fraction <= self.right_fraction:
            raise ValueError("event fraction must lie inside the final bracket")
        if not self.selected.is_valid:
            raise ValueError("selected event branch must be a valid observation")
        if not self.events:
            raise ValueError("event batch must contain at least one event")

    def restarted(self) -> ContactTopologyEventBatch:
        return ContactTopologyEventBatch(
            "restarted",
            self.left_fraction,
            self.event_fraction,
            self.right_fraction,
            self.selected_fraction,
            self.selected_branch,
            self.events,
            self.selected,
        )
