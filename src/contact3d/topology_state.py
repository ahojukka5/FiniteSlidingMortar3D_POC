"""Deterministic localization of contact-topology transitions."""

from __future__ import annotations

from collections.abc import Callable

from .topology_diff import _KIND_ORDER, same_branch, signature_events
from .topology_model import (
    ContactTopologyEvent,
    ContactTopologyEventBatch,
    EventKind,
    TopologyEventLocalizationOptions,
    TopologyObservation,
)


class ContactTopologyStateMachine:
    """Localize the first discrete contact event on a smooth trial segment."""

    def __init__(
        self,
        options: TopologyEventLocalizationOptions | None = None,
    ) -> None:
        self.options = TopologyEventLocalizationOptions() if options is None else options

    def localize(
        self,
        left: TopologyObservation,
        right: TopologyObservation,
        observe: Callable[[float], TopologyObservation],
    ) -> ContactTopologyEventBatch:
        if not left.is_valid or not right.is_valid:
            raise ValueError("localization endpoints must be valid observations")
        if left.signatures == right.signatures:
            raise ValueError("localization endpoints must lie on different branches")
        if not left.fraction < right.fraction:
            raise ValueError("localization requires an ordered nonempty segment")

        baseline = left.signatures
        assert baseline is not None
        left_bound = left.fraction
        right_bound = right.fraction
        left_sample = left
        right_sample = right
        recoverable: list[tuple[EventKind, str]] = []
        for _ in range(self.options.maximum_iterations):
            if right_bound - left_bound <= self.options.fraction_tolerance:
                break
            fraction = 0.5 * (left_bound + right_bound)
            sample = observe(fraction)
            if sample.recoverable_kind is not None:
                recoverable.append((sample.recoverable_kind, sample.detail))
                right_bound = fraction
            elif same_branch(sample, baseline):
                left_bound = fraction
                left_sample = sample
            else:
                right_bound = fraction
                right_sample = sample

        if recoverable:
            lower = right_bound
            upper = right.fraction
            closest_right = right
            for _ in range(self.options.maximum_iterations):
                if upper - lower <= self.options.fraction_tolerance:
                    break
                fraction = 0.5 * (lower + upper)
                sample = observe(fraction)
                if sample.recoverable_kind is not None:
                    recoverable.append((sample.recoverable_kind, sample.detail))
                    lower = fraction
                else:
                    upper = fraction
                    closest_right = sample
            right_bound = upper
            right_sample = closest_right

        event_fraction = 0.5 * (left_bound + right_bound)
        branch = self.options.branch_selection
        selected = left_sample if branch == "left" else right_sample
        events = list(
            signature_events(
                baseline,
                right_sample.signatures or (),
                fraction=event_fraction,
                branch=branch,
            )
        )
        seen = {event.kind for event in events}
        for kind, detail in recoverable:
            if kind in seen:
                continue
            events.append(
                ContactTopologyEvent(
                    kind,
                    -1,
                    (),
                    event_fraction,
                    branch,
                    detail or "recoverable topology singularity",
                )
            )
            seen.add(kind)
        events.sort(key=lambda item: (item.interface, _KIND_ORDER[item.kind], item.entity))
        if not events:
            raise RuntimeError("branch changed without a classifiable contact event")
        return ContactTopologyEventBatch(
            "localized",
            left_bound,
            event_fraction,
            right_bound,
            selected.fraction,
            branch,
            tuple(events),
            selected,
        )
