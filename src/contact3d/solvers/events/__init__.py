"""Topology-event-aware nonlinear solver variants."""

from .scaling import (
    EventAwareScaleAwareAugmentedContactResult,
    solve_event_aware_scale_aware_augmented_contact,
)

__all__ = [
    "EventAwareScaleAwareAugmentedContactResult",
    "solve_event_aware_scale_aware_augmented_contact",
]
