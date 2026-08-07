"""Compatibility façade for event-aware adaptive continuation.

New code should import this API from :mod:`contact3d.solvers.events`.
This module remains only until the compatibility cleanup in #136.
"""

from .solvers.events.adaptive import (
    AdaptiveTopologyEventBatch,
    EventAwareAdaptiveContactResult,
    solve_event_aware_adaptive_contact_path,
)

__all__ = [
    "AdaptiveTopologyEventBatch",
    "EventAwareAdaptiveContactResult",
    "solve_event_aware_adaptive_contact_path",
]
