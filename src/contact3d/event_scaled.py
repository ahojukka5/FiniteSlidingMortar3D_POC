"""Compatibility façade for event-aware scale-aware augmentation.

New code should import these objects from :mod:`contact3d.solvers.events`.
This module remains only until the compatibility cleanup in #136.
"""

from .solvers.events import (
    EventAwareScaleAwareAugmentedContactResult,
    solve_event_aware_scale_aware_augmented_contact,
)

__all__ = [
    "EventAwareScaleAwareAugmentedContactResult",
    "solve_event_aware_scale_aware_augmented_contact",
]
