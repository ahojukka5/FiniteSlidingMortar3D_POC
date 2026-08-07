"""Compatibility façade for event-aware solver result models.

New code should import these records from :mod:`contact3d.solvers.events`.
This module remains only until the compatibility cleanup in #136.
"""

from .solvers.events.results import (
    EventAwareAugmentedContactResult,
    EventAwareCoupledNewtonResult,
)

__all__ = [
    "EventAwareAugmentedContactResult",
    "EventAwareCoupledNewtonResult",
]
