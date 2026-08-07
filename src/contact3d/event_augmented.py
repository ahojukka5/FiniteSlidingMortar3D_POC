"""Compatibility façade for event-aware augmented contact iteration.

New code should import the solver from :mod:`contact3d.solvers.events`.
This module remains only until the compatibility cleanup in #136.
"""

from .solvers.events.augmentation import solve_event_aware_augmented_contact

__all__ = ["solve_event_aware_augmented_contact"]
