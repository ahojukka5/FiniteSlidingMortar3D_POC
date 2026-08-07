"""Compatibility façade for adaptive continuation options.

New code should import these records from :mod:`contact3d.solvers`.
This module remains only until the compatibility cleanup in #136.
"""

from .solvers.continuation import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
)

__all__ = [
    "AdaptiveContactOptions",
    "AdaptiveLoadOptions",
    "AdaptivePenaltyOptions",
]
