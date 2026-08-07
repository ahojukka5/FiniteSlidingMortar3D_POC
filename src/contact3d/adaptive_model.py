"""Compatibility façade for adaptive continuation result contracts.

New code should import these records from :mod:`contact3d.solvers`.
This module remains only until the compatibility cleanup in #136.
"""

from .solvers.continuation import (
    AdaptiveAcceptedStep,
    AdaptiveAttemptAction,
    AdaptiveContactAttempt,
    AdaptiveContactResult,
    AdaptiveTerminationReason,
)

__all__ = [
    "AdaptiveAcceptedStep",
    "AdaptiveAttemptAction",
    "AdaptiveContactAttempt",
    "AdaptiveContactResult",
    "AdaptiveTerminationReason",
]
