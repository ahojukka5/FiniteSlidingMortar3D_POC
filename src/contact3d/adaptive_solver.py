"""Compatibility façade for the adaptive continuation driver.

New code should import these helpers from :mod:`contact3d.solvers`.
This module remains only until the compatibility cleanup in #136.
"""

from .solvers.continuation import (
    AdaptiveSolver,
    contact_penalties,
    solve_adaptive_contact_path,
    with_contact_penalties,
)

__all__ = [
    "AdaptiveSolver",
    "contact_penalties",
    "solve_adaptive_contact_path",
    "with_contact_penalties",
]
