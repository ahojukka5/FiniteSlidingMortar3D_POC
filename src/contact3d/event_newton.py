"""Compatibility façade for event-localized Newton equilibrium.

New code should import the solver from :mod:`contact3d.solvers.events`.
This module remains only until the compatibility cleanup in #136.
"""

from .solvers.events.newton import solve_event_aware_coupled_equilibrium

__all__ = ["solve_event_aware_coupled_equilibrium"]
