"""Compatibility façade for event multiplier transport.

New code should import these objects from :mod:`contact3d.solvers.events`.
This module remains only until the compatibility cleanup in #136.
"""

from .solvers.events.multiplier_transport import (
    MultiplierTransportRecord,
    transport_multiplier_states,
)

__all__ = [
    "MultiplierTransportRecord",
    "transport_multiplier_states",
]
