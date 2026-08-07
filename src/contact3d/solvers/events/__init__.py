"""Topology-event-aware nonlinear solver variants."""

from importlib import import_module

from .multiplier_transport import (
    MultiplierTransportRecord,
    transport_multiplier_states,
)
from .results import (
    EventAwareAugmentedContactResult,
    EventAwareCoupledNewtonResult,
)

_SCALING_EXPORTS = frozenset(
    {
        "EventAwareScaleAwareAugmentedContactResult",
        "solve_event_aware_scale_aware_augmented_contact",
    }
)


def __getattr__(name: str) -> object:
    if name in _SCALING_EXPORTS:
        module = import_module(f"{__name__}.scaling")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _SCALING_EXPORTS)


__all__ = [
    "EventAwareAugmentedContactResult",
    "EventAwareCoupledNewtonResult",
    "EventAwareScaleAwareAugmentedContactResult",
    "MultiplierTransportRecord",
    "solve_event_aware_scale_aware_augmented_contact",
    "transport_multiplier_states",
]
