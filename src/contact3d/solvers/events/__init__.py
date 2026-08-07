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

_LAZY_EXPORTS = {
    "AdaptiveTopologyEventBatch": ".adaptive",
    "EventAwareAdaptiveContactResult": ".adaptive",
    "EventAwareScaleAwareAugmentedContactResult": ".scaling",
    "RestartAttemptDiagnostic": ".restart",
    "RestartCount": ".restart",
    "RestartDiagnosticOptions": ".restart",
    "RestartDiagnostics": ".restart",
    "RestartEventRecord": ".restart",
    "RestartLoopDiagnostic": ".restart",
    "RestartTerminationReason": ".restart",
    "analyze_restart_diagnostics": ".restart",
    "solve_event_aware_adaptive_contact_path": ".adaptive",
    "solve_event_aware_augmented_contact": ".augmentation",
    "solve_event_aware_coupled_equilibrium": ".newton",
    "solve_event_aware_scale_aware_augmented_contact": ".scaling",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        module = import_module(f"{__name__}{module_name}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "AdaptiveTopologyEventBatch",
    "EventAwareAdaptiveContactResult",
    "EventAwareAugmentedContactResult",
    "EventAwareCoupledNewtonResult",
    "EventAwareScaleAwareAugmentedContactResult",
    "MultiplierTransportRecord",
    "RestartAttemptDiagnostic",
    "RestartCount",
    "RestartDiagnosticOptions",
    "RestartDiagnostics",
    "RestartEventRecord",
    "RestartLoopDiagnostic",
    "RestartTerminationReason",
    "analyze_restart_diagnostics",
    "solve_event_aware_adaptive_contact_path",
    "solve_event_aware_augmented_contact",
    "solve_event_aware_coupled_equilibrium",
    "solve_event_aware_scale_aware_augmented_contact",
    "transport_multiplier_states",
]
