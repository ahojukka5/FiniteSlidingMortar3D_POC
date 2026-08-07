"""Public API for event-localized coupled contact solvers."""

from .event_adaptive import (
    AdaptiveTopologyEventBatch,
    EventAwareAdaptiveContactResult,
    solve_event_aware_adaptive_contact_path,
)
from .restart_diagnostics import (
    RestartAttemptDiagnostic,
    RestartCount,
    RestartDiagnosticOptions,
    RestartDiagnostics,
    RestartEventRecord,
    RestartLoopDiagnostic,
    analyze_restart_diagnostics,
)
from .solvers.events import (
    EventAwareAugmentedContactResult,
    EventAwareCoupledNewtonResult,
    EventAwareScaleAwareAugmentedContactResult,
    MultiplierTransportRecord,
    solve_event_aware_augmented_contact,
    solve_event_aware_coupled_equilibrium,
    solve_event_aware_scale_aware_augmented_contact,
    transport_multiplier_states,
)

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
    "analyze_restart_diagnostics",
    "solve_event_aware_adaptive_contact_path",
    "solve_event_aware_augmented_contact",
    "solve_event_aware_coupled_equilibrium",
    "solve_event_aware_scale_aware_augmented_contact",
    "transport_multiplier_states",
]
