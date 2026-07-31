"""Public API for event-localized coupled contact solvers."""

from .event_augmented import solve_event_aware_augmented_contact
from .event_model import EventAwareAugmentedContactResult, EventAwareCoupledNewtonResult
from .event_newton import solve_event_aware_coupled_equilibrium

__all__ = [
    "EventAwareAugmentedContactResult",
    "EventAwareCoupledNewtonResult",
    "solve_event_aware_augmented_contact",
    "solve_event_aware_coupled_equilibrium",
]
