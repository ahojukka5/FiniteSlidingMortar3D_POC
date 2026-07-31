"""Public API for typed contact-topology events and segment localization."""

from .topology_model import (
    BranchSelection,
    BranchSignature,
    ContactTopologyEvent,
    ContactTopologyEventBatch,
    ContactTopologySignature,
    EventKind,
    MachineState,
    TopologyEventLocalizationOptions,
    TopologyObservation,
)
from .topology_state import ContactTopologyStateMachine

__all__ = [
    "BranchSelection",
    "BranchSignature",
    "ContactTopologyEvent",
    "ContactTopologyEventBatch",
    "ContactTopologySignature",
    "ContactTopologyStateMachine",
    "EventKind",
    "MachineState",
    "TopologyEventLocalizationOptions",
    "TopologyObservation",
]
