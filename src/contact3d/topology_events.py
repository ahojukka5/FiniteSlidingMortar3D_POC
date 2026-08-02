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
from .topology_signature import (
    TOPOLOGY_SEQUENCE_SCHEMA,
    TOPOLOGY_SIGNATURE_SCHEMA,
    canonical_topology_json,
    canonical_topology_sequence,
    canonical_topology_signature,
    topology_sequence_hash,
    topology_signature_hash,
    validate_topology_signature_record,
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
    "TOPOLOGY_SEQUENCE_SCHEMA",
    "TOPOLOGY_SIGNATURE_SCHEMA",
    "TopologyEventLocalizationOptions",
    "TopologyObservation",
    "canonical_topology_json",
    "canonical_topology_sequence",
    "canonical_topology_signature",
    "topology_sequence_hash",
    "topology_signature_hash",
    "validate_topology_signature_record",
]
