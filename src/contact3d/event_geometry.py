"""Branch observations for event-localized contact solves."""

from __future__ import annotations

import numpy as np

from .clipping import ClippingTopologyError
from .coupled import (
    ContactInterfaceEvaluation,
    CoupledEquilibriumEvaluation,
    CoupledEquilibriumProblem,
    evaluate_coupled_equilibrium,
)
from .enforcement_state import AugmentedLagrangeState
from .geometry import polygon_signed_area
from .model import FloatArray
from .overlap import build_facet_overlap
from .pallets import PalletTopologyError
from .parametric import InverseMapTopologyError
from .topology_events import ContactTopologySignature, EventKind, TopologyObservation

_RECOVERABLE_ERRORS = (ClippingTopologyError, PalletTopologyError, InverseMapTopologyError)


def _validated_states(
    problem: CoupledEquilibriumProblem,
    states: tuple[AugmentedLagrangeState, ...] | None,
) -> tuple[AugmentedLagrangeState, ...]:
    values = problem.initial_states() if states is None else tuple(states)
    if len(values) != len(problem.interfaces):
        raise ValueError("one multiplier state is required for every contact interface")
    return values


def _recoverable_kind(error: BaseException) -> EventKind:
    if isinstance(error, ClippingTopologyError):
        return "clipping_vertex_edge"
    if isinstance(error, PalletTopologyError):
        return "pallet_transition"
    if isinstance(error, InverseMapTopologyError):
        return "inverse_map_boundary"
    raise TypeError(f"unsupported recoverable contact event: {type(error).__name__}")


def _relative_residual(norm: float, initial_norm: float) -> float:
    return norm / max(initial_norm, np.finfo(float).tiny)


def contact_topology_signatures(
    problem: CoupledEquilibriumProblem,
    displacement: FloatArray,
    contacts: tuple[ContactInterfaceEvaluation, ...],
    *,
    tolerance: float,
) -> tuple[ContactTopologySignature, ...]:
    """Return event-compatible signatures for already evaluated interfaces."""

    values = np.asarray(displacement, dtype=float).reshape((-1, 3))
    if len(contacts) != len(problem.interfaces):
        raise ValueError("one contact evaluation is required for every interface")
    signatures: list[ContactTopologySignature] = []
    for interface, contact in zip(problem.interfaces, contacts, strict=True):
        geometry_tokens: list[tuple[int, int, int, int, int]] = []
        pair = getattr(interface, "pair", None)
        slave_nodes = getattr(interface, "slave_nodes", None)
        master_nodes = getattr(interface, "master_nodes", None)
        if pair is not None and slave_nodes is not None and master_nodes is not None:
            slave_current = pair.slave.reference_nodes + values[slave_nodes]
            master_current = pair.master.reference_nodes + values[master_nodes]
            for slave_index, master_index in contact.signature.facet_pairs:
                slave_facet = pair.slave.facets[slave_index]
                master_facet = pair.master.facets[master_index]
                overlap = build_facet_overlap(
                    slave_current[slave_facet],
                    master_current[master_facet],
                    tolerance=tolerance,
                )
                signed_area = polygon_signed_area(overlap.intersection_polygon)
                orientation = int(np.sign(signed_area))
                geometry_tokens.append(
                    (
                        int(slave_index),
                        int(master_index),
                        len(overlap.intersection_polygon),
                        len(overlap.pallets),
                        orientation,
                    )
                )
        signatures.append(
            ContactTopologySignature(
                tuple(contact.signature.facet_pairs),
                tuple(contact.signature.active_rows),
                tuple(contact.signature.supported_rows),
                tuple(geometry_tokens),
            )
        )
    return tuple(signatures)


def _event_signatures(
    problem: CoupledEquilibriumProblem,
    evaluation: CoupledEquilibriumEvaluation,
    *,
    tolerance: float,
) -> tuple[ContactTopologySignature, ...]:
    return contact_topology_signatures(
        problem,
        evaluation.displacement,
        evaluation.contacts,
        tolerance=tolerance,
    )


def _observation(
    problem: CoupledEquilibriumProblem,
    states: tuple[AugmentedLagrangeState, ...],
    displacement: FloatArray,
    step: FloatArray,
    fraction: float,
    *,
    load_factor: float,
    tolerance: float,
) -> TopologyObservation:
    try:
        evaluation = evaluate_coupled_equilibrium(
            problem,
            displacement + fraction * step,
            states,
            load_factor=load_factor,
            assemble_tangent=False,
            tolerance=tolerance,
        )
    except _RECOVERABLE_ERRORS as error:
        return TopologyObservation.recoverable(
            fraction,
            _recoverable_kind(error),
            str(error),
        )
    signatures = _event_signatures(problem, evaluation, tolerance=tolerance)
    return TopologyObservation.valid(fraction, signatures, evaluation)
