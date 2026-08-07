"""Solver-independent contact-topology signature geometry."""

from __future__ import annotations

import numpy as np

from .coupling import ContactInterfaceEvaluation, CoupledEquilibriumProblem
from .geometry import polygon_signed_area
from .mechanics import FloatArray
from .overlap import build_facet_overlap
from .topology_model import ContactTopologySignature


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


__all__ = ["contact_topology_signatures"]
