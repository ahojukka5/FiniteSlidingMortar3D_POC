"""Frictionless standard-mortar residual evaluation."""

from __future__ import annotations

import numpy as np

from ..geometry import FacetPair, FloatArray, averaged_nodal_normals
from .model import ContactEvaluation, ContactPair, GlobalMortarWeights
from .operators import assemble_mortar_weights


def _validate_frozen_weights(pair: ContactPair, weights: GlobalMortarWeights) -> None:
    if weights.d.shape != (pair.slave.node_count, pair.slave.node_count):
        raise ValueError("frozen D matrix has an incompatible shape")
    if weights.m.shape != (pair.slave.node_count, pair.master.node_count):
        raise ValueError("frozen M matrix has an incompatible shape")


def evaluate_contact(
    pair: ContactPair,
    slave_displacement: FloatArray | None = None,
    master_displacement: FloatArray | None = None,
    *,
    facet_pairs: tuple[FacetPair, ...] | None = None,
    active_rows: np.ndarray | None = None,
    frozen_weights: GlobalMortarWeights | None = None,
    tolerance: float = 1.0e-12,
) -> ContactEvaluation:
    """Evaluate an area-normalized penalty residual in the current configuration.

    The exact weighted mortar gap is retained in ``weighted_gap_vectors`` and
    ``weighted_normal_gaps``. Enforcement uses the corresponding row-area-
    normalized gap distance for a mesh-independent penalty scale.
    """

    slave_nodes = pair.slave.current_nodes(slave_displacement)
    master_nodes = pair.master.current_nodes(master_displacement)
    normals = averaged_nodal_normals(pair.slave, slave_nodes)
    if frozen_weights is None:
        weights = assemble_mortar_weights(
            pair,
            slave_nodes,
            master_nodes,
            facet_pairs=facet_pairs,
            tolerance=tolerance,
        )
    else:
        _validate_frozen_weights(pair, frozen_weights)
        weights = frozen_weights

    weighted_gap_vectors = weights.d @ slave_nodes - weights.m @ master_nodes
    weighted_normal_gaps = np.einsum("ij,ij->i", normals, weighted_gap_vectors)
    row_areas = weights.row_areas
    supported = row_areas > tolerance
    normal_gaps = np.zeros(pair.slave.node_count, dtype=float)
    normal_gaps[supported] = weighted_normal_gaps[supported] / row_areas[supported]

    if active_rows is None:
        active = supported & (normal_gaps > 0.0)
    else:
        active = np.asarray(active_rows, dtype=bool)
        if active.shape != (pair.slave.node_count,):
            raise ValueError("active_rows must match the slave-node count")
        active = active & supported

    pressure = pair.normal_penalty * np.where(active, normal_gaps, 0.0)
    traction = pressure[:, None] * normals
    slave_force = weights.d.T @ traction
    master_force = -(weights.m.T @ traction)
    residual = np.concatenate([slave_force.ravel(), master_force.ravel()])
    return ContactEvaluation(
        residual=residual,
        slave_nodes=slave_nodes,
        master_nodes=master_nodes,
        slave_force=slave_force,
        master_force=master_force,
        nodal_normals=normals,
        weighted_gap_vectors=weighted_gap_vectors,
        weighted_normal_gaps=weighted_normal_gaps,
        normal_gaps=normal_gaps,
        pressure=pressure,
        active_rows=active,
        weights=weights,
    )
