"""Frictionless mortar contact residual and numerical tangent oracle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FloatArray
from .overlap import integrate_facet_pair
from .surface import ContactSurface, FacetPair, averaged_nodal_normals, discover_facet_pairs


@dataclass(frozen=True, slots=True)
class ContactPair:
    """Biased frictionless contact pair with a non-mortar slave surface."""

    slave: ContactSurface
    master: ContactSurface
    normal_penalty: float
    search_distance: float
    quadrature_points: int = 7

    def __post_init__(self) -> None:
        if self.normal_penalty <= 0.0:
            raise ValueError("normal_penalty must be positive")
        if self.search_distance <= 0.0:
            raise ValueError("search_distance must be positive")
        if self.quadrature_points not in (1, 3, 7):
            raise ValueError("quadrature_points must be 1, 3, or 7")


@dataclass(frozen=True, slots=True)
class GlobalMortarWeights:
    """Global standard-mortar operators assembled from overlapping facet pairs."""

    d: FloatArray
    m: FloatArray
    facet_pairs: tuple[FacetPair, ...]
    overlap_areas: FloatArray

    @property
    def row_areas(self) -> FloatArray:
        return np.sum(self.d, axis=1)

    @property
    def consistency_error(self) -> float:
        return float(np.max(np.abs(np.sum(self.d, axis=1) - np.sum(self.m, axis=1))))

    @property
    def total_area(self) -> float:
        return float(np.sum(self.overlap_areas))


@dataclass(frozen=True, slots=True)
class ContactEvaluation:
    """Current frictionless mortar residual and its principal diagnostics."""

    residual: FloatArray
    slave_nodes: FloatArray
    master_nodes: FloatArray
    slave_force: FloatArray
    master_force: FloatArray
    nodal_normals: FloatArray
    weighted_gap_vectors: FloatArray
    weighted_normal_gaps: FloatArray
    normal_gaps: FloatArray
    pressure: FloatArray
    active_rows: np.ndarray
    weights: GlobalMortarWeights

    @property
    def force_balance(self) -> FloatArray:
        return np.sum(self.slave_force, axis=0) + np.sum(self.master_force, axis=0)

    @property
    def force_balance_norm(self) -> float:
        return float(np.linalg.norm(self.force_balance))

    @property
    def moment_balance(self) -> FloatArray:
        slave_moment = np.sum(np.cross(self.slave_nodes, self.slave_force), axis=0)
        master_moment = np.sum(np.cross(self.master_nodes, self.master_force), axis=0)
        return slave_moment + master_moment

    @property
    def moment_balance_norm(self) -> float:
        return float(np.linalg.norm(self.moment_balance))

    @property
    def maximum_penetration(self) -> float:
        return float(np.max(np.maximum(self.normal_gaps, 0.0), initial=0.0))


def assemble_mortar_weights(
    pair: ContactPair,
    slave_nodes: FloatArray,
    master_nodes: FloatArray,
    *,
    facet_pairs: tuple[FacetPair, ...] | None = None,
    tolerance: float = 1.0e-12,
) -> GlobalMortarWeights:
    """Assemble global D/M operators from every current overlapping facet pair."""

    if facet_pairs is None:
        candidates = discover_facet_pairs(
            pair.slave,
            pair.master,
            slave_nodes,
            master_nodes,
            search_distance=pair.search_distance,
        )
    else:
        candidates = tuple((int(slave), int(master)) for slave, master in facet_pairs)

    dmat = np.zeros((pair.slave.node_count, pair.slave.node_count), dtype=float)
    mmat = np.zeros((pair.slave.node_count, pair.master.node_count), dtype=float)
    integrated_pairs: list[FacetPair] = []
    overlap_areas: list[float] = []

    for slave_index, master_index in candidates:
        if not 0 <= slave_index < len(pair.slave.facets):
            raise ValueError("slave facet pair index is out of range")
        if not 0 <= master_index < len(pair.master.facets):
            raise ValueError("master facet pair index is out of range")
        slave_facet = pair.slave.facets[slave_index]
        master_facet = pair.master.facets[master_index]
        local = integrate_facet_pair(
            slave_nodes[slave_facet],
            master_nodes[master_facet],
            quadrature_points=pair.quadrature_points,
            tolerance=tolerance,
        )
        if local.overlap.area <= tolerance:
            continue
        dmat[np.ix_(slave_facet, slave_facet)] += local.d
        mmat[np.ix_(slave_facet, master_facet)] += local.m
        integrated_pairs.append((slave_index, master_index))
        overlap_areas.append(local.overlap.area)

    return GlobalMortarWeights(
        dmat,
        mmat,
        tuple(integrated_pairs),
        np.asarray(overlap_areas, dtype=float),
    )


def evaluate_contact(
    pair: ContactPair,
    slave_displacement: FloatArray | None = None,
    master_displacement: FloatArray | None = None,
    *,
    facet_pairs: tuple[FacetPair, ...] | None = None,
    active_rows: np.ndarray | None = None,
    tolerance: float = 1.0e-12,
) -> ContactEvaluation:
    """Evaluate an area-normalized penalty residual in the current configuration.

    The exact weighted mortar gap from Eq. (6) is retained in
    ``weighted_gap_vectors`` and ``weighted_normal_gaps``. For a mesh-independent
    penalty scale, enforcement uses the corresponding row-area-normalized gap
    distance. This normalization is explicit and can later be replaced by the
    augmented-Lagrange multiplier update used in the paper.
    """

    slave_nodes = pair.slave.current_nodes(slave_displacement)
    master_nodes = pair.master.current_nodes(master_displacement)
    normals = averaged_nodal_normals(pair.slave, slave_nodes)
    weights = assemble_mortar_weights(
        pair,
        slave_nodes,
        master_nodes,
        facet_pairs=facet_pairs,
        tolerance=tolerance,
    )
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


def numerical_contact_tangent(
    pair: ContactPair,
    slave_displacement: FloatArray | None = None,
    master_displacement: FloatArray | None = None,
    *,
    relative_step: float = 2.0e-7,
    freeze_facet_pairs: bool = True,
    freeze_active_rows: bool = True,
) -> FloatArray:
    """Return a dense centered-difference tangent for verification-sized models."""

    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    slave_u = (
        np.zeros_like(pair.slave.reference_nodes)
        if slave_displacement is None
        else np.asarray(slave_displacement, dtype=float)
        .reshape(pair.slave.reference_nodes.shape)
        .copy()
    )
    master_u = (
        np.zeros_like(pair.master.reference_nodes)
        if master_displacement is None
        else np.asarray(master_displacement, dtype=float)
        .reshape(pair.master.reference_nodes.shape)
        .copy()
    )
    base = evaluate_contact(pair, slave_u, master_u)
    frozen_pairs = base.weights.facet_pairs if freeze_facet_pairs else None
    frozen_active = base.active_rows if freeze_active_rows else None
    coordinates = np.vstack(
        [pair.slave.reference_nodes + slave_u, pair.master.reference_nodes + master_u]
    )
    ndof = 3 * (pair.slave.node_count + pair.master.node_count)
    tangent = np.zeros((ndof, ndof), dtype=float)

    for column in range(ndof):
        node, component = divmod(column, 3)
        step = relative_step * max(1.0, abs(float(coordinates[node, component])))
        plus_slave = slave_u.copy()
        minus_slave = slave_u.copy()
        plus_master = master_u.copy()
        minus_master = master_u.copy()
        if node < pair.slave.node_count:
            plus_slave[node, component] += step
            minus_slave[node, component] -= step
        else:
            master_node = node - pair.slave.node_count
            plus_master[master_node, component] += step
            minus_master[master_node, component] -= step
        plus = evaluate_contact(
            pair,
            plus_slave,
            plus_master,
            facet_pairs=frozen_pairs,
            active_rows=frozen_active,
        ).residual
        minus = evaluate_contact(
            pair,
            minus_slave,
            minus_master,
            facet_pairs=frozen_pairs,
            active_rows=frozen_active,
        ).residual
        tangent[:, column] = (plus - minus) / (2.0 * step)
    return tangent
