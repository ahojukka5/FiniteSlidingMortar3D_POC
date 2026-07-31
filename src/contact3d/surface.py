"""Surface topology, broad-phase discovery, and Puso nodal normals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .broad_phase import (
    BroadPhaseDiagnostics,
    FacetAABBTree,
    FacetPair,
    FacetPairSearchResult,
    facet_aabbs,
)
from .model import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class ContactSurface:
    """Reference surface mesh with consistently oriented TRI3/QUAD4 facets."""

    reference_nodes: FloatArray
    facets: tuple[IntArray, ...]
    normal_sign: float = 1.0
    _broad_phase_tree: FacetAABBTree = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        nodes = np.asarray(self.reference_nodes, dtype=float)
        facets = tuple(np.asarray(facet, dtype=np.int64) for facet in self.facets)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("surface nodes must have shape (node_count, 3)")
        if len(nodes) == 0 or len(facets) == 0:
            raise ValueError("surface must contain nodes and at least one facet")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("surface nodes must be finite")
        if self.normal_sign not in (-1.0, 1.0):
            raise ValueError("normal_sign must be -1 or +1")
        for facet in facets:
            if facet.ndim != 1 or len(facet) not in (3, 4):
                raise ValueError("surface facets must contain three or four node indices")
            if len(np.unique(facet)) != len(facet):
                raise ValueError("surface facet contains duplicate nodes")
            if np.any(facet < 0) or np.any(facet >= len(nodes)):
                raise ValueError("surface facet node index is out of range")
        copied_nodes = nodes.copy()
        copied_facets = tuple(facet.copy() for facet in facets)
        object.__setattr__(self, "reference_nodes", copied_nodes)
        object.__setattr__(self, "facets", copied_facets)
        object.__setattr__(
            self,
            "_broad_phase_tree",
            FacetAABBTree.build(copied_nodes, copied_facets),
        )

    @property
    def node_count(self) -> int:
        return len(self.reference_nodes)

    @property
    def broad_phase_tree(self) -> FacetAABBTree:
        """Return the immutable reference-topology facet BVH."""

        return self._broad_phase_tree

    def current_nodes(self, displacement: FloatArray | None = None) -> FloatArray:
        """Return current coordinates after validating an optional displacement."""

        if displacement is None:
            return self.reference_nodes.copy()
        values = np.asarray(displacement, dtype=float)
        if values.shape == (3 * self.node_count,):
            values = values.reshape((-1, 3))
        if values.shape != self.reference_nodes.shape:
            raise ValueError("surface displacement must match the nodal coordinate shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("surface displacement must be finite")
        return self.reference_nodes + values


def _corner_cross(points: FloatArray, local_node: int) -> FloatArray:
    """Return the oriented corner area vector for one facet vertex."""

    current = points[local_node]
    next_point = points[(local_node + 1) % len(points)]
    previous_point = points[(local_node - 1) % len(points)]
    return np.cross(next_point - current, previous_point - current)


def _cross_matrix(vector: FloatArray) -> FloatArray:
    """Return the matrix ``W`` such that ``W @ value == vector x value``."""

    x, y, z = map(float, vector)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def _normal_workspace(
    surface: ContactSurface,
    current_nodes: FloatArray,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, IntArray, FloatArray]:
    """Accumulate the fixed reference and current corner-normal sums."""

    reference_sum = np.zeros_like(surface.reference_nodes)
    current_sum = np.zeros_like(current_nodes)
    attached = np.zeros(surface.node_count, dtype=np.int64)

    for facet in surface.facets:
        reference_points = surface.reference_nodes[facet]
        current_points = current_nodes[facet]
        for local_node, global_node in enumerate(facet):
            reference_cross = _corner_cross(reference_points, local_node)
            reference_measure = float(np.linalg.norm(reference_cross))
            if reference_measure <= tolerance:
                raise ValueError("surface facet has a degenerate reference corner")
            current_cross = _corner_cross(current_points, local_node)
            reference_sum[global_node] += reference_cross / reference_measure
            current_sum[global_node] += current_cross / reference_measure
            attached[global_node] += 1

    used = attached > 0
    denominators = np.zeros(surface.node_count, dtype=float)
    denominators[used] = np.linalg.norm(reference_sum[used], axis=1)
    current_measures = np.linalg.norm(current_sum[used], axis=1)
    if np.any(denominators[used] <= tolerance):
        raise ValueError("surface has a singular averaged reference normal")
    if np.any(current_measures <= tolerance):
        raise ValueError("surface has a singular averaged current normal")
    return reference_sum, current_sum, attached, denominators


def averaged_nodal_normals(
    surface: ContactSurface,
    current_nodes: FloatArray,
    *,
    tolerance: float = 1.0e-14,
) -> FloatArray:
    """Evaluate the current-configuration nodal normals from Appendix A.

    Each current corner area vector is divided by its reference magnitude before
    accumulation. The nodal sum is divided by the magnitude of the corresponding
    reference sum, matching Puso and Laursen's nominal normal in Eqs. (A.3)-(A.4).
    The result is generally not unit length after deformation.
    """

    current = np.asarray(current_nodes, dtype=float)
    if current.shape != surface.reference_nodes.shape:
        raise ValueError("current surface coordinates must match the reference surface")
    _, current_sum, attached, denominators = _normal_workspace(
        surface,
        current,
        tolerance,
    )
    normals = np.zeros_like(current)
    used = attached > 0
    normals[used] = current_sum[used] / denominators[used, None]
    normals *= surface.normal_sign
    return normals


def averaged_nodal_normal_jacobian(
    surface: ContactSurface,
    current_nodes: FloatArray,
    *,
    tolerance: float = 1.0e-14,
) -> FloatArray:
    """Differentiate the Appendix A nominal normals analytically.

    The returned tensor has axes ``(output_node, output_component, input_node,
    input_component)``. Puso and Laursen's Appendix A keeps the normalization
    denominator in the reference configuration, so only the current corner cross
    products contribute to this derivative.
    """

    current = np.asarray(current_nodes, dtype=float)
    if current.shape != surface.reference_nodes.shape:
        raise ValueError("current surface coordinates must match the reference surface")
    _, _, _, denominators = _normal_workspace(surface, current, tolerance)
    jacobian = np.zeros(
        (surface.node_count, 3, surface.node_count, 3),
        dtype=float,
    )

    for facet in surface.facets:
        reference_points = surface.reference_nodes[facet]
        current_points = current[facet]
        facet_node_count = len(facet)
        for local_node, global_node_value in enumerate(facet):
            global_node = int(global_node_value)
            next_local = (local_node + 1) % facet_node_count
            previous_local = (local_node - 1) % facet_node_count
            next_node = int(facet[next_local])
            previous_node = int(facet[previous_local])

            reference_cross = _corner_cross(reference_points, local_node)
            reference_measure = float(np.linalg.norm(reference_cross))
            scale = surface.normal_sign / (
                reference_measure * denominators[global_node]
            )
            next_edge = current_points[next_local] - current_points[local_node]
            previous_edge = current_points[previous_local] - current_points[local_node]

            # dc = d(next-current) x previous + next x d(previous-current)
            jacobian[global_node, :, next_node, :] += scale * (
                -_cross_matrix(previous_edge)
            )
            jacobian[global_node, :, previous_node, :] += scale * _cross_matrix(
                next_edge
            )
            jacobian[global_node, :, global_node, :] += scale * (
                _cross_matrix(previous_edge) - _cross_matrix(next_edge)
            )

    return jacobian


def _validate_pair_search(
    slave: ContactSurface,
    master: ContactSurface,
    slave_nodes: FloatArray,
    master_nodes: FloatArray,
    search_distance: float,
) -> tuple[FloatArray, FloatArray]:
    if not np.isfinite(search_distance) or search_distance <= 0.0:
        raise ValueError("search_distance must be finite and positive")
    slave_values = np.asarray(slave_nodes, dtype=float)
    master_values = np.asarray(master_nodes, dtype=float)
    if slave_values.shape != slave.reference_nodes.shape:
        raise ValueError("slave coordinates must match the slave surface")
    if master_values.shape != master.reference_nodes.shape:
        raise ValueError("master coordinates must match the master surface")
    if not np.all(np.isfinite(slave_values)) or not np.all(np.isfinite(master_values)):
        raise ValueError("contact coordinates must be finite")
    return slave_values, master_values


def discover_facet_pairs_with_diagnostics(
    slave: ContactSurface,
    master: ContactSurface,
    slave_nodes: FloatArray,
    master_nodes: FloatArray,
    *,
    search_distance: float,
) -> FacetPairSearchResult:
    """Query the cached master BVH and return pairs plus operation diagnostics."""

    slave_values, master_values = _validate_pair_search(
        slave,
        master,
        slave_nodes,
        master_nodes,
        search_distance,
    )
    slave_minimums, slave_maximums = facet_aabbs(
        slave_values,
        slave.facets,
        node_count=slave.node_count,
    )
    master_tree = master.broad_phase_tree.refit(master_values)
    return master_tree.query(
        slave_minimums,
        slave_maximums,
        search_distance=search_distance,
    )


def discover_facet_pairs(
    slave: ContactSurface,
    master: ContactSurface,
    slave_nodes: FloatArray,
    master_nodes: FloatArray,
    *,
    search_distance: float,
) -> tuple[FacetPair, ...]:
    """Return every facet pair whose current AABBs are within the search band."""

    return discover_facet_pairs_with_diagnostics(
        slave,
        master,
        slave_nodes,
        master_nodes,
        search_distance=search_distance,
    ).pairs


def discover_facet_pairs_brute_force(
    slave: ContactSurface,
    master: ContactSurface,
    slave_nodes: FloatArray,
    master_nodes: FloatArray,
    *,
    search_distance: float,
) -> tuple[FacetPair, ...]:
    """Return the quadratic broad-phase oracle used to verify the BVH."""

    slave_values, master_values = _validate_pair_search(
        slave,
        master,
        slave_nodes,
        master_nodes,
        search_distance,
    )
    slave_minimums, slave_maximums = facet_aabbs(
        slave_values,
        slave.facets,
        node_count=slave.node_count,
    )
    master_minimums, master_maximums = facet_aabbs(
        master_values,
        master.facets,
        node_count=master.node_count,
    )
    pairs: list[FacetPair] = []
    for slave_index in range(len(slave.facets)):
        for master_index in range(len(master.facets)):
            separation = np.maximum(
                np.maximum(
                    slave_minimums[slave_index] - master_maximums[master_index],
                    master_minimums[master_index] - slave_maximums[slave_index],
                ),
                0.0,
            )
            if float(np.linalg.norm(separation)) <= search_distance:
                pairs.append((slave_index, master_index))
    return tuple(pairs)


__all__ = [
    "BroadPhaseDiagnostics",
    "ContactSurface",
    "FacetAABBTree",
    "FacetPair",
    "FacetPairSearchResult",
    "averaged_nodal_normal_jacobian",
    "averaged_nodal_normals",
    "discover_facet_pairs",
    "discover_facet_pairs_brute_force",
    "discover_facet_pairs_with_diagnostics",
]
