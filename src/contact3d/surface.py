"""Surface topology, broad-phase discovery, and Puso nodal normals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FloatArray, IntArray

FacetPair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ContactSurface:
    """Reference surface mesh with consistently oriented TRI3/QUAD4 facets."""

    reference_nodes: FloatArray
    facets: tuple[IntArray, ...]
    normal_sign: float = 1.0

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
        object.__setattr__(self, "reference_nodes", nodes.copy())
        object.__setattr__(self, "facets", facets)

    @property
    def node_count(self) -> int:
        return len(self.reference_nodes)

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
    reference_sum = np.zeros_like(surface.reference_nodes)
    current_sum = np.zeros_like(current)
    attached = np.zeros(surface.node_count, dtype=np.int64)

    for facet in surface.facets:
        reference_points = surface.reference_nodes[facet]
        current_points = current[facet]
        for local_node, global_node in enumerate(facet):
            reference_cross = _corner_cross(reference_points, local_node)
            reference_measure = float(np.linalg.norm(reference_cross))
            if reference_measure <= tolerance:
                raise ValueError("surface facet has a degenerate reference corner")
            current_cross = _corner_cross(current_points, local_node)
            reference_sum[global_node] += reference_cross / reference_measure
            current_sum[global_node] += current_cross / reference_measure
            attached[global_node] += 1

    normals = np.zeros_like(current)
    used = attached > 0
    denominators = np.linalg.norm(reference_sum[used], axis=1)
    current_measures = np.linalg.norm(current_sum[used], axis=1)
    if np.any(denominators <= tolerance):
        raise ValueError("surface has a singular averaged reference normal")
    if np.any(current_measures <= tolerance):
        raise ValueError("surface has a singular averaged current normal")
    normals[used] = current_sum[used] / denominators[:, None]
    normals *= surface.normal_sign
    return normals


def _aabb_distance(first: FloatArray, second: FloatArray) -> float:
    first_min = np.min(first, axis=0)
    first_max = np.max(first, axis=0)
    second_min = np.min(second, axis=0)
    second_max = np.max(second, axis=0)
    separation = np.maximum(np.maximum(first_min - second_max, second_min - first_max), 0.0)
    return float(np.linalg.norm(separation))


def discover_facet_pairs(
    slave: ContactSurface,
    master: ContactSurface,
    slave_nodes: FloatArray,
    master_nodes: FloatArray,
    *,
    search_distance: float,
) -> tuple[FacetPair, ...]:
    """Return every facet pair whose current AABBs are within the search band."""

    if search_distance <= 0.0:
        raise ValueError("search_distance must be positive")
    if np.asarray(slave_nodes).shape != slave.reference_nodes.shape:
        raise ValueError("slave coordinates must match the slave surface")
    if np.asarray(master_nodes).shape != master.reference_nodes.shape:
        raise ValueError("master coordinates must match the master surface")

    pairs: list[FacetPair] = []
    for slave_index, slave_facet in enumerate(slave.facets):
        slave_points = slave_nodes[slave_facet]
        for master_index, master_facet in enumerate(master.facets):
            master_points = master_nodes[master_facet]
            if _aabb_distance(slave_points, master_points) <= search_distance:
                pairs.append((slave_index, master_index))
    return tuple(pairs)
