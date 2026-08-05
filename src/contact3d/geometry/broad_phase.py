"""Deterministic refittable AABB trees for contact-facet discovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FloatArray, IntArray

FacetPair = tuple[int, int]


def _validated_coordinates(nodes: FloatArray, node_count: int) -> FloatArray:
    values = np.asarray(nodes, dtype=float)
    if values.shape != (node_count, 3):
        raise ValueError("current coordinates must have shape (node_count, 3)")
    if not np.all(np.isfinite(values)):
        raise ValueError("current coordinates must be finite")
    return values


def facet_aabbs(
    nodes: FloatArray,
    facets: tuple[IntArray, ...],
    *,
    node_count: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return cached minimum and maximum coordinates for every facet."""

    expected = len(nodes) if node_count is None else node_count
    values = _validated_coordinates(nodes, expected)
    minimums = np.empty((len(facets), 3), dtype=float)
    maximums = np.empty((len(facets), 3), dtype=float)
    for index, facet in enumerate(facets):
        points = values[facet]
        minimums[index] = np.min(points, axis=0)
        maximums[index] = np.max(points, axis=0)
    return minimums, maximums


def _aabb_distance(
    first_minimum: FloatArray,
    first_maximum: FloatArray,
    second_minimum: FloatArray,
    second_maximum: FloatArray,
) -> float:
    separation = np.maximum(
        np.maximum(first_minimum - second_maximum, second_minimum - first_maximum),
        0.0,
    )
    return float(np.linalg.norm(separation))


@dataclass(frozen=True, slots=True)
class AABBTreeNode:
    """One immutable topology node in a facet AABB tree."""

    start: int
    stop: int
    left: int = -1
    right: int = -1

    @property
    def is_leaf(self) -> bool:
        return self.left < 0


@dataclass(frozen=True, slots=True)
class BroadPhaseDiagnostics:
    """Operation counts for one slave-against-master tree query."""

    slave_facet_count: int
    master_facet_count: int
    tree_node_count: int
    leaf_node_count: int
    node_visits: int
    leaf_visits: int
    facet_tests: int
    accepted_pairs: int

    @property
    def brute_force_tests(self) -> int:
        return self.slave_facet_count * self.master_facet_count

    @property
    def tested_fraction(self) -> float:
        if self.brute_force_tests == 0:
            return 0.0
        return self.facet_tests / self.brute_force_tests


@dataclass(frozen=True, slots=True)
class FacetPairSearchResult:
    """Deterministically ordered broad-phase pairs and query diagnostics."""

    pairs: tuple[FacetPair, ...]
    diagnostics: BroadPhaseDiagnostics


@dataclass(frozen=True, slots=True)
class FacetAABBTree:
    """Reference-topology BVH that can be cheaply refitted to current coordinates."""

    node_count: int
    facets: tuple[IntArray, ...]
    facet_order: IntArray
    nodes: tuple[AABBTreeNode, ...]
    root: int
    leaf_size: int

    @classmethod
    def build(
        cls,
        reference_nodes: FloatArray,
        facets: tuple[IntArray, ...],
        *,
        leaf_size: int = 8,
    ) -> FacetAABBTree:
        """Build a deterministic median-split topology from reference facet centroids."""

        if leaf_size < 1:
            raise ValueError("leaf_size must be positive")
        values = np.asarray(reference_nodes, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
            raise ValueError("reference coordinates must have shape (node_count, 3)")
        if not np.all(np.isfinite(values)):
            raise ValueError("reference coordinates must be finite")
        if len(facets) == 0:
            raise ValueError("AABB tree requires at least one facet")

        minimums, maximums = facet_aabbs(values, facets, node_count=len(values))
        centroids = 0.5 * (minimums + maximums)
        order = np.arange(len(facets), dtype=np.int64)
        topology: list[AABBTreeNode | None] = []

        def build_range(start: int, stop: int) -> int:
            node_index = len(topology)
            topology.append(None)
            count = stop - start
            if count <= leaf_size:
                topology[node_index] = AABBTreeNode(start, stop)
                return node_index

            facet_ids = order[start:stop]
            extents = np.ptp(centroids[facet_ids], axis=0)
            axis = int(np.argmax(extents))
            ordered = sorted(
                (int(value) for value in facet_ids),
                key=lambda value: (float(centroids[value, axis]), value),
            )
            order[start:stop] = ordered
            middle = start + count // 2
            left = build_range(start, middle)
            right = build_range(middle, stop)
            topology[node_index] = AABBTreeNode(start, stop, left, right)
            return node_index

        root = build_range(0, len(facets))
        nodes = tuple(node for node in topology if node is not None)
        if len(nodes) != len(topology):
            raise RuntimeError("AABB tree construction left an incomplete node")
        return cls(
            len(values),
            tuple(np.asarray(facet, dtype=np.int64).copy() for facet in facets),
            order.copy(),
            nodes,
            root,
            leaf_size,
        )

    @property
    def leaf_count(self) -> int:
        return sum(node.is_leaf for node in self.nodes)

    def refit(self, current_nodes: FloatArray) -> RefitFacetAABBTree:
        """Refit all facet and tree-node bounds without rebuilding topology."""

        minimums, maximums = facet_aabbs(
            current_nodes,
            self.facets,
            node_count=self.node_count,
        )
        node_minimums = np.empty((len(self.nodes), 3), dtype=float)
        node_maximums = np.empty((len(self.nodes), 3), dtype=float)
        for node_index in range(len(self.nodes) - 1, -1, -1):
            node = self.nodes[node_index]
            if node.is_leaf:
                facet_ids = self.facet_order[node.start : node.stop]
                node_minimums[node_index] = np.min(minimums[facet_ids], axis=0)
                node_maximums[node_index] = np.max(maximums[facet_ids], axis=0)
            else:
                node_minimums[node_index] = np.minimum(
                    node_minimums[node.left],
                    node_minimums[node.right],
                )
                node_maximums[node_index] = np.maximum(
                    node_maximums[node.left],
                    node_maximums[node.right],
                )
        return RefitFacetAABBTree(
            self,
            minimums,
            maximums,
            node_minimums,
            node_maximums,
        )


@dataclass(frozen=True, slots=True)
class RefitFacetAABBTree:
    """Current-configuration bounds attached to an immutable tree topology."""

    topology: FacetAABBTree
    facet_minimums: FloatArray
    facet_maximums: FloatArray
    node_minimums: FloatArray
    node_maximums: FloatArray

    def query(
        self,
        slave_minimums: FloatArray,
        slave_maximums: FloatArray,
        *,
        search_distance: float,
    ) -> FacetPairSearchResult:
        """Query all slave facet boxes against this master tree in one pass."""

        if not np.isfinite(search_distance) or search_distance <= 0.0:
            raise ValueError("search_distance must be finite and positive")
        slave_minimums = np.asarray(slave_minimums, dtype=float)
        slave_maximums = np.asarray(slave_maximums, dtype=float)
        if slave_minimums.shape != slave_maximums.shape:
            raise ValueError("slave minimum and maximum bounds must have matching shapes")
        if slave_minimums.ndim != 2 or slave_minimums.shape[1] != 3:
            raise ValueError("slave bounds must have shape (facet_count, 3)")
        if not np.all(np.isfinite(slave_minimums)) or not np.all(
            np.isfinite(slave_maximums)
        ):
            raise ValueError("slave bounds must be finite")
        if np.any(slave_minimums > slave_maximums):
            raise ValueError("slave minimum bounds must not exceed maximum bounds")

        pairs: list[FacetPair] = []
        node_visits = 0
        leaf_visits = 0
        facet_tests = 0
        for slave_index, (slave_minimum, slave_maximum) in enumerate(
            zip(slave_minimums, slave_maximums, strict=True)
        ):
            stack = [self.topology.root]
            while stack:
                node_index = stack.pop()
                node_visits += 1
                if (
                    _aabb_distance(
                        slave_minimum,
                        slave_maximum,
                        self.node_minimums[node_index],
                        self.node_maximums[node_index],
                    )
                    > search_distance
                ):
                    continue

                node = self.topology.nodes[node_index]
                if node.is_leaf:
                    leaf_visits += 1
                    for master_index_value in self.topology.facet_order[
                        node.start : node.stop
                    ]:
                        master_index = int(master_index_value)
                        facet_tests += 1
                        if (
                            _aabb_distance(
                                slave_minimum,
                                slave_maximum,
                                self.facet_minimums[master_index],
                                self.facet_maximums[master_index],
                            )
                            <= search_distance
                        ):
                            pairs.append((slave_index, master_index))
                else:
                    # Push right first so the deterministic left child is visited first.
                    stack.append(node.right)
                    stack.append(node.left)

        pairs.sort()
        diagnostics = BroadPhaseDiagnostics(
            len(slave_minimums),
            len(self.topology.facets),
            len(self.topology.nodes),
            self.topology.leaf_count,
            node_visits,
            leaf_visits,
            facet_tests,
            len(pairs),
        )
        return FacetPairSearchResult(tuple(pairs), diagnostics)
