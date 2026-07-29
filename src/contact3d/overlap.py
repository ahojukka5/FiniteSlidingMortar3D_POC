"""Facet-pair overlap construction and standard mortar integration."""

from __future__ import annotations

import numpy as np

from .geometry import (
    clip_convex_polygon,
    ensure_counterclockwise,
    facet_projection_plane,
    project_to_plane,
    triangulate_convex_polygon,
)
from .model import FacetOverlap, FloatArray, LocalMortarWeights
from .quadrature import triangle_rule
from .shapes import infer_facet_kind, inverse_map_2d, shape_values


def build_facet_overlap(
    slave_points: FloatArray,
    master_points: FloatArray,
    *,
    tolerance: float = 1.0e-12,
) -> FacetOverlap:
    """Project, clip, and triangulate one slave/master facet pair."""

    slave_points = np.asarray(slave_points, dtype=float)
    master_points = np.asarray(master_points, dtype=float)
    slave_kind = infer_facet_kind(slave_points)
    infer_facet_kind(master_points)
    plane = facet_projection_plane(slave_points, slave_kind)
    slave_polygon = ensure_counterclockwise(project_to_plane(slave_points, plane))
    master_polygon = ensure_counterclockwise(project_to_plane(master_points, plane))
    intersection = clip_convex_polygon(slave_polygon, master_polygon, tolerance=tolerance)
    pallets = triangulate_convex_polygon(intersection, tolerance=tolerance)
    return FacetOverlap(plane, slave_polygon, master_polygon, intersection, pallets)


def integrate_facet_pair(
    slave_points: FloatArray,
    master_points: FloatArray,
    *,
    quadrature_points: int = 7,
    tolerance: float = 1.0e-12,
) -> LocalMortarWeights:
    """Integrate local D and M matrices over the projected intersection polygon."""

    slave_points = np.asarray(slave_points, dtype=float)
    master_points = np.asarray(master_points, dtype=float)
    slave_kind = infer_facet_kind(slave_points)
    master_kind = infer_facet_kind(master_points)
    overlap = build_facet_overlap(slave_points, master_points, tolerance=tolerance)
    dmat = np.zeros((len(slave_points), len(slave_points)), dtype=float)
    mmat = np.zeros((len(slave_points), len(master_points)), dtype=float)
    barycentric, weights = triangle_rule(quadrature_points)

    for pallet in overlap.pallets:
        for bary, weight in zip(barycentric, weights, strict=True):
            point = bary @ pallet.vertices
            slave_parent = inverse_map_2d(
                overlap.slave_polygon,
                slave_kind,
                point,
                tolerance=tolerance,
            )
            master_parent = inverse_map_2d(
                overlap.master_polygon,
                master_kind,
                point,
                tolerance=tolerance,
            )
            slave_shape = shape_values(slave_kind, slave_parent)
            master_shape = shape_values(master_kind, master_parent)
            factor = pallet.area * float(weight)
            dmat += factor * np.outer(slave_shape, slave_shape)
            mmat += factor * np.outer(slave_shape, master_shape)

    return LocalMortarWeights(dmat, mmat, overlap)
