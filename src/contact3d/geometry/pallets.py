"""Centroid-fan mortar pallet geometry and analytical derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clipping import FacetIntersectionLinearization, linearize_facet_intersection
from .model import FloatArray, MortarPallet


class PalletTopologyError(ValueError):
    """Raised when a centroid fan lies on a nonsmooth or degenerate geometry event."""


@dataclass(frozen=True, slots=True)
class SignedAreaLinearization:
    """Signed polygon area and its derivative with respect to all input DOFs."""

    area: float
    jacobian: FloatArray


@dataclass(frozen=True, slots=True)
class MortarPalletLinearization:
    """One centroid-fan pallet and its coordinate and area derivatives.

    ``vertex_jacobian`` has axes ``(pallet_vertex, component, input_dof)``.
    ``area_jacobian`` has one entry per input DOF.
    """

    pallet: MortarPallet
    vertex_jacobian: FloatArray
    signed_area: float
    area_jacobian: FloatArray


@dataclass(frozen=True, slots=True)
class PalletFanLinearization:
    """Centroid-fan pallets differentiated from one intersection polygon."""

    center: FloatArray
    center_jacobian: FloatArray
    pallets: tuple[MortarPalletLinearization, ...]
    polygon_area: float
    polygon_area_jacobian: FloatArray
    total_area: float
    total_area_jacobian: FloatArray

    @property
    def area_consistency_error(self) -> float:
        """Difference between polygon area and the sum of pallet areas."""

        return abs(self.total_area - self.polygon_area)

    @property
    def area_jacobian_consistency_error(self) -> float:
        """Derivative-level area decomposition error."""

        difference = self.total_area_jacobian - self.polygon_area_jacobian
        return float(np.max(np.abs(difference), initial=0.0))


@dataclass(frozen=True, slots=True)
class FacetPalletLinearization:
    """Full projected-intersection and centroid-fan derivative chain."""

    intersection: FacetIntersectionLinearization
    fan: PalletFanLinearization


def _cross2(first: FloatArray, second: FloatArray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _cross2_derivative(
    first: FloatArray,
    second: FloatArray,
    derivative_first: FloatArray,
    derivative_second: FloatArray,
) -> FloatArray:
    return (
        second[1] * derivative_first[0]
        - second[0] * derivative_first[1]
        - first[1] * derivative_second[0]
        + first[0] * derivative_second[1]
    )


def _validate_polygon_jacobian(
    polygon: FloatArray,
    polygon_jacobian: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    values = np.asarray(polygon, dtype=float)
    derivative = np.asarray(polygon_jacobian, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("polygon must have shape (vertex_count, 2)")
    if derivative.ndim != 3 or derivative.shape[:2] != values.shape:
        raise ValueError(
            "polygon Jacobian must have shape (vertex_count, 2, input_dof)"
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(derivative)):
        raise ValueError("polygon and Jacobian values must be finite")
    return values, derivative


def polygon_signed_area_linearized(
    polygon: FloatArray,
    polygon_jacobian: FloatArray,
) -> SignedAreaLinearization:
    """Differentiate the shoelace signed area of a moving polygon."""

    values, derivative = _validate_polygon_jacobian(polygon, polygon_jacobian)
    dof_count = derivative.shape[2]
    if len(values) < 3:
        return SignedAreaLinearization(0.0, np.zeros(dof_count, dtype=float))

    area = 0.0
    area_jacobian = np.zeros(dof_count, dtype=float)
    for index, first in enumerate(values):
        second_index = (index + 1) % len(values)
        second = values[second_index]
        area += 0.5 * _cross2(first, second)
        area_jacobian += 0.5 * _cross2_derivative(
            first,
            second,
            derivative[index],
            derivative[second_index],
        )
    return SignedAreaLinearization(float(area), area_jacobian)


def linearize_centroid_fan(
    polygon: FloatArray,
    polygon_jacobian: FloatArray,
    *,
    tolerance: float = 1.0e-14,
) -> PalletFanLinearization:
    """Differentiate the centroid-fan pallets of one frozen intersection polygon.

    The polygon must be counterclockwise. Empty polygons represent a smooth
    no-overlap branch and return an empty fan. One- and two-vertex states, inverted
    polygons, and zero-area pallets are treated as outer geometry events.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    values, derivative = _validate_polygon_jacobian(polygon, polygon_jacobian)
    dof_count = derivative.shape[2]
    if len(values) == 0:
        zeros = np.zeros(dof_count, dtype=float)
        return PalletFanLinearization(
            center=np.zeros(2, dtype=float),
            center_jacobian=np.zeros((2, dof_count), dtype=float),
            pallets=(),
            polygon_area=0.0,
            polygon_area_jacobian=zeros.copy(),
            total_area=0.0,
            total_area_jacobian=zeros,
        )
    if len(values) < 3:
        raise PalletTopologyError(
            "intersection polygon has fewer than three vertices"
        )

    polygon_area = polygon_signed_area_linearized(values, derivative)
    if polygon_area.area <= tolerance:
        raise PalletTopologyError(
            "intersection polygon is inverted or has degenerate signed area"
        )

    center = np.mean(values, axis=0)
    center_jacobian = np.mean(derivative, axis=0)
    pallets: list[MortarPalletLinearization] = []
    total_area = 0.0
    total_area_jacobian = np.zeros(dof_count, dtype=float)

    for index, first in enumerate(values):
        second_index = (index + 1) % len(values)
        second = values[second_index]
        derivative_first = derivative[index]
        derivative_second = derivative[second_index]
        first_relative = first - center
        second_relative = second - center
        derivative_first_relative = derivative_first - center_jacobian
        derivative_second_relative = derivative_second - center_jacobian

        signed_double_area = _cross2(first_relative, second_relative)
        if signed_double_area <= 2.0 * tolerance:
            raise PalletTopologyError(
                "centroid fan contains a degenerate or inverted pallet"
            )
        signed_area = 0.5 * signed_double_area
        area_jacobian = 0.5 * _cross2_derivative(
            first_relative,
            second_relative,
            derivative_first_relative,
            derivative_second_relative,
        )
        vertices = np.vstack([center, first, second])
        vertex_jacobian = np.stack(
            [center_jacobian, derivative_first, derivative_second],
            axis=0,
        )
        pallets.append(
            MortarPalletLinearization(
                pallet=MortarPallet(vertices, signed_area),
                vertex_jacobian=vertex_jacobian,
                signed_area=signed_area,
                area_jacobian=area_jacobian,
            )
        )
        total_area += signed_area
        total_area_jacobian += area_jacobian

    return PalletFanLinearization(
        center=center,
        center_jacobian=center_jacobian,
        pallets=tuple(pallets),
        polygon_area=polygon_area.area,
        polygon_area_jacobian=polygon_area.jacobian,
        total_area=float(total_area),
        total_area_jacobian=total_area_jacobian,
    )


def linearize_facet_pallets(
    slave_points: FloatArray,
    master_points: FloatArray,
    *,
    tolerance: float = 1.0e-12,
    event_tolerance: float | None = None,
) -> FacetPalletLinearization:
    """Differentiate projected clipping and centroid-fan geometry for a facet pair."""

    intersection = linearize_facet_intersection(
        slave_points,
        master_points,
        tolerance=tolerance,
        event_tolerance=event_tolerance,
    )
    fan = linearize_centroid_fan(
        intersection.intersection_polygon,
        intersection.intersection_jacobian,
        tolerance=tolerance,
    )
    return FacetPalletLinearization(intersection, fan)
