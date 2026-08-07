"""Internal centered-difference verification oracles for bulk mechanics."""

from __future__ import annotations

import numpy as np

from .bulk_material import NeoHookeanMaterial, evaluate_neo_hookean
from .model import FloatArray
from .tet4 import (
    Tet4Mesh,
    Tet4Reference,
    _validated_displacement,
    evaluate_tet4,
    evaluate_tet4_mesh,
)


def numerical_neo_hookean_tangent(
    deformation_gradient: FloatArray,
    material: NeoHookeanMaterial,
    *,
    relative_step: float = 2.0e-7,
) -> FloatArray:
    """Return a centered-difference first-Piola derivative for verification."""

    deformation = np.asarray(deformation_gradient, dtype=float)
    if deformation.shape != (3, 3):
        raise ValueError("deformation_gradient must have shape (3, 3)")
    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    tangent = np.zeros((3, 3, 3, 3), dtype=float)
    for row in range(3):
        for column in range(3):
            step = relative_step * max(1.0, abs(float(deformation[row, column])))
            plus = deformation.copy()
            minus = deformation.copy()
            plus[row, column] += step
            minus[row, column] -= step
            tangent[:, :, row, column] = (
                evaluate_neo_hookean(plus, material).first_piola
                - evaluate_neo_hookean(minus, material).first_piola
            ) / (2.0 * step)
    return tangent


def numerical_tet4_tangent(
    reference: Tet4Reference,
    displacement: FloatArray,
    material: NeoHookeanMaterial,
    *,
    relative_step: float = 2.0e-7,
) -> FloatArray:
    """Return a centered-difference element residual tangent for verification."""

    values = _validated_displacement(displacement, 4).copy()
    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    tangent = np.zeros((12, 12), dtype=float)
    current = reference.reference_nodes + values
    for column in range(12):
        node, component = divmod(column, 3)
        step = relative_step * max(1.0, abs(float(current[node, component])))
        plus = values.copy()
        minus = values.copy()
        plus[node, component] += step
        minus[node, component] -= step
        tangent[:, column] = (
            evaluate_tet4(reference, plus, material).internal_force.ravel()
            - evaluate_tet4(reference, minus, material).internal_force.ravel()
        ) / (2.0 * step)
    return tangent


def numerical_tet4_mesh_tangent(
    mesh: Tet4Mesh,
    displacement: FloatArray,
    material: NeoHookeanMaterial,
    *,
    relative_step: float = 2.0e-7,
) -> FloatArray:
    """Return a centered-difference assembled residual tangent for verification."""

    values = _validated_displacement(displacement, mesh.node_count).copy()
    if relative_step <= 0.0:
        raise ValueError("relative_step must be positive")
    ndof = 3 * mesh.node_count
    tangent = np.zeros((ndof, ndof), dtype=float)
    current = mesh.reference_nodes + values
    for column in range(ndof):
        node, component = divmod(column, 3)
        step = relative_step * max(1.0, abs(float(current[node, component])))
        plus = values.copy()
        minus = values.copy()
        plus[node, component] += step
        minus[node, component] -= step
        tangent[:, column] = (
            evaluate_tet4_mesh(mesh, plus, material).residual.ravel()
            - evaluate_tet4_mesh(mesh, minus, material).residual.ravel()
        ) / (2.0 * step)
    return tangent
