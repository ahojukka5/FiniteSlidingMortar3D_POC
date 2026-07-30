from __future__ import annotations

import numpy as np
import pytest

from contact3d.bulk import (
    BulkGeometryError,
    NeoHookeanMaterial,
    Tet4Mesh,
    Tet4Reference,
    evaluate_neo_hookean,
    evaluate_tet4,
    evaluate_tet4_mesh,
    numerical_neo_hookean_tangent,
    numerical_tet4_mesh_tangent,
    numerical_tet4_tangent,
)


def _material() -> NeoHookeanMaterial:
    return NeoHookeanMaterial.from_young_poisson(young_modulus=210.0, poisson_ratio=0.3)


def _reference() -> Tet4Reference:
    return Tet4Reference(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.2, 0.1, 0.0],
                [0.1, 1.1, 0.05],
                [0.0, 0.2, 0.9],
            ]
        )
    )


def _affine_displacement(
    nodes: np.ndarray,
    deformation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return nodes @ deformation.T + translation - nodes


def _cube_star_mesh() -> Tet4Mesh:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )
    triangles = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ]
    elements = np.array([(8, *triangle) for triangle in triangles], dtype=np.int64)
    return Tet4Mesh(nodes, elements)


def test_material_tangent_matches_centered_difference() -> None:
    deformation = np.array(
        [[1.12, 0.13, -0.04], [0.02, 0.93, 0.08], [0.03, -0.06, 1.07]]
    )
    response = evaluate_neo_hookean(deformation, _material())
    numerical = numerical_neo_hookean_tangent(deformation, _material())
    relative_error = np.linalg.norm(response.tangent - numerical) / np.linalg.norm(numerical)
    assert relative_error < 2.0e-9


def test_element_tangent_and_energy_gradient_match_numerical_oracles() -> None:
    reference = _reference()
    deformation = np.array(
        [[1.08, 0.11, -0.03], [0.04, 0.96, 0.07], [0.01, -0.05, 1.09]]
    )
    displacement = _affine_displacement(
        reference.reference_nodes,
        deformation,
        np.array([0.2, -0.1, 0.05]),
    )
    evaluation = evaluate_tet4(reference, displacement, _material())
    numerical = numerical_tet4_tangent(reference, displacement, _material())
    relative_error = np.linalg.norm(evaluation.tangent - numerical) / np.linalg.norm(numerical)
    assert relative_error < 3.0e-9

    step = 2.0e-7
    energy_gradient = np.zeros(12)
    for column in range(12):
        plus = displacement.copy().ravel()
        minus = displacement.copy().ravel()
        plus[column] += step
        minus[column] -= step
        energy_gradient[column] = (
            evaluate_tet4(reference, plus, _material()).energy
            - evaluate_tet4(reference, minus, _material()).energy
        ) / (2.0 * step)
    np.testing.assert_allclose(energy_gradient, evaluation.internal_force.ravel(), atol=2.0e-8)


def test_element_preserves_rigid_motion_and_balance() -> None:
    reference = _reference()
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    displacement = _affine_displacement(
        reference.reference_nodes,
        rotation,
        np.array([0.4, -0.2, 0.3]),
    )
    evaluation = evaluate_tet4(reference, displacement, _material())
    assert abs(evaluation.energy) < 2.0e-13
    assert np.linalg.norm(evaluation.internal_force) < 2.0e-12
    assert np.linalg.norm(evaluation.force_balance) < 2.0e-13
    assert np.linalg.norm(evaluation.moment_balance) < 2.0e-13


def test_cube_star_affine_patch_has_zero_interior_residual() -> None:
    mesh = _cube_star_mesh()
    deformation = np.array(
        [[1.08, 0.12, -0.04], [0.03, 0.94, 0.08], [0.02, -0.05, 1.11]]
    )
    displacement = _affine_displacement(
        mesh.reference_nodes,
        deformation,
        np.array([0.2, -0.1, 0.05]),
    )
    evaluation = evaluate_tet4_mesh(mesh, displacement, _material())

    assert abs(mesh.reference_volume - 1.0) < 2.0e-14
    assert np.linalg.norm(evaluation.residual[8]) < 5.0e-13
    assert np.linalg.norm(evaluation.force_balance) < 5.0e-13
    assert np.linalg.norm(evaluation.moment_balance) < 5.0e-13
    assert max(
        np.linalg.norm(element.deformation_gradient - deformation)
        for element in evaluation.element_evaluations
    ) < 2.0e-15


def test_assembled_tangent_matches_centered_difference() -> None:
    mesh = _cube_star_mesh()
    deformation = np.array(
        [[1.04, 0.06, -0.02], [0.01, 0.98, 0.03], [0.02, -0.01, 1.06]]
    )
    displacement = _affine_displacement(mesh.reference_nodes, deformation, np.zeros(3))
    evaluation = evaluate_tet4_mesh(mesh, displacement, _material())
    numerical = numerical_tet4_mesh_tangent(
        mesh,
        displacement,
        _material(),
        relative_step=4.0e-7,
    )
    relative_error = np.linalg.norm(evaluation.tangent - numerical) / np.linalg.norm(numerical)
    assert relative_error < 5.0e-9


def test_invalid_reference_and_current_geometry_are_rejected() -> None:
    with pytest.raises(BulkGeometryError, match="positively oriented"):
        Tet4Reference(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        )

    reference = Tet4Reference(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    inverted = _affine_displacement(
        reference.reference_nodes,
        np.diag([-1.0, 1.0, 1.0]),
        np.zeros(3),
    )
    with pytest.raises(BulkGeometryError, match="singular or inverted"):
        evaluate_tet4(reference, inverted, _material())
