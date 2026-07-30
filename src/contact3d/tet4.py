"""Total-Lagrangian finite-strain TET4 elements and dense mesh assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bulk_material import (
    BulkGeometryError,
    NeoHookeanMaterial,
    evaluate_neo_hookean,
)
from .model import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class Tet4Reference:
    """Reference geometry of one positively oriented linear tetrahedron."""

    reference_nodes: FloatArray
    tolerance: float = field(default=1.0e-12, repr=False)
    shape_gradients: FloatArray = field(init=False)
    volume: float = field(init=False)

    def __post_init__(self) -> None:
        nodes = np.asarray(self.reference_nodes, dtype=float)
        if nodes.shape != (4, 3):
            raise ValueError("TET4 reference_nodes must have shape (4, 3)")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("TET4 reference_nodes must be finite")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")

        reference_jacobian = np.column_stack(
            [nodes[1] - nodes[0], nodes[2] - nodes[0], nodes[3] - nodes[0]]
        )
        determinant = float(np.linalg.det(reference_jacobian))
        if determinant <= self.tolerance:
            raise BulkGeometryError(
                "TET4 reference geometry must be positively oriented and nondegenerate"
            )

        inverse = np.linalg.inv(reference_jacobian)
        gradients = np.empty((4, 3), dtype=float)
        gradients[1:] = inverse
        gradients[0] = -np.sum(gradients[1:], axis=0)

        object.__setattr__(self, "reference_nodes", nodes.copy())
        object.__setattr__(self, "shape_gradients", gradients)
        object.__setattr__(self, "volume", determinant / 6.0)


@dataclass(frozen=True, slots=True)
class Tet4Evaluation:
    """Finite-strain element energy, residual, and consistent tangent."""

    current_nodes: FloatArray
    deformation_gradient: FloatArray
    jacobian: float
    energy_density: float
    energy: float
    first_piola: FloatArray
    material_tangent: FloatArray
    internal_force: FloatArray
    tangent: FloatArray

    @property
    def force_balance(self) -> FloatArray:
        """Net internal force of the isolated element."""

        return np.sum(self.internal_force, axis=0)

    @property
    def moment_balance(self) -> FloatArray:
        """Net internal moment about the current origin."""

        return np.sum(np.cross(self.current_nodes, self.internal_force), axis=0)


@dataclass(frozen=True, slots=True)
class Tet4Mesh:
    """Reference mesh containing only positively oriented TET4 elements."""

    reference_nodes: FloatArray
    elements: IntArray
    tolerance: float = field(default=1.0e-12, repr=False)
    element_references: tuple[Tet4Reference, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        nodes = np.asarray(self.reference_nodes, dtype=float)
        elements = np.asarray(self.elements, dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) == 0:
            raise ValueError("mesh reference_nodes must have shape (node_count, 3)")
        if elements.ndim != 2 or elements.shape[1] != 4 or len(elements) == 0:
            raise ValueError("mesh elements must have shape (element_count, 4)")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("mesh reference_nodes must be finite")
        if np.any(elements < 0) or np.any(elements >= len(nodes)):
            raise ValueError("mesh element node index is out of range")
        if any(len(np.unique(element)) != 4 for element in elements):
            raise ValueError("mesh TET4 element contains duplicate nodes")

        references = tuple(
            Tet4Reference(nodes[element], tolerance=self.tolerance) for element in elements
        )
        object.__setattr__(self, "reference_nodes", nodes.copy())
        object.__setattr__(self, "elements", elements.copy())
        object.__setattr__(self, "element_references", references)

    @property
    def node_count(self) -> int:
        return len(self.reference_nodes)

    @property
    def element_count(self) -> int:
        return len(self.elements)

    @property
    def reference_volume(self) -> float:
        return float(sum(reference.volume for reference in self.element_references))


@dataclass(frozen=True, slots=True)
class Tet4MeshEvaluation:
    """Assembled finite-strain TET4 mesh energy, residual, and tangent."""

    current_nodes: FloatArray
    energy: float
    residual: FloatArray
    tangent: FloatArray
    element_evaluations: tuple[Tet4Evaluation, ...]

    @property
    def minimum_jacobian(self) -> float:
        return min(evaluation.jacobian for evaluation in self.element_evaluations)

    @property
    def force_balance(self) -> FloatArray:
        return np.sum(self.residual, axis=0)

    @property
    def moment_balance(self) -> FloatArray:
        return np.sum(np.cross(self.current_nodes, self.residual), axis=0)


def _validated_displacement(displacement: FloatArray, node_count: int) -> FloatArray:
    values = np.asarray(displacement, dtype=float)
    if values.shape == (3 * node_count,):
        values = values.reshape((node_count, 3))
    if values.shape != (node_count, 3):
        raise ValueError("displacement must have shape (node_count, 3) or (3*node_count,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("displacement must be finite")
    return values


def tet4_deformation_gradient(
    reference: Tet4Reference,
    displacement: FloatArray,
) -> FloatArray:
    """Return the constant deformation gradient of one TET4 element."""

    values = _validated_displacement(displacement, 4)
    current_nodes = reference.reference_nodes + values
    return current_nodes.T @ reference.shape_gradients


def evaluate_tet4(
    reference: Tet4Reference,
    displacement: FloatArray,
    material: NeoHookeanMaterial,
    *,
    tolerance: float = 1.0e-12,
) -> Tet4Evaluation:
    """Evaluate one total-Lagrangian finite-strain TET4 element."""

    values = _validated_displacement(displacement, 4)
    current_nodes = reference.reference_nodes + values
    deformation_gradient = current_nodes.T @ reference.shape_gradients
    response = evaluate_neo_hookean(
        deformation_gradient,
        material,
        tolerance=tolerance,
    )

    internal_force = reference.volume * np.einsum(
        "ij,aj->ai",
        response.first_piola,
        reference.shape_gradients,
    )
    tangent_blocks = reference.volume * np.einsum(
        "ijkl,aj,bl->aibk",
        response.tangent,
        reference.shape_gradients,
        reference.shape_gradients,
    )
    tangent = tangent_blocks.reshape((12, 12))
    return Tet4Evaluation(
        current_nodes=current_nodes,
        deformation_gradient=deformation_gradient,
        jacobian=response.jacobian,
        energy_density=response.energy_density,
        energy=reference.volume * response.energy_density,
        first_piola=response.first_piola,
        material_tangent=response.tangent,
        internal_force=internal_force,
        tangent=tangent,
    )


def evaluate_tet4_mesh(
    mesh: Tet4Mesh,
    displacement: FloatArray,
    material: NeoHookeanMaterial,
    *,
    tolerance: float = 1.0e-12,
) -> Tet4MeshEvaluation:
    """Assemble the energy, internal residual, and dense tangent of a TET4 mesh."""

    values = _validated_displacement(displacement, mesh.node_count)
    residual = np.zeros((mesh.node_count, 3), dtype=float)
    tangent = np.zeros((3 * mesh.node_count, 3 * mesh.node_count), dtype=float)
    evaluations: list[Tet4Evaluation] = []
    energy = 0.0

    for element, reference in zip(mesh.elements, mesh.element_references, strict=True):
        evaluation = evaluate_tet4(
            reference,
            values[element],
            material,
            tolerance=tolerance,
        )
        evaluations.append(evaluation)
        energy += evaluation.energy
        residual[element] += evaluation.internal_force
        dofs = np.array(
            [3 * int(node) + component for node in element for component in range(3)],
            dtype=np.int64,
        )
        tangent[np.ix_(dofs, dofs)] += evaluation.tangent

    return Tet4MeshEvaluation(
        current_nodes=mesh.reference_nodes + values,
        energy=float(energy),
        residual=residual,
        tangent=tangent,
        element_evaluations=tuple(evaluations),
    )
