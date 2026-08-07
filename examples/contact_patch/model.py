"""Small nonmatching frictionless contact-patch model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contact3d import (
    ContactPair,
    ContactSurface,
    CoupledEquilibriumProblem,
    DeadLoad,
    DirichletConstraints,
    LinearBoundaryPath,
    LinearPathValue,
    MortarContactInterface,
    NeoHookeanMaterial,
    Tet4Mesh,
)
from contact3d.scaling import ScaleAwareConvergenceOptions
from contact3d.solvers import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    AugmentedContactOptions,
    NewtonOptions,
)


@dataclass(frozen=True, slots=True)
class ContactPatchModel:
    """Two small TET4 bodies and one nonmatching mortar interface."""

    problem: CoupledEquilibriumProblem
    path: LinearBoundaryPath
    lower_nodes: np.ndarray
    upper_nodes: np.ndarray
    slave_nodes: np.ndarray
    master_nodes: np.ndarray
    support_nodes: np.ndarray
    tool_nodes: np.ndarray
    initial_separation: float


def _star_elements(offset: int) -> np.ndarray:
    triangles = (
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
    )
    return np.asarray(
        [(offset + 8, offset + a, offset + b, offset + c) for a, b, c in triangles],
        dtype=np.int64,
    )


def _lower_block() -> np.ndarray:
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.000],
            [1.0, 0.0, 1.004],
            [1.0, 1.0, 0.998],
            [0.0, 1.0, 1.003],
        ]
    )
    return np.vstack([corners, np.mean(corners, axis=0)])


def _upper_block() -> np.ndarray:
    footprint = np.array(
        [
            [0.14, -0.08],
            [1.14, 0.18],
            [0.84, 1.18],
            [-0.16, 0.86],
        ]
    )
    bottom_z = np.array([1.038, 1.044, 1.036, 1.042])
    bottom = np.column_stack([footprint, bottom_z])
    top = np.column_stack(
        [
            footprint + np.array([0.01, 0.0]),
            bottom_z + 1.0 + np.array([0.002, -0.001, 0.003, -0.002]),
        ]
    )
    corners = np.vstack([bottom, top])
    return np.vstack([corners, np.mean(corners, axis=0)])


def _minimum_reference_determinant(nodes: np.ndarray, elements: np.ndarray) -> float:
    determinants = []
    for element in elements:
        origin = nodes[element[0]]
        matrix = np.column_stack(
            [nodes[element[index]] - origin for index in (1, 2, 3)]
        )
        determinants.append(float(np.linalg.det(matrix)))
    return min(determinants)


def build_model() -> ContactPatchModel:
    """Build the deterministic nonmatching contact-patch problem."""

    lower = _lower_block()
    upper = _upper_block()
    nodes = np.vstack([lower, upper])
    elements = np.vstack([_star_elements(0), _star_elements(9)])
    mesh = Tet4Mesh(nodes, elements)

    slave_nodes = np.array([9, 10, 11, 12], dtype=np.int64)
    master_nodes = np.array([4, 5, 6, 7], dtype=np.int64)
    support_nodes = np.array([0, 1, 2, 3], dtype=np.int64)
    tool_nodes = np.array([13, 14, 15, 16], dtype=np.int64)
    slave = ContactSurface(
        nodes[slave_nodes],
        (np.array([0, 1, 2, 3], dtype=np.int64),),
        normal_sign=-1.0,
    )
    master = ContactSurface(
        nodes[master_nodes],
        (
            np.array([0, 1, 2], dtype=np.int64),
            np.array([0, 2, 3], dtype=np.int64),
        ),
    )
    pair = ContactPair(
        slave,
        master,
        normal_penalty=3200.0,
        search_distance=0.20,
        quadrature_points=7,
    )
    interface = MortarContactInterface(pair, slave_nodes, master_nodes)

    constrained_dofs: list[int] = []
    final_values: list[float] = []
    for node in support_nodes:
        for component in range(3):
            constrained_dofs.append(3 * int(node) + component)
            final_values.append(0.0)
    for node in tool_nodes:
        for component, value in enumerate((0.04, 0.0, -0.09)):
            constrained_dofs.append(3 * int(node) + component)
            final_values.append(value)
    constraints = DirichletConstraints(
        np.asarray(constrained_dofs, dtype=np.int64),
        np.asarray(final_values, dtype=float),
    )
    force = np.zeros(3 * mesh.node_count)
    force[3 * 17] = 0.50
    load = DeadLoad(force)
    material = NeoHookeanMaterial.from_young_poisson(210.0, 0.30)
    problem = CoupledEquilibriumProblem(mesh, material, constraints, load, (interface,))
    path = LinearBoundaryPath.proportional_mixed(
        problem,
        values=(
            LinearPathValue("tool_x", 0.0, 0.04),
            LinearPathValue("tool_z", 0.0, -0.09),
            LinearPathValue("dead_load_x", 0.0, 0.50),
        ),
    )
    separation = float(np.min(nodes[slave_nodes, 2]) - np.max(nodes[master_nodes, 2]))
    if separation <= 0.0:
        raise ValueError("contact-patch interfaces must be initially separated")
    if _minimum_reference_determinant(nodes, elements) <= 0.0:
        raise ValueError("contact-patch reference mesh contains an inverted tetrahedron")
    return ContactPatchModel(
        problem,
        path,
        lower,
        upper,
        slave_nodes,
        master_nodes,
        support_nodes,
        tool_nodes,
        separation,
    )


def solver_options() -> AdaptiveContactOptions:
    """Return the bounded production settings used by the example."""

    return AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.10,
            minimum_step=1.0 / 2048.0,
            maximum_step=0.10,
            cutback_factor=0.5,
            growth_factor=1.5,
            easy_newton_iterations=8,
            maximum_attempts=180,
        ),
        penalty=AdaptivePenaltyOptions(
            increase_factor=2.0,
            maximum_penalty=2.0e6,
            maximum_updates_per_step=4,
            normalized_penetration_target=2.0e-7,
            interface_local=True,
            minimum_scale_factor=0.25,
            maximum_scale_factor=1.0e3,
        ),
        augmented=AugmentedContactOptions(
            maximum_augmentations=20,
            gap_tolerance=1.0e-8,
            complementarity_tolerance=1.0e-7,
            projection_tolerance=1.0e-5,
            multiplier_tolerance=1.0e-8,
            event_policy="restart",
            newton=NewtonOptions(
                maximum_iterations=50,
                absolute_tolerance=1.0e-10,
                relative_tolerance=1.0e-10,
                maximum_line_search_iterations=24,
                minimum_step=2.0**-20,
            ),
        ),
        scaling=ScaleAwareConvergenceOptions(
            enabled=True,
            equilibrium_tolerance=1.0e-9,
            gap_tolerance=2.0e-7,
            complementarity_tolerance=1.0e-7,
            projection_tolerance=1.0e-7,
            multiplier_tolerance=1.0e-8,
        ),
    )
