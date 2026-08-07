"""Small nonmatching rotating-blocks large-sliding model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contact3d import (
    ContactPair,
    ContactSurface,
    CoupledEquilibriumProblem,
    DeadLoad,
    DirichletConstraints,
    MortarContactInterface,
    NeoHookeanMaterial,
    StagedRigidBodyBoundaryPath,
    Tet4Mesh,
)
from contact3d.scaling import ScaleAwareConvergenceOptions
from contact3d.solvers import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    AugmentedContactOptions,
    LinearSolverOptions,
    NewtonOptions,
)


@dataclass(frozen=True, slots=True)
class RotatingBlocksGeometry:
    """Dimensions and prescribed compression, translation, and rotation."""

    lower_minimum: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    lower_maximum: tuple[float, float, float] = (1.0, 1.0, 0.5)
    upper_minimum: tuple[float, float, float] = (-0.62, -0.32, 0.521)
    upper_maximum: tuple[float, float, float] = (0.68, 0.38, 0.821)
    pivot: tuple[float, float, float] = (0.03, 0.03, 0.521)
    compression: tuple[float, float, float] = (0.0, 0.0, -0.04)
    tangential_translation: tuple[float, float, float] = (0.10, 0.0, 0.0)
    final_angle: float = 0.5 * np.pi
    compression_end: float = 0.25
    search_distance: float = 0.12
    normal_penalty: float = 3200.0

    @property
    def initial_separation(self) -> float:
        """Return the initial distance between the two contact planes."""

        return self.upper_minimum[2] - self.lower_maximum[2]


@dataclass(frozen=True, slots=True)
class RotatingBlocksModel:
    """Two coarse TET4 blocks and one moving QUAD4 mortar interface."""

    geometry: RotatingBlocksGeometry
    problem: CoupledEquilibriumProblem
    path: StagedRigidBodyBoundaryPath
    lower_nodes: np.ndarray
    upper_nodes: np.ndarray
    lower_elements: np.ndarray
    upper_elements: np.ndarray
    fixed_nodes: np.ndarray
    controlled_nodes: np.ndarray
    slave_nodes: np.ndarray
    master_nodes: np.ndarray
    minimum_reference_determinant: float


LOWER_CELLS = (2, 2, 1)
UPPER_CELLS = (3, 2, 1)


def _node_index(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return (k * (ny + 1) + j) * (nx + 1) + i


def _structured_box(
    cells: tuple[int, int, int],
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = cells
    x_values = np.linspace(minimum[0], maximum[0], nx + 1)
    y_values = np.linspace(minimum[1], maximum[1], ny + 1)
    z_values = np.linspace(minimum[2], maximum[2], nz + 1)
    nodes = np.asarray(
        [
            (x, y, z)
            for z in z_values
            for y in y_values
            for x in x_values
        ],
        dtype=float,
    )
    elements: list[tuple[int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v000 = _node_index(i, j, k, nx, ny)
                v100 = _node_index(i + 1, j, k, nx, ny)
                v110 = _node_index(i + 1, j + 1, k, nx, ny)
                v010 = _node_index(i, j + 1, k, nx, ny)
                v001 = _node_index(i, j, k + 1, nx, ny)
                v101 = _node_index(i + 1, j, k + 1, nx, ny)
                v111 = _node_index(i + 1, j + 1, k + 1, nx, ny)
                v011 = _node_index(i, j + 1, k + 1, nx, ny)
                elements.extend(
                    (
                        (v000, v100, v110, v111),
                        (v000, v110, v010, v111),
                        (v000, v010, v011, v111),
                        (v000, v011, v001, v111),
                        (v000, v001, v101, v111),
                        (v000, v101, v100, v111),
                    )
                )
    return nodes, np.asarray(elements, dtype=np.int64)


def _surface_grid(
    cells: tuple[int, int, int],
    layer: int,
    *,
    offset: int = 0,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    nx, ny, nz = cells
    if not 0 <= layer <= nz:
        raise ValueError("surface layer is outside the structured box")
    nodes = np.asarray(
        [
            offset + _node_index(i, j, layer, nx, ny)
            for j in range(ny + 1)
            for i in range(nx + 1)
        ],
        dtype=np.int64,
    )
    facets: list[np.ndarray] = []
    for j in range(ny):
        for i in range(nx):
            v00 = j * (nx + 1) + i
            v10 = v00 + 1
            v01 = (j + 1) * (nx + 1) + i
            v11 = v01 + 1
            facets.append(np.asarray((v00, v10, v11, v01), dtype=np.int64))
    return nodes, tuple(facets)


def _reference_determinants(
    nodes: np.ndarray,
    elements: np.ndarray,
) -> np.ndarray:
    values = []
    for element in elements:
        origin = nodes[element[0]]
        matrix = np.column_stack(
            [nodes[element[index]] - origin for index in (1, 2, 3)]
        )
        values.append(float(np.linalg.det(matrix)))
    return np.asarray(values)


def _constraints(
    fixed_nodes: np.ndarray,
    controlled_nodes: np.ndarray,
) -> DirichletConstraints:
    nodes = np.concatenate((fixed_nodes, controlled_nodes))
    dofs = np.asarray(
        [3 * int(node) + component for node in nodes for component in range(3)],
        dtype=np.int64,
    )
    return DirichletConstraints(dofs, np.zeros(len(dofs)))


def build_model() -> RotatingBlocksModel:
    """Build the verification-sized v0.1 rotating-blocks problem."""

    geometry = RotatingBlocksGeometry()
    lower_nodes, lower_elements = _structured_box(
        LOWER_CELLS,
        geometry.lower_minimum,
        geometry.lower_maximum,
    )
    upper_nodes, upper_elements = _structured_box(
        UPPER_CELLS,
        geometry.upper_minimum,
        geometry.upper_maximum,
    )
    lower_count = len(lower_nodes)
    nodes = np.vstack((lower_nodes, upper_nodes))
    elements = np.vstack((lower_elements, upper_elements + lower_count))
    mesh = Tet4Mesh(nodes, elements)

    master_nodes, master_facets = _surface_grid(
        LOWER_CELLS,
        LOWER_CELLS[2],
    )
    slave_local_nodes, slave_facets = _surface_grid(UPPER_CELLS, 0)
    slave_nodes = slave_local_nodes + lower_count
    interface = MortarContactInterface(
        ContactPair(
            ContactSurface(
                nodes[slave_nodes],
                slave_facets,
                normal_sign=-1.0,
            ),
            ContactSurface(
                nodes[master_nodes],
                master_facets,
                normal_sign=1.0,
            ),
            normal_penalty=geometry.normal_penalty,
            search_distance=geometry.search_distance,
            quadrature_points=7,
        ),
        slave_nodes,
        master_nodes,
    )

    fixed_nodes, _ = _surface_grid(LOWER_CELLS, 0)
    controlled_nodes = np.arange(
        lower_count,
        lower_count + len(upper_nodes),
        dtype=np.int64,
    )
    problem = CoupledEquilibriumProblem(
        mesh,
        NeoHookeanMaterial.from_young_poisson(210.0, 0.30),
        _constraints(fixed_nodes, controlled_nodes),
        DeadLoad(np.zeros(3 * mesh.node_count)),
        (interface,),
    )
    path = StagedRigidBodyBoundaryPath.compression_then_rotation(
        problem,
        controlled_nodes,
        compression=np.asarray(geometry.compression),
        end_angle=geometry.final_angle,
        tangential_translation=np.asarray(geometry.tangential_translation),
        compression_end=geometry.compression_end,
        pivot=np.asarray(geometry.pivot),
        axis=np.asarray((0.0, 0.0, 1.0)),
    )
    minimum_determinant = float(
        np.min(_reference_determinants(nodes, elements))
    )
    if minimum_determinant <= 0.0:
        raise ValueError("rotating-blocks mesh contains an inverted TET4 element")
    if geometry.initial_separation <= 0.0:
        raise ValueError("rotating-blocks surfaces must start separated")
    if geometry.search_distance <= geometry.initial_separation:
        raise ValueError("contact search distance must exceed initial separation")
    return RotatingBlocksModel(
        geometry,
        problem,
        path,
        lower_nodes,
        upper_nodes,
        lower_elements,
        upper_elements,
        fixed_nodes,
        controlled_nodes,
        slave_nodes,
        master_nodes,
        minimum_determinant,
    )


def solver_options() -> AdaptiveContactOptions:
    """Return bounded production settings for the coarse example."""

    scaling = ScaleAwareConvergenceOptions(
        enabled=True,
        equilibrium_tolerance=1.0e-8,
        gap_tolerance=1.0e-7,
        complementarity_tolerance=1.0e-7,
        projection_tolerance=1.0e-5,
        multiplier_tolerance=1.0e-7,
    )
    maximum_augmentations = 32
    return AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=1.0 / 16.0,
            minimum_step=1.0 / 1024.0,
            maximum_step=1.0 / 8.0,
            easy_newton_iterations=maximum_augmentations,
            maximum_attempts=128,
        ),
        penalty=AdaptivePenaltyOptions(
            enabled=True,
            normalized_penetration_target=scaling.gap_tolerance,
            maximum_updates_per_step=4,
            interface_local=True,
        ),
        augmented=AugmentedContactOptions(
            maximum_augmentations=maximum_augmentations,
            gap_tolerance=1.0e-8,
            complementarity_tolerance=1.0e-7,
            projection_tolerance=1.0e-5,
            multiplier_tolerance=1.0e-8,
            event_policy="restart",
            newton=NewtonOptions(
                maximum_iterations=40,
                absolute_tolerance=1.0e-10,
                relative_tolerance=1.0e-10,
                linear_solver=LinearSolverOptions(
                    backend="auto",
                    dense_threshold=96,
                ),
            ),
        ),
        scaling=scaling,
    )
