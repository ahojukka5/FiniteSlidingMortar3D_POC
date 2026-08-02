"""Deterministic model factory for the rotating-blocks contact benchmark."""

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


@dataclass(frozen=True, slots=True)
class RotatingBlocksGeometry:
    """Physical geometry and prescribed motion shared by every mesh profile."""

    lower_minimum: tuple[float, float, float] = (-1.0, -1.0, 0.0)
    lower_maximum: tuple[float, float, float] = (1.0, 1.0, 0.5)
    upper_minimum: tuple[float, float, float] = (-0.65, -0.35, 0.52)
    upper_maximum: tuple[float, float, float] = (0.65, 0.35, 0.82)
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.52)
    compression: tuple[float, float, float] = (0.0, 0.0, -0.04)
    tangential_translation: tuple[float, float, float] = (0.10, 0.0, 0.0)
    final_angle: float = 0.5 * np.pi
    compression_end: float = 0.25
    search_distance: float = 0.12
    normal_penalty: float = 3200.0

    @property
    def initial_separation(self) -> float:
        return self.upper_minimum[2] - self.lower_maximum[2]


@dataclass(frozen=True, slots=True)
class RotatingBlocksProfile:
    """Mesh-resolution profile without changing the physical benchmark."""

    name: str
    lower_cells: tuple[int, int, int]
    upper_cells: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.name not in ("quick", "full"):
            raise ValueError("rotating-blocks profile name must be 'quick' or 'full'")
        if any(value <= 0 for value in (*self.lower_cells, *self.upper_cells)):
            raise ValueError("rotating-blocks mesh cell counts must be positive")


QUICK_PROFILE = RotatingBlocksProfile("quick", (4, 4, 2), (3, 2, 1))
FULL_PROFILE = RotatingBlocksProfile("full", (8, 8, 4), (3, 2, 1))
PROFILES = {profile.name: profile for profile in (QUICK_PROFILE, FULL_PROFILE)}


@dataclass(frozen=True, slots=True)
class RotatingBlocksModel:
    """Complete reusable geometry, contact, constraints, and motion definition."""

    profile: RotatingBlocksProfile
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

    @property
    def initial_separation(self) -> float:
        return self.geometry.initial_separation


def rotating_blocks_profile(name: str) -> RotatingBlocksProfile:
    """Return one named deterministic mesh profile."""

    try:
        return PROFILES[str(name)]
    except KeyError as error:
        raise ValueError("rotating-blocks profile must be 'quick' or 'full'") from error


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
            facets.append(np.asarray([v00, v10, v11, v01], dtype=np.int64))
    return nodes, tuple(facets)


def reference_determinants(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Return the signed reference determinant of every TET4 element."""

    values = []
    for element in np.asarray(elements, dtype=np.int64):
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
    nodes = np.concatenate([fixed_nodes, controlled_nodes])
    dofs = np.asarray(
        [3 * int(node) + component for node in nodes for component in range(3)],
        dtype=np.int64,
    )
    return DirichletConstraints(dofs, np.zeros(len(dofs)))


def _validate_model(model: RotatingBlocksModel) -> None:
    geometry = model.geometry
    problem = model.problem
    if model.minimum_reference_determinant <= 0.0:
        raise ValueError("rotating-blocks mesh contains an inverted TET4 element")
    if model.initial_separation <= 0.0:
        raise ValueError("rotating-blocks contact surfaces must start separated")
    if geometry.search_distance <= model.initial_separation:
        raise ValueError("contact search distance must exceed the initial separation")
    if np.intersect1d(model.fixed_nodes, model.controlled_nodes).size:
        raise ValueError("fixed and controlled rotating-blocks nodes must be disjoint")

    expected_dofs = np.asarray(
        [
            3 * int(node) + component
            for node in np.concatenate([model.fixed_nodes, model.controlled_nodes])
            for component in range(3)
        ],
        dtype=np.int64,
    )
    if not np.array_equal(problem.constraints.dofs, expected_dofs):
        raise ValueError("rotating-blocks constraints do not match fixed and controlled nodes")
    if np.any(problem.constraints.values):
        raise ValueError("rotating-blocks reference prescribed values must be zero")

    interface = problem.interfaces[0]
    interface.validate_for(problem.mesh)
    if any(len(facet) != 4 for facet in interface.pair.slave.facets):
        raise ValueError("rotating-blocks slave surface must contain only QUAD4 facets")
    if any(len(facet) != 4 for facet in interface.pair.master.facets):
        raise ValueError("rotating-blocks master surface must contain only QUAD4 facets")
    if interface.pair.slave.node_count == interface.pair.master.node_count:
        raise ValueError("rotating-blocks contact surfaces must be nonmatching")

    lower_count = len(model.lower_nodes)
    expected_controlled = np.arange(
        lower_count,
        lower_count + len(model.upper_nodes),
        dtype=np.int64,
    )
    if not np.array_equal(model.controlled_nodes, expected_controlled):
        raise ValueError("every upper-block node must be controlled")
    if not np.allclose(
        problem.mesh.reference_nodes[model.slave_nodes, 2],
        geometry.upper_minimum[2],
    ):
        raise ValueError("slave nodes must lie on the upper-block bottom surface")
    if not np.allclose(
        problem.mesh.reference_nodes[model.master_nodes, 2],
        geometry.lower_maximum[2],
    ):
        raise ValueError("master nodes must lie on the lower-block top surface")


def build_rotating_blocks_model(
    profile: str | RotatingBlocksProfile = "quick",
    *,
    geometry: RotatingBlocksGeometry | None = None,
) -> RotatingBlocksModel:
    """Build one deterministic rotating-blocks benchmark model."""

    selected = rotating_blocks_profile(profile) if isinstance(profile, str) else profile
    if selected.name not in PROFILES:
        raise ValueError("custom rotating-blocks profiles must use a known profile name")
    geometry = RotatingBlocksGeometry() if geometry is None else geometry

    lower_nodes, lower_elements = _structured_box(
        selected.lower_cells,
        geometry.lower_minimum,
        geometry.lower_maximum,
    )
    upper_nodes, upper_elements = _structured_box(
        selected.upper_cells,
        geometry.upper_minimum,
        geometry.upper_maximum,
    )
    lower_count = len(lower_nodes)
    nodes = np.vstack([lower_nodes, upper_nodes])
    elements = np.vstack([lower_elements, upper_elements + lower_count])
    mesh = Tet4Mesh(nodes, elements)

    master_nodes, master_facets = _surface_grid(
        selected.lower_cells,
        selected.lower_cells[2],
    )
    slave_local_nodes, slave_facets = _surface_grid(selected.upper_cells, 0)
    slave_nodes = slave_local_nodes + lower_count
    slave = ContactSurface(nodes[slave_nodes], slave_facets, normal_sign=-1.0)
    master = ContactSurface(nodes[master_nodes], master_facets, normal_sign=1.0)
    interface = MortarContactInterface(
        ContactPair(
            slave,
            master,
            normal_penalty=geometry.normal_penalty,
            search_distance=geometry.search_distance,
            quadrature_points=7,
        ),
        slave_nodes,
        master_nodes,
    )

    fixed_nodes, _ = _surface_grid(selected.lower_cells, 0)
    controlled_nodes = np.arange(
        lower_count,
        lower_count + len(upper_nodes),
        dtype=np.int64,
    )
    constraints = _constraints(fixed_nodes, controlled_nodes)
    problem = CoupledEquilibriumProblem(
        mesh,
        NeoHookeanMaterial.from_young_poisson(210.0, 0.30),
        constraints,
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
        axis=np.array([0.0, 0.0, 1.0]),
    )

    determinants = reference_determinants(nodes, elements)
    model = RotatingBlocksModel(
        profile=selected,
        geometry=geometry,
        problem=problem,
        path=path,
        lower_nodes=lower_nodes,
        upper_nodes=upper_nodes,
        lower_elements=lower_elements,
        upper_elements=upper_elements,
        fixed_nodes=fixed_nodes,
        controlled_nodes=controlled_nodes,
        slave_nodes=slave_nodes,
        master_nodes=master_nodes,
        minimum_reference_determinant=float(np.min(determinants)),
    )
    _validate_model(model)
    return model
