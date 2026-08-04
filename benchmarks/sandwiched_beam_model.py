"""Deterministic model family for the sandwiched-beam bending benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contact3d import (
    ContactPair,
    ContactSurface,
    CoupledEquilibriumProblem,
    CoupledPathState,
    DeadLoad,
    DirichletConstraints,
    EquilibriumProblem,
    MortarContactInterface,
    NeoHookeanMaterial,
    Tet4Mesh,
    with_coupled_boundary_data,
)


@dataclass(frozen=True, slots=True)
class SandwichedBeamGeometry:
    """Physical configuration shared by every mesh level and slave choice."""

    length: float = 10.0
    width: float = 1.0
    beam_thickness: float = 1.0
    ambient_pressure: float = 0.1
    end_moment: float = 0.12
    compression_end: float = 0.25
    search_distance: float = 0.15
    normal_penalty: float = 200.0
    young_modulus: float = 1.0
    poisson_ratio: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.length,
            self.width,
            self.beam_thickness,
            self.ambient_pressure,
            self.end_moment,
            self.search_distance,
            self.normal_penalty,
            self.young_modulus,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "sandwiched-beam physical values must be finite and positive"
            )
        if not 0.0 < self.compression_end < 1.0:
            raise ValueError("compression_end must lie strictly inside (0, 1)")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie inside the isotropic range")

    @property
    def interface_z(self) -> float:
        return self.beam_thickness

    @property
    def total_thickness(self) -> float:
        return 2.0 * self.beam_thickness

    @property
    def loaded_area(self) -> float:
        return self.length * self.width


@dataclass(frozen=True, slots=True)
class SandwichedBeamLevel:
    """One nonmatching contact mesh and its monolithic reference mesh."""

    name: str
    lower_cells: tuple[int, int, int]
    upper_cells: tuple[int, int, int]
    reference_cells: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.name not in ("coarse", "medium", "fine"):
            raise ValueError(
                "sandwiched-beam level must be coarse, medium, or fine"
            )
        values = (*self.lower_cells, *self.upper_cells, *self.reference_cells)
        if any(value <= 0 for value in values):
            raise ValueError("sandwiched-beam cell counts must be positive")
        lower_surface = (self.lower_cells[0] + 1) * (self.lower_cells[1] + 1)
        upper_surface = (self.upper_cells[0] + 1) * (self.upper_cells[1] + 1)
        if lower_surface == upper_surface:
            raise ValueError("sandwiched-beam contact meshes must be nonmatching")
        if self.reference_cells[2] % 2:
            raise ValueError(
                "reference mesh must contain a layer at the beam interface"
            )


COARSE_LEVEL = SandwichedBeamLevel("coarse", (4, 1, 1), (5, 2, 1), (6, 2, 2))
MEDIUM_LEVEL = SandwichedBeamLevel("medium", (8, 2, 2), (11, 3, 2), (12, 3, 4))
FINE_LEVEL = SandwichedBeamLevel("fine", (16, 4, 3), (21, 5, 3), (24, 5, 6))
LEVELS = {
    level.name: level for level in (COARSE_LEVEL, MEDIUM_LEVEL, FINE_LEVEL)
}


@dataclass(frozen=True, slots=True)
class SandwichedBeamLoadPath:
    """Ramp ambient pressure first and then apply an end moment."""

    pressure_force: np.ndarray
    moment_force: np.ndarray
    compression_end: float

    def __post_init__(self) -> None:
        pressure = np.asarray(self.pressure_force, dtype=float)
        moment = np.asarray(self.moment_force, dtype=float)
        if pressure.ndim != 1 or moment.shape != pressure.shape:
            raise ValueError(
                "sandwiched-beam path forces must be aligned flat vectors"
            )
        if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(moment)):
            raise ValueError("sandwiched-beam path forces must be finite")
        if not 0.0 < self.compression_end < 1.0:
            raise ValueError("compression_end must lie strictly inside (0, 1)")
        object.__setattr__(self, "pressure_force", pressure.copy())
        object.__setattr__(self, "moment_force", moment.copy())

    @property
    def end_parameter(self) -> float:
        return 1.0

    def phase_name(self, parameter: float) -> str:
        value = self._parameter(parameter)
        return "compression" if value < self.compression_end else "bending"

    def scales(self, parameter: float) -> tuple[float, float, int, float]:
        value = self._parameter(parameter)
        if value < self.compression_end:
            pressure = value / self.compression_end
            return pressure, 0.0, 0, pressure
        bending = (value - self.compression_end) / (1.0 - self.compression_end)
        return 1.0, bending, 1, bending

    def force(self, parameter: float) -> np.ndarray:
        pressure, moment, _, _ = self.scales(parameter)
        return pressure * self.pressure_force + moment * self.moment_force

    def load(self, parameter: float) -> DeadLoad:
        return DeadLoad(self.force(parameter))

    def evaluate(
        self,
        problem: CoupledEquilibriumProblem,
        parameter: float,
    ) -> CoupledPathState:
        value = self._parameter(parameter)
        pressure, moment, phase, phase_parameter = self.scales(value)
        force = pressure * self.pressure_force + moment * self.moment_force
        load = DeadLoad(force)
        updated = with_coupled_boundary_data(problem, problem.constraints, load)
        return CoupledPathState(
            value,
            updated,
            1.0,
            updated.constraints.dofs,
            updated.constraints.values,
            force,
            (
                ("phase_index", float(phase)),
                ("phase_parameter", phase_parameter),
                ("pressure_scale", pressure),
                ("moment_scale", moment),
            ),
        )

    @staticmethod
    def _parameter(parameter: float) -> float:
        value = float(parameter)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "sandwiched-beam path parameter must lie inside [0, 1]"
            )
        return value


@dataclass(frozen=True, slots=True)
class SandwichedBeamModel:
    """Contact model, monolithic reference, paths, and mesh metadata."""

    level: SandwichedBeamLevel
    geometry: SandwichedBeamGeometry
    slave_side: str
    problem: CoupledEquilibriumProblem
    path: SandwichedBeamLoadPath
    reference_problem: EquilibriumProblem
    reference_path: SandwichedBeamLoadPath
    lower_nodes: np.ndarray
    upper_nodes: np.ndarray
    lower_elements: np.ndarray
    upper_elements: np.ndarray
    lower_interface_nodes: np.ndarray
    upper_interface_nodes: np.ndarray
    slave_nodes: np.ndarray
    master_nodes: np.ndarray
    minimum_reference_determinant: float
    reference_minimum_determinant: float


def sandwiched_beam_level(name: str) -> SandwichedBeamLevel:
    """Return one named deterministic refinement level."""

    try:
        return LEVELS[str(name)]
    except KeyError as error:
        raise ValueError(
            "sandwiched-beam level must be coarse, medium, or fine"
        ) from error


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


def _z_surface_grid(
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


def _z_surface_weights(
    cells: tuple[int, int, int],
    layer: int,
    length: float,
    width: float,
    *,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, _ = cells
    nodes, _ = _z_surface_grid(cells, layer, offset=offset)
    weights = np.zeros((ny + 1, nx + 1), dtype=float)
    area = (length / nx) * (width / ny)
    for j in range(ny):
        for i in range(nx):
            weights[j : j + 2, i : i + 2] += 0.25 * area
    return nodes, weights.ravel()


def _x_surface_weights(
    cells: tuple[int, int, int],
    layer: int,
    width: float,
    thickness: float,
    *,
    offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = cells
    if not 0 <= layer <= nx:
        raise ValueError("surface layer is outside the structured box")
    nodes = np.asarray(
        [
            offset + _node_index(layer, j, k, nx, ny)
            for k in range(nz + 1)
            for j in range(ny + 1)
        ],
        dtype=np.int64,
    )
    weights = np.zeros((nz + 1, ny + 1), dtype=float)
    area = (width / ny) * (thickness / nz)
    for k in range(nz):
        for j in range(ny):
            weights[k : k + 2, j : j + 2] += 0.25 * area
    return nodes, weights.ravel()


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


def _symmetry_constraints(
    nodes: np.ndarray,
    interface_z: float,
) -> DirichletConstraints:
    tolerance = 1.0e-12
    x_symmetry = np.flatnonzero(np.isclose(nodes[:, 0], 0.0, atol=tolerance))
    y_symmetry = np.flatnonzero(np.isclose(nodes[:, 1], 0.0, atol=tolerance))
    candidates = np.flatnonzero(
        np.isclose(nodes[:, 0], 0.0, atol=tolerance)
        & np.isclose(nodes[:, 1], 0.0, atol=tolerance)
        & np.isclose(nodes[:, 2], interface_z, atol=tolerance)
    )
    if candidates.size == 0:
        candidates = np.flatnonzero(
            np.isclose(nodes[:, 0], 0.0, atol=tolerance)
            & np.isclose(nodes[:, 1], 0.0, atol=tolerance)
        )
    anchor = int(candidates[0])
    dofs = {3 * int(node) for node in x_symmetry}
    dofs.update(3 * int(node) + 1 for node in y_symmetry)
    dofs.add(3 * anchor + 2)
    ordered = np.asarray(sorted(dofs), dtype=np.int64)
    return DirichletConstraints(ordered, np.zeros(len(ordered)))


def _contact_pressure_force(
    level: SandwichedBeamLevel,
    geometry: SandwichedBeamGeometry,
    lower_count: int,
    node_count: int,
) -> np.ndarray:
    force = np.zeros((node_count, 3), dtype=float)
    lower_nodes, lower_weights = _z_surface_weights(
        level.lower_cells,
        0,
        geometry.length,
        geometry.width,
    )
    upper_nodes, upper_weights = _z_surface_weights(
        level.upper_cells,
        level.upper_cells[2],
        geometry.length,
        geometry.width,
        offset=lower_count,
    )
    force[lower_nodes, 2] += geometry.ambient_pressure * lower_weights
    force[upper_nodes, 2] -= geometry.ambient_pressure * upper_weights
    return force.ravel()


def _reference_pressure_force(
    cells: tuple[int, int, int],
    geometry: SandwichedBeamGeometry,
    node_count: int,
) -> np.ndarray:
    force = np.zeros((node_count, 3), dtype=float)
    bottom, bottom_weights = _z_surface_weights(
        cells,
        0,
        geometry.length,
        geometry.width,
    )
    top, top_weights = _z_surface_weights(
        cells,
        cells[2],
        geometry.length,
        geometry.width,
    )
    force[bottom, 2] += geometry.ambient_pressure * bottom_weights
    force[top, 2] -= geometry.ambient_pressure * top_weights
    return force.ravel()


def _normalized_end_moment(
    nodes: np.ndarray,
    end_nodes: np.ndarray,
    weights: np.ndarray,
    geometry: SandwichedBeamGeometry,
) -> np.ndarray:
    force = np.zeros_like(nodes)
    lever = nodes[end_nodes, 2] - geometry.interface_z
    weighted_mean = float(np.dot(weights, lever) / np.sum(weights))
    shape = weights * (lever - weighted_mean)
    denominator = float(np.dot(lever, shape))
    if denominator <= 0.0:
        raise ValueError("end-face discretization cannot represent the bending moment")
    force[end_nodes, 0] = geometry.end_moment * shape / denominator
    return force.ravel()


def _contact_moment_force(
    level: SandwichedBeamLevel,
    geometry: SandwichedBeamGeometry,
    lower_count: int,
    nodes: np.ndarray,
) -> np.ndarray:
    lower_nodes, lower_weights = _x_surface_weights(
        level.lower_cells,
        level.lower_cells[0],
        geometry.width,
        geometry.beam_thickness,
    )
    upper_nodes, upper_weights = _x_surface_weights(
        level.upper_cells,
        level.upper_cells[0],
        geometry.width,
        geometry.beam_thickness,
        offset=lower_count,
    )
    end_nodes = np.concatenate([lower_nodes, upper_nodes])
    weights = np.concatenate([lower_weights, upper_weights])
    return _normalized_end_moment(nodes, end_nodes, weights, geometry)


def _reference_moment_force(
    cells: tuple[int, int, int],
    geometry: SandwichedBeamGeometry,
    nodes: np.ndarray,
) -> np.ndarray:
    end_nodes, weights = _x_surface_weights(
        cells,
        cells[0],
        geometry.width,
        geometry.total_thickness,
    )
    return _normalized_end_moment(nodes, end_nodes, weights, geometry)


def _resultant_moment(
    nodes: np.ndarray,
    force: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(force, dtype=float).reshape((-1, 3))
    return np.sum(np.cross(nodes - origin, vectors), axis=0)


def _validate_model(model: SandwichedBeamModel) -> None:
    geometry = model.geometry
    problem = model.problem
    interface = problem.interfaces[0]
    mesh = problem.mesh

    if model.slave_side not in ("upper", "lower"):
        raise ValueError("sandwiched-beam slave side must be 'upper' or 'lower'")
    if model.minimum_reference_determinant <= 0.0:
        raise ValueError("sandwiched-beam contact mesh contains an inverted TET4")
    if model.reference_minimum_determinant <= 0.0:
        raise ValueError("sandwiched-beam reference mesh contains an inverted TET4")

    interface.validate_for(mesh)
    if any(len(facet) != 4 for facet in interface.pair.slave.facets):
        raise ValueError(
            "sandwiched-beam slave surface must contain only QUAD4 facets"
        )
    if any(len(facet) != 4 for facet in interface.pair.master.facets):
        raise ValueError(
            "sandwiched-beam master surface must contain only QUAD4 facets"
        )
    if interface.pair.slave.node_count == interface.pair.master.node_count:
        raise ValueError("sandwiched-beam contact surfaces must be nonmatching")
    if np.intersect1d(model.slave_nodes, model.master_nodes).size:
        raise ValueError(
            "sandwiched-beam contact bodies must have distinct mapped nodes"
        )
    if not np.allclose(
        mesh.reference_nodes[model.lower_interface_nodes, 2],
        geometry.interface_z,
    ):
        raise ValueError("lower contact nodes must lie on the interface plane")
    if not np.allclose(
        mesh.reference_nodes[model.upper_interface_nodes, 2],
        geometry.interface_z,
    ):
        raise ValueError("upper contact nodes must lie on the interface plane")

    pressure = model.path.pressure_force.reshape((-1, 3))
    if not np.allclose(np.sum(pressure, axis=0), 0.0, atol=1.0e-12):
        raise ValueError("ambient pressure load must have zero global resultant")
    lower_resultant = float(np.sum(pressure[: len(model.lower_nodes), 2]))
    expected = geometry.ambient_pressure * geometry.loaded_area
    if not np.isclose(lower_resultant, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError("ambient pressure resultant does not match the loaded area")

    moment_force = model.path.moment_force.reshape((-1, 3))
    if not np.allclose(np.sum(moment_force, axis=0), 0.0, atol=1.0e-12):
        raise ValueError("sandwiched-beam end moment must have zero resultant force")
    origin = np.asarray((geometry.length, 0.0, geometry.interface_z))
    resultant = _resultant_moment(mesh.reference_nodes, moment_force, origin)
    if not np.isclose(resultant[1], geometry.end_moment, atol=1.0e-12):
        raise ValueError(
            "sandwiched-beam end couple does not reproduce the target moment"
        )
    if not np.allclose(resultant[[0, 2]], 0.0, atol=1.0e-12):
        raise ValueError("sandwiched-beam end couple contains an unintended moment")

    if model.path.pressure_force.shape != model.problem.load.force.shape:
        raise ValueError("contact path force shape does not match the contact problem")
    if (
        model.reference_path.pressure_force.shape
        != model.reference_problem.load.force.shape
    ):
        raise ValueError("reference path force shape does not match the reference problem")


def build_sandwiched_beam_model(
    level: str | SandwichedBeamLevel = "coarse",
    *,
    slave_side: str = "upper",
    geometry: SandwichedBeamGeometry | None = None,
) -> SandwichedBeamModel:
    """Build one deterministic contact model and monolithic reference."""

    selected = sandwiched_beam_level(level) if isinstance(level, str) else level
    if selected.name not in LEVELS:
        raise ValueError("custom sandwiched-beam levels must use a known level name")
    if slave_side not in ("upper", "lower"):
        raise ValueError("sandwiched-beam slave side must be 'upper' or 'lower'")
    geometry = SandwichedBeamGeometry() if geometry is None else geometry

    lower_nodes, lower_elements = _structured_box(
        selected.lower_cells,
        (0.0, 0.0, 0.0),
        (geometry.length, geometry.width, geometry.interface_z),
    )
    upper_nodes, upper_elements = _structured_box(
        selected.upper_cells,
        (0.0, 0.0, geometry.interface_z),
        (geometry.length, geometry.width, geometry.total_thickness),
    )
    lower_count = len(lower_nodes)
    nodes = np.vstack([lower_nodes, upper_nodes])
    elements = np.vstack([lower_elements, upper_elements + lower_count])
    mesh = Tet4Mesh(nodes, elements)

    lower_interface_nodes, lower_facets = _z_surface_grid(
        selected.lower_cells,
        selected.lower_cells[2],
    )
    upper_local_nodes, upper_facets = _z_surface_grid(selected.upper_cells, 0)
    upper_interface_nodes = upper_local_nodes + lower_count
    lower_surface = ContactSurface(
        nodes[lower_interface_nodes],
        lower_facets,
        normal_sign=1.0,
    )
    upper_surface = ContactSurface(
        nodes[upper_interface_nodes],
        upper_facets,
        normal_sign=-1.0,
    )
    if slave_side == "upper":
        slave_surface, master_surface = upper_surface, lower_surface
        slave_nodes, master_nodes = upper_interface_nodes, lower_interface_nodes
    else:
        slave_surface, master_surface = lower_surface, upper_surface
        slave_nodes, master_nodes = lower_interface_nodes, upper_interface_nodes
    interface = MortarContactInterface(
        ContactPair(
            slave_surface,
            master_surface,
            normal_penalty=geometry.normal_penalty,
            search_distance=geometry.search_distance,
            quadrature_points=7,
        ),
        slave_nodes,
        master_nodes,
    )

    material = NeoHookeanMaterial.from_young_poisson(
        geometry.young_modulus,
        geometry.poisson_ratio,
    )
    constraints = _symmetry_constraints(nodes, geometry.interface_z)
    problem = CoupledEquilibriumProblem(
        mesh,
        material,
        constraints,
        DeadLoad(np.zeros(3 * len(nodes))),
        (interface,),
    )
    path = SandwichedBeamLoadPath(
        _contact_pressure_force(selected, geometry, lower_count, len(nodes)),
        _contact_moment_force(selected, geometry, lower_count, nodes),
        geometry.compression_end,
    )

    reference_nodes, reference_elements = _structured_box(
        selected.reference_cells,
        (0.0, 0.0, 0.0),
        (geometry.length, geometry.width, geometry.total_thickness),
    )
    reference_mesh = Tet4Mesh(reference_nodes, reference_elements)
    reference_problem = EquilibriumProblem(
        reference_mesh,
        material,
        _symmetry_constraints(reference_nodes, geometry.interface_z),
        DeadLoad(np.zeros(3 * len(reference_nodes))),
    )
    reference_path = SandwichedBeamLoadPath(
        _reference_pressure_force(
            selected.reference_cells,
            geometry,
            len(reference_nodes),
        ),
        _reference_moment_force(
            selected.reference_cells,
            geometry,
            reference_nodes,
        ),
        geometry.compression_end,
    )

    determinants = reference_determinants(nodes, elements)
    monolithic_determinants = reference_determinants(
        reference_nodes,
        reference_elements,
    )
    model = SandwichedBeamModel(
        level=selected,
        geometry=geometry,
        slave_side=slave_side,
        problem=problem,
        path=path,
        reference_problem=reference_problem,
        reference_path=reference_path,
        lower_nodes=lower_nodes,
        upper_nodes=upper_nodes,
        lower_elements=lower_elements,
        upper_elements=upper_elements,
        lower_interface_nodes=lower_interface_nodes,
        upper_interface_nodes=upper_interface_nodes,
        slave_nodes=slave_nodes,
        master_nodes=master_nodes,
        minimum_reference_determinant=float(np.min(determinants)),
        reference_minimum_determinant=float(np.min(monolithic_determinants)),
    )
    _validate_model(model)
    return model
