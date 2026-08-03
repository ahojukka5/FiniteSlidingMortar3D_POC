"""Deterministic warped nonmatching contact-patch model family.

The family is the geometric and analytical foundation for the convergence study
tracked in issue #23.  The interface warp decreases with mesh size so that the
sequence approaches the flat patch-test limit while retaining nonplanar facets at
every finite level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from contact3d import (
    ContactPair,
    ContactSurface,
    CoupledEquilibriumProblem,
    DeadLoad,
    DirichletConstraints,
    MortarContactInterface,
    NeoHookeanMaterial,
    Tet4Mesh,
    evaluate_neo_hookean,
)

FacetKind = Literal["tri3", "quad4"]
BiasSide = Literal["lower", "upper"]


@dataclass(frozen=True, slots=True)
class WarpedPatchProfile:
    """One nonmatching bulk-mesh refinement level."""

    name: str
    lower_cells: tuple[int, int, int]
    upper_cells: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.name not in {"coarse", "medium", "fine"}:
            raise ValueError("warped patch profile name is not recognized")
        if any(value <= 0 for value in (*self.lower_cells, *self.upper_cells)):
            raise ValueError("warped patch cell counts must be positive")
        if self.lower_cells[:2] == self.upper_cells[:2]:
            raise ValueError("warped patch interface meshes must be nonmatching")

    @property
    def characteristic_size(self) -> float:
        """Return the largest in-plane cell width across both sides."""

        nx_values = (self.lower_cells[0], self.upper_cells[0])
        ny_values = (self.lower_cells[1], self.upper_cells[1])
        return max(1.0 / min(nx_values), 1.0 / min(ny_values))


COARSE_PROFILE = WarpedPatchProfile("coarse", (2, 2, 1), (3, 2, 1))
MEDIUM_PROFILE = WarpedPatchProfile("medium", (3, 3, 2), (4, 3, 2))
FINE_PROFILE = WarpedPatchProfile("fine", (4, 4, 2), (5, 4, 2))
PROFILES = {
    profile.name: profile
    for profile in (COARSE_PROFILE, MEDIUM_PROFILE, FINE_PROFILE)
}


@dataclass(frozen=True, slots=True)
class WarpedPatchSurfaceFamily:
    """Facet interpolation used on the two physical interface sides."""

    name: str
    lower_kind: FacetKind
    upper_kind: FacetKind

    def __post_init__(self) -> None:
        if self.lower_kind not in {"tri3", "quad4"}:
            raise ValueError("lower warped patch facet kind is not supported")
        if self.upper_kind not in {"tri3", "quad4"}:
            raise ValueError("upper warped patch facet kind is not supported")


SURFACE_FAMILIES = {
    item.name: item
    for item in (
        WarpedPatchSurfaceFamily("quad-quad", "quad4", "quad4"),
        WarpedPatchSurfaceFamily("tri-quad", "tri3", "quad4"),
        WarpedPatchSurfaceFamily("quad-tri", "quad4", "tri3"),
        WarpedPatchSurfaceFamily("tri-tri", "tri3", "tri3"),
    )
}


@dataclass(frozen=True, slots=True)
class WarpedPatchGeometry:
    """Physical dimensions and refinement-scaled interface distortion."""

    lower_depth: float = 1.0
    upper_height: float = 1.0
    initial_gap: float = 0.04
    compression: float = 0.06
    warp_ratio: float = 0.02
    upper_grid_skew: float = 0.12
    normal_penalty: float = 6400.0
    search_distance: float = 0.12
    young_modulus: float = 210.0
    poisson_ratio: float = 0.30

    def __post_init__(self) -> None:
        positive = (
            self.lower_depth,
            self.upper_height,
            self.initial_gap,
            self.compression,
            self.warp_ratio,
            self.normal_penalty,
            self.search_distance,
            self.young_modulus,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("warped patch physical parameters must be positive")
        if self.compression <= self.initial_gap:
            raise ValueError("compression must close the initial gap")
        if not np.isfinite(self.upper_grid_skew):
            raise ValueError("upper grid skew must be finite")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("Poisson ratio must lie strictly between -1 and 0.5")

    def warp_amplitude(self, profile: WarpedPatchProfile) -> float:
        """Return a mesh-scaled distortion that vanishes under refinement."""

        return self.warp_ratio * profile.characteristic_size

    @property
    def axial_strain(self) -> float:
        """Return the affine flat-interface reference strain."""

        total_height = self.lower_depth + self.upper_height
        return (self.initial_gap - self.compression) / total_height


@dataclass(frozen=True, slots=True)
class WarpedPatchModel:
    """Complete mesh, contact mapping, constraints, and reference quantities."""

    profile: WarpedPatchProfile
    surface_family: WarpedPatchSurfaceFamily
    bias_side: BiasSide
    geometry: WarpedPatchGeometry
    problem: CoupledEquilibriumProblem
    lower_nodes: np.ndarray
    upper_nodes: np.ndarray
    lower_elements: np.ndarray
    upper_elements: np.ndarray
    lower_interface_nodes: np.ndarray
    upper_interface_nodes: np.ndarray
    fixed_nodes: np.ndarray
    controlled_nodes: np.ndarray
    minimum_reference_determinant: float
    minimum_reference_separation: float
    warp_amplitude: float

    @property
    def interface(self) -> MortarContactInterface:
        """Return the single production mortar interface."""

        value = self.problem.interfaces[0]
        if not isinstance(value, MortarContactInterface):
            raise TypeError("warped patch model requires a mortar interface")
        return value

    @property
    def axial_strain(self) -> float:
        return self.geometry.axial_strain

    @property
    def characteristic_size(self) -> float:
        return self.profile.characteristic_size


def warped_patch_profile(name: str) -> WarpedPatchProfile:
    """Return one named refinement level."""

    try:
        return PROFILES[str(name)]
    except KeyError as error:
        raise ValueError("warped patch profile must be coarse, medium, or fine") from error


def warped_patch_surface_family(name: str) -> WarpedPatchSurfaceFamily:
    """Return one named TRI3/QUAD4 interface combination."""

    try:
        return SURFACE_FAMILIES[str(name)]
    except KeyError as error:
        raise ValueError(
            "warped patch surface family must be quad-quad, tri-quad, "
            "quad-tri, or tri-tri"
        ) from error


def _node_index(i: int, j: int, k: int, nx: int, ny: int) -> int:
    return (k * (ny + 1) + j) * (nx + 1) + i


def _in_plane_coordinates(
    i: int,
    j: int,
    nx: int,
    ny: int,
    *,
    skew: float,
) -> tuple[float, float]:
    xi = i / nx
    eta = j / ny
    bubble = xi * (1.0 - xi) * eta * (1.0 - eta)
    x = xi + skew * bubble * (2.0 * eta - 1.0)
    y = eta - 0.75 * skew * bubble * (2.0 * xi - 1.0)
    return float(x), float(y)


def interface_height(x: float, y: float, amplitude: float) -> float:
    """Return the common smooth surface sampled by both nonmatching meshes."""

    return float(
        amplitude
        * (
            np.sin(np.pi * x) * np.sin(np.pi * y)
            + 0.35 * np.sin(2.0 * np.pi * x) * np.sin(np.pi * y)
        )
    )


def _structured_block(
    cells: tuple[int, int, int],
    *,
    lower: bool,
    geometry: WarpedPatchGeometry,
    warp_amplitude: float,
    skew: float,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = cells
    nodes: list[tuple[float, float, float]] = []
    for k in range(nz + 1):
        zeta = k / nz
        for j in range(ny + 1):
            for i in range(nx + 1):
                x, y = _in_plane_coordinates(i, j, nx, ny, skew=skew)
                height = interface_height(x, y, warp_amplitude)
                if lower:
                    bottom = -geometry.lower_depth
                    top = height
                else:
                    bottom = height + geometry.initial_gap
                    top = geometry.initial_gap + geometry.upper_height
                z = (1.0 - zeta) * bottom + zeta * top
                nodes.append((x, y, float(z)))

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
    return np.asarray(nodes, dtype=float), np.asarray(elements, dtype=np.int64)


def _surface_grid(
    cells: tuple[int, int, int],
    layer: int,
    kind: FacetKind,
    *,
    offset: int = 0,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    nx, ny, nz = cells
    if not 0 <= layer <= nz:
        raise ValueError("surface layer lies outside the structured block")
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
            if kind == "quad4":
                facets.append(np.asarray((v00, v10, v11, v01), dtype=np.int64))
            elif (i + j) % 2 == 0:
                facets.extend(
                    (
                        np.asarray((v00, v10, v11), dtype=np.int64),
                        np.asarray((v00, v11, v01), dtype=np.int64),
                    )
                )
            else:
                facets.extend(
                    (
                        np.asarray((v00, v10, v01), dtype=np.int64),
                        np.asarray((v10, v11, v01), dtype=np.int64),
                    )
                )
    return nodes, tuple(facets)


def reference_determinants(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Return the signed reference determinant of every TET4 element."""

    values: list[float] = []
    for element in np.asarray(elements, dtype=np.int64):
        origin = nodes[element[0]]
        matrix = np.column_stack(
            [nodes[element[index]] - origin for index in (1, 2, 3)]
        )
        values.append(float(np.linalg.det(matrix)))
    return np.asarray(values, dtype=float)


def _constraints(
    node_count: int,
    fixed_nodes: np.ndarray,
    controlled_nodes: np.ndarray,
    compression: float,
) -> DirichletConstraints:
    prescribed: dict[int, float] = {}
    for node in range(node_count):
        prescribed[3 * node] = 0.0
        prescribed[3 * node + 1] = 0.0
    for node in fixed_nodes:
        prescribed[3 * int(node) + 2] = 0.0
    for node in controlled_nodes:
        prescribed[3 * int(node) + 2] = -compression
    dofs = np.asarray(sorted(prescribed), dtype=np.int64)
    values = np.asarray([prescribed[int(dof)] for dof in dofs], dtype=float)
    return DirichletConstraints(dofs, values)


def manufactured_displacement(model: WarpedPatchModel) -> np.ndarray:
    """Return the affine flat-limit displacement field on both warped meshes."""

    geometry = model.geometry
    strain = geometry.axial_strain
    lower_count = len(model.lower_nodes)
    values = np.zeros((model.problem.mesh.node_count, 3), dtype=float)
    lower_z = model.problem.mesh.reference_nodes[:lower_count, 2]
    upper_z = model.problem.mesh.reference_nodes[lower_count:, 2]
    values[:lower_count, 2] = strain * (lower_z + geometry.lower_depth)
    values[lower_count:, 2] = (
        -geometry.compression
        + strain * (upper_z - geometry.initial_gap - geometry.upper_height)
    )
    return values.ravel()


def reference_pressure(model: WarpedPatchModel) -> float:
    """Return the compressive first-Piola traction in the flat patch limit."""

    deformation = np.diag((1.0, 1.0, 1.0 + model.axial_strain))
    response = evaluate_neo_hookean(deformation, model.problem.material)
    return float(-response.first_piola[2, 2])


def reference_vertical_reaction(model: WarpedPatchModel) -> float:
    """Return the unit-area vertical reaction in the flat patch limit."""

    return reference_pressure(model)


def analytical_deformed_gap(
    model: WarpedPatchModel,
    x: float,
    y: float,
) -> float:
    """Evaluate the manufactured continuous-surface gap at one point."""

    geometry = model.geometry
    height = interface_height(x, y, model.warp_amplitude)
    lower_z = height
    upper_z = height + geometry.initial_gap
    strain = geometry.axial_strain
    lower_displacement = strain * (lower_z + geometry.lower_depth)
    upper_displacement = (
        -geometry.compression
        + strain * (upper_z - geometry.initial_gap - geometry.upper_height)
    )
    return float(
        upper_z
        + upper_displacement
        - lower_z
        - lower_displacement
    )


def _validate_model(model: WarpedPatchModel) -> None:
    geometry = model.geometry
    if model.minimum_reference_determinant <= 0.0:
        raise ValueError("warped patch mesh contains an inverted TET4 element")
    if model.minimum_reference_separation <= 0.0:
        raise ValueError("warped patch surfaces must begin separated")
    if geometry.search_distance <= model.minimum_reference_separation:
        raise ValueError("contact search distance must exceed the initial separation")
    if not -1.0 < model.axial_strain < 0.0:
        raise ValueError("warped patch reference strain must be compressive and valid")

    interface = model.interface
    interface.validate_for(model.problem.mesh)
    expected_slave = (
        model.lower_interface_nodes
        if model.bias_side == "lower"
        else model.upper_interface_nodes
    )
    expected_master = (
        model.upper_interface_nodes
        if model.bias_side == "lower"
        else model.lower_interface_nodes
    )
    if not np.array_equal(interface.slave_nodes, expected_slave):
        raise ValueError("warped patch slave mapping does not match the bias side")
    if not np.array_equal(interface.master_nodes, expected_master):
        raise ValueError("warped patch master mapping does not match the bias side")
    if len(model.lower_interface_nodes) == len(model.upper_interface_nodes):
        raise ValueError("warped patch interface node sets must be nonmatching")

    manufactured = manufactured_displacement(model)
    constraints = model.problem.constraints
    if not np.allclose(manufactured[constraints.dofs], constraints.values):
        raise ValueError("manufactured field must satisfy every prescribed value")
    if reference_pressure(model) <= 0.0:
        raise ValueError("warped patch reference pressure must be compressive")


def build_warped_patch_model(
    profile: str | WarpedPatchProfile = "coarse",
    *,
    surface_family: str | WarpedPatchSurfaceFamily = "quad-quad",
    bias_side: BiasSide = "upper",
    geometry: WarpedPatchGeometry | None = None,
) -> WarpedPatchModel:
    """Build one deterministic warped, nonmatching contact patch problem."""

    selected_profile = (
        warped_patch_profile(profile) if isinstance(profile, str) else profile
    )
    selected_family = (
        warped_patch_surface_family(surface_family)
        if isinstance(surface_family, str)
        else surface_family
    )
    if bias_side not in {"lower", "upper"}:
        raise ValueError("warped patch bias side must be lower or upper")
    geometry = WarpedPatchGeometry() if geometry is None else geometry
    amplitude = geometry.warp_amplitude(selected_profile)

    lower_nodes, lower_elements = _structured_block(
        selected_profile.lower_cells,
        lower=True,
        geometry=geometry,
        warp_amplitude=amplitude,
        skew=0.0,
    )
    upper_nodes, upper_elements = _structured_block(
        selected_profile.upper_cells,
        lower=False,
        geometry=geometry,
        warp_amplitude=amplitude,
        skew=geometry.upper_grid_skew,
    )
    lower_count = len(lower_nodes)
    nodes = np.vstack((lower_nodes, upper_nodes))
    elements = np.vstack((lower_elements, upper_elements + lower_count))
    mesh = Tet4Mesh(nodes, elements)

    lower_interface_nodes, lower_facets = _surface_grid(
        selected_profile.lower_cells,
        selected_profile.lower_cells[2],
        selected_family.lower_kind,
    )
    upper_interface_nodes, upper_facets = _surface_grid(
        selected_profile.upper_cells,
        0,
        selected_family.upper_kind,
        offset=lower_count,
    )
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
    if bias_side == "upper":
        slave_surface = upper_surface
        master_surface = lower_surface
        slave_nodes = upper_interface_nodes
        master_nodes = lower_interface_nodes
    else:
        slave_surface = lower_surface
        master_surface = upper_surface
        slave_nodes = lower_interface_nodes
        master_nodes = upper_interface_nodes
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

    fixed_nodes, _ = _surface_grid(
        selected_profile.lower_cells,
        0,
        "quad4",
    )
    controlled_nodes, _ = _surface_grid(
        selected_profile.upper_cells,
        selected_profile.upper_cells[2],
        "quad4",
        offset=lower_count,
    )
    constraints = _constraints(
        mesh.node_count,
        fixed_nodes,
        controlled_nodes,
        geometry.compression,
    )
    problem = CoupledEquilibriumProblem(
        mesh,
        NeoHookeanMaterial.from_young_poisson(
            geometry.young_modulus,
            geometry.poisson_ratio,
        ),
        constraints,
        DeadLoad(np.zeros(3 * mesh.node_count, dtype=float)),
        (interface,),
    )

    determinants = reference_determinants(nodes, elements)
    lower_interface_z = nodes[lower_interface_nodes, 2]
    upper_interface_z = nodes[upper_interface_nodes, 2]
    model = WarpedPatchModel(
        profile=selected_profile,
        surface_family=selected_family,
        bias_side=bias_side,
        geometry=geometry,
        problem=problem,
        lower_nodes=lower_nodes,
        upper_nodes=upper_nodes,
        lower_elements=lower_elements,
        upper_elements=upper_elements,
        lower_interface_nodes=lower_interface_nodes,
        upper_interface_nodes=upper_interface_nodes,
        fixed_nodes=fixed_nodes,
        controlled_nodes=controlled_nodes,
        minimum_reference_determinant=float(np.min(determinants)),
        minimum_reference_separation=float(
            np.min(upper_interface_z) - np.max(lower_interface_z)
        ),
        warp_amplitude=amplitude,
    )
    _validate_model(model)
    return model
