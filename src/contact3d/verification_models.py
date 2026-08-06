"""Reusable structured verification models for solver and contact benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coupled import CoupledEquilibriumProblem
from .coupled_oracle import FrozenMatchingMortarInterface
from .mechanics import DeadLoad, DirichletConstraints, NeoHookeanMaterial, Tet4Mesh


@dataclass(frozen=True, slots=True)
class StackedBlockContactModel:
    """Two structured TET4 blocks coupled by matching mortar facets."""

    problem: CoupledEquilibriumProblem
    resolution: int
    layers: int
    indentation: float
    interface_area: float
    lower_node_count: int
    interface_count: int
    bottom_nodes: np.ndarray
    top_nodes: np.ndarray

    @property
    def total_dofs(self) -> int:
        return 3 * self.problem.mesh.node_count

    @property
    def free_dofs(self) -> np.ndarray:
        return self.problem.constraints.free_dofs(self.total_dofs)


def _node_index(i: int, j: int, k: int, resolution: int) -> int:
    width = resolution + 1
    return (k * width + j) * width + i


def _structured_block(
    resolution: int,
    layers: int,
    z_start: float,
    z_stop: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.linspace(0.0, 1.0, resolution + 1)
    y_values = np.linspace(0.0, 1.0, resolution + 1)
    z_values = np.linspace(z_start, z_stop, layers + 1)
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
    for k in range(layers):
        for j in range(resolution):
            for i in range(resolution):
                v000 = _node_index(i, j, k, resolution)
                v100 = _node_index(i + 1, j, k, resolution)
                v110 = _node_index(i + 1, j + 1, k, resolution)
                v010 = _node_index(i, j + 1, k, resolution)
                v001 = _node_index(i, j, k + 1, resolution)
                v101 = _node_index(i + 1, j, k + 1, resolution)
                v111 = _node_index(i + 1, j + 1, k + 1, resolution)
                v011 = _node_index(i, j + 1, k + 1, resolution)
                elements.extend(
                    [
                        (v000, v100, v110, v111),
                        (v000, v110, v010, v111),
                        (v000, v010, v011, v111),
                        (v000, v011, v001, v111),
                        (v000, v001, v101, v111),
                        (v000, v101, v100, v111),
                    ]
                )
    return nodes, np.asarray(elements, dtype=np.int64)


def _surface_nodes(
    resolution: int,
    layer: int,
    *,
    offset: int = 0,
) -> np.ndarray:
    return np.asarray(
        [
            offset + _node_index(i, j, layer, resolution)
            for j in range(resolution + 1)
            for i in range(resolution + 1)
        ],
        dtype=np.int64,
    )


def _surface_quads(
    resolution: int,
    layer: int,
    *,
    offset: int = 0,
) -> tuple[np.ndarray, ...]:
    quads: list[np.ndarray] = []
    for j in range(resolution):
        for i in range(resolution):
            quads.append(
                np.asarray(
                    [
                        offset + _node_index(i, j, layer, resolution),
                        offset + _node_index(i + 1, j, layer, resolution),
                        offset + _node_index(i + 1, j + 1, layer, resolution),
                        offset + _node_index(i, j + 1, layer, resolution),
                    ],
                    dtype=np.int64,
                )
            )
    return tuple(quads)


def stacked_matching_block_contact_model(
    resolution: int,
    *,
    layers: int = 2,
    indentation: float = 0.04,
    penalty: float = 6400.0,
    young_modulus: float = 210.0,
    poisson_ratio: float = 0.3,
) -> StackedBlockContactModel:
    """Build a mesh-refinable coupled problem for linear-solver studies."""

    if resolution <= 0 or layers <= 0:
        raise ValueError("resolution and layers must be positive")
    for name, value in (
        ("indentation", indentation),
        ("penalty", penalty),
        ("young_modulus", young_modulus),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if indentation >= 0.5:
        raise ValueError("indentation must be smaller than one block height")
    if not np.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie between -1 and 0.5")

    lower_nodes, lower_elements = _structured_block(resolution, layers, 0.0, 0.5)
    upper_nodes, upper_elements = _structured_block(resolution, layers, 0.5, 1.0)
    lower_node_count = len(lower_nodes)
    nodes = np.vstack([lower_nodes, upper_nodes])
    elements = np.vstack([lower_elements, upper_elements + lower_node_count])
    mesh = Tet4Mesh(nodes, elements)

    lower_quads = _surface_quads(resolution, layers)
    upper_quads = _surface_quads(
        resolution,
        0,
        offset=lower_node_count,
    )
    interface_area = 1.0 / resolution**2
    normal = np.array([0.0, 0.0, -1.0])
    interfaces = tuple(
        FrozenMatchingMortarInterface(
            upper,
            lower,
            normal,
            penalty,
            area=interface_area,
        )
        for lower, upper in zip(lower_quads, upper_quads, strict=True)
    )

    bottom_nodes = _surface_nodes(resolution, 0)
    top_nodes = _surface_nodes(
        resolution,
        layers,
        offset=lower_node_count,
    )
    constrained_dofs: list[int] = []
    constrained_values: list[float] = []
    for node in bottom_nodes:
        for component in range(3):
            constrained_dofs.append(3 * int(node) + component)
            constrained_values.append(0.0)
    for node in top_nodes:
        for component in range(3):
            constrained_dofs.append(3 * int(node) + component)
            constrained_values.append(-indentation if component == 2 else 0.0)

    problem = CoupledEquilibriumProblem(
        mesh,
        NeoHookeanMaterial.from_young_poisson(young_modulus, poisson_ratio),
        DirichletConstraints(
            np.asarray(constrained_dofs, dtype=np.int64),
            np.asarray(constrained_values, dtype=float),
        ),
        DeadLoad(np.zeros(3 * mesh.node_count)),
        interfaces,
    )
    return StackedBlockContactModel(
        problem=problem,
        resolution=resolution,
        layers=layers,
        indentation=float(indentation),
        interface_area=float(interface_area),
        lower_node_count=lower_node_count,
        interface_count=len(interfaces),
        bottom_nodes=bottom_nodes,
        top_nodes=top_nodes,
    )
