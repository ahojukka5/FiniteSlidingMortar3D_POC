"""Immutable coupled mechanics and contact problem definition."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..mechanics import (
    DeadLoad,
    DirichletConstraints,
    NeoHookeanMaterial,
    Tet4Mesh,
    Tet4Sparsity,
)
from ..mortar.enforcement import AugmentedLagrangeState
from .protocols import CoupledContactInterface


@dataclass(frozen=True, slots=True)
class CoupledEquilibriumProblem:
    """Finite-strain bulk problem with mapped contact interfaces."""

    mesh: Tet4Mesh
    material: NeoHookeanMaterial
    constraints: DirichletConstraints
    load: DeadLoad
    interfaces: tuple[CoupledContactInterface, ...]
    sparsity: Tet4Sparsity = field(init=False, repr=False)

    def __post_init__(self) -> None:
        total_dofs = 3 * self.mesh.node_count
        self.constraints.validate_for(total_dofs)
        if self.load.force.shape != (total_dofs,):
            raise ValueError("dead-load vector must match the mesh DOF count")
        interfaces = tuple(self.interfaces)
        if not interfaces:
            raise ValueError("coupled problem must contain at least one contact interface")
        for interface in interfaces:
            if np.any(interface.dofs < 0) or np.any(interface.dofs >= total_dofs):
                raise ValueError("contact interface DOF is outside the bulk mesh")
            validate = getattr(interface, "validate_for", None)
            if validate is not None:
                validate(self.mesh)
        pattern = Tet4Sparsity.from_mesh(
            self.mesh,
            tuple(interface.dofs for interface in interfaces),
        )
        object.__setattr__(self, "interfaces", interfaces)
        object.__setattr__(self, "sparsity", pattern)

    def initial_states(self) -> tuple[AugmentedLagrangeState, ...]:
        return tuple(interface.initial_state() for interface in self.interfaces)

    def validate_states(
        self,
        states: tuple[AugmentedLagrangeState, ...] | None,
    ) -> tuple[AugmentedLagrangeState, ...]:
        values = self.initial_states() if states is None else tuple(states)
        if len(values) != len(self.interfaces):
            raise ValueError("one multiplier state is required for every contact interface")
        return values
