"""Adapters that map mortar contact pairs into global mechanics DOFs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mechanics import FloatArray, IntArray, Tet4Mesh
from ..mortar import ContactPair
from ..mortar.enforcement import (
    AugmentedLagrangeEvaluation,
    AugmentedLagrangeState,
    augment_multipliers,
    augmented_lagrange_contact_tangent,
    evaluate_augmented_lagrange,
)
from .results import (
    ContactBranchSignature,
    ContactInterfaceEvaluation,
    ContactInterfaceUpdate,
)


@dataclass(frozen=True, slots=True)
class MortarContactInterface:
    """Map one mortar contact pair into global bulk-mesh DOFs."""

    pair: ContactPair
    slave_nodes: IntArray
    master_nodes: IntArray

    def __post_init__(self) -> None:
        slave = np.asarray(self.slave_nodes, dtype=np.int64)
        master = np.asarray(self.master_nodes, dtype=np.int64)
        if slave.shape != (self.pair.slave.node_count,):
            raise ValueError("slave_nodes must match the mortar slave-node count")
        if master.shape != (self.pair.master.node_count,):
            raise ValueError("master_nodes must match the mortar master-node count")
        if np.any(slave < 0) or np.any(master < 0):
            raise ValueError("mapped contact node indices must be nonnegative")
        if len(np.unique(slave)) != len(slave) or len(np.unique(master)) != len(master):
            raise ValueError("mapped nodes must be unique on each contact surface")
        object.__setattr__(self, "slave_nodes", slave.copy())
        object.__setattr__(self, "master_nodes", master.copy())

    @property
    def dofs(self) -> IntArray:
        nodes = np.concatenate([self.slave_nodes, self.master_nodes])
        return np.asarray(
            [3 * int(node) + component for node in nodes for component in range(3)],
            dtype=np.int64,
        )

    def validate_for(self, mesh: Tet4Mesh, *, tolerance: float = 1.0e-12) -> None:
        if np.any(self.slave_nodes >= mesh.node_count) or np.any(
            self.master_nodes >= mesh.node_count
        ):
            raise ValueError("mapped contact node index is outside the bulk mesh")
        if not np.allclose(
            mesh.reference_nodes[self.slave_nodes],
            self.pair.slave.reference_nodes,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("slave contact reference coordinates do not match the bulk mesh")
        if not np.allclose(
            mesh.reference_nodes[self.master_nodes],
            self.pair.master.reference_nodes,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("master contact reference coordinates do not match the bulk mesh")

    def initial_state(self) -> AugmentedLagrangeState:
        return AugmentedLagrangeState.zeros(self.pair.slave.node_count)

    def _surface_displacements(
        self,
        displacement: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(displacement, dtype=float).reshape((-1, 3))
        return values[self.slave_nodes], values[self.master_nodes]

    @staticmethod
    def _raw_evaluation(
        evaluation: ContactInterfaceEvaluation,
    ) -> AugmentedLagrangeEvaluation:
        raw = evaluation.raw
        if not isinstance(raw, AugmentedLagrangeEvaluation):
            raise TypeError("mortar interface evaluation payload has the wrong type")
        return raw

    def evaluate(
        self,
        displacement: FloatArray,
        state: AugmentedLagrangeState,
        *,
        tolerance: float,
    ) -> ContactInterfaceEvaluation:
        slave, master = self._surface_displacements(displacement)
        result = evaluate_augmented_lagrange(
            self.pair,
            state,
            slave,
            master,
            tolerance=tolerance,
        )
        supported = result.contact.weights.row_areas > tolerance
        signature = ContactBranchSignature(
            tuple(result.contact.weights.facet_pairs),
            tuple(bool(value) for value in result.contact.active_rows),
            tuple(bool(value) for value in supported),
        )
        return ContactInterfaceEvaluation(
            residual=result.contact.residual.copy(),
            diagnostics=result.diagnostics,
            signature=signature,
            normal_gaps=result.contact.normal_gaps.copy(),
            pressure=result.contact.pressure.copy(),
            raw=result,
        )

    def tangent(
        self,
        displacement: FloatArray,
        state: AugmentedLagrangeState,
        evaluation: ContactInterfaceEvaluation,
        *,
        tolerance: float,
    ) -> FloatArray:
        slave, master = self._surface_displacements(displacement)
        raw = self._raw_evaluation(evaluation)
        return augmented_lagrange_contact_tangent(
            self.pair,
            state,
            slave,
            master,
            facet_pairs=raw.contact.weights.facet_pairs,
            active_rows=raw.contact.active_rows,
            tolerance=tolerance,
        )

    def augment(
        self,
        evaluation: ContactInterfaceEvaluation,
        *,
        tolerance: float,
    ) -> ContactInterfaceUpdate:
        raw = self._raw_evaluation(evaluation)
        update = augment_multipliers(self.pair, raw, tolerance=tolerance)
        return ContactInterfaceUpdate(
            update.state,
            update.increment.copy(),
            update.diagnostics_after,
        )
