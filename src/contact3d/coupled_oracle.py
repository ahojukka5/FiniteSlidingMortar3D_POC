"""Topology-frozen matching-facet mortar kernel for coupled-driver verification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contact3d.coupled import (
    ContactBranchSignature,
    ContactInterfaceEvaluation,
    ContactInterfaceUpdate,
)
from contact3d.enforcement_state import AugmentedLagrangeState, kkt_diagnostics

MATCHING_QUAD_MASS = np.array(
    [
        [4.0, 2.0, 1.0, 2.0],
        [2.0, 4.0, 2.0, 1.0],
        [1.0, 2.0, 4.0, 2.0],
        [2.0, 1.0, 2.0, 4.0],
    ]
) / 36.0


@dataclass(frozen=True, slots=True)
class FrozenMatchingMortarInterface:
    """Exact standard-mortar operator for two matching unit-square Q1 facets.

    This verification kernel deliberately freezes the overlap, normal, and D/M
    operators. It tests global coupling and augmentation independently from the
    already verified moving-overlap derivatives.
    """

    slave_nodes: np.ndarray
    master_nodes: np.ndarray
    normal: np.ndarray
    penalty: float
    event_gap: float | None = None
    initial_normal_gap: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.initial_normal_gap):
            raise ValueError("initial_normal_gap must be finite")

    @property
    def dofs(self) -> np.ndarray:
        nodes = np.concatenate([self.slave_nodes, self.master_nodes])
        return np.asarray(
            [3 * int(node) + component for node in nodes for component in range(3)],
            dtype=np.int64,
        )

    def initial_state(self) -> AugmentedLagrangeState:
        return AugmentedLagrangeState.zeros(4)

    def _kinematics(self, displacement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(displacement, dtype=float).reshape((-1, 3))
        slave = values[self.slave_nodes]
        master = values[self.master_nodes]
        row_areas = np.sum(MATCHING_QUAD_MASS, axis=1)
        weighted_gap = MATCHING_QUAD_MASS @ slave - MATCHING_QUAD_MASS @ master
        normal_gaps = self.initial_normal_gap + weighted_gap @ self.normal / row_areas
        return row_areas, normal_gaps

    def evaluate(
        self,
        displacement: np.ndarray,
        state: AugmentedLagrangeState,
        *,
        tolerance: float,
    ) -> ContactInterfaceEvaluation:
        _, gaps = self._kinematics(displacement)
        supported = np.ones(4, dtype=bool)
        trial = state.multipliers + self.penalty * gaps
        active = trial > 0.0
        pressure = np.maximum(trial, 0.0)
        traction = pressure[:, None] * self.normal
        slave_force = MATCHING_QUAD_MASS.T @ traction
        master_force = -(MATCHING_QUAD_MASS.T @ traction)
        residual = np.concatenate([slave_force.ravel(), master_force.ravel()])
        region = 0
        if self.event_gap is not None:
            region = int(float(np.mean(gaps)) > self.event_gap)
        signature = ContactBranchSignature(
            ((0, region),),
            tuple(bool(value) for value in active),
            (True, True, True, True),
        )
        diagnostics = kkt_diagnostics(
            state.multipliers,
            gaps,
            self.penalty,
            supported,
        )
        return ContactInterfaceEvaluation(
            residual,
            diagnostics,
            signature,
            gaps,
            pressure,
            (state, gaps, active),
        )

    def tangent(
        self,
        displacement: np.ndarray,
        state: AugmentedLagrangeState,
        evaluation: ContactInterfaceEvaluation,
        *,
        tolerance: float,
    ) -> np.ndarray:
        del displacement, state, tolerance
        _, _, active = evaluation.raw
        row_areas = np.sum(MATCHING_QUAD_MASS, axis=1)
        gap_jacobian = np.zeros((4, 24), dtype=float)
        for row in range(4):
            for node in range(4):
                gap_jacobian[row, 3 * node : 3 * node + 3] += (
                    MATCHING_QUAD_MASS[row, node] * self.normal / row_areas[row]
                )
                gap_jacobian[row, 12 + 3 * node : 15 + 3 * node] -= (
                    MATCHING_QUAD_MASS[row, node] * self.normal / row_areas[row]
                )
        pressure_jacobian = self.penalty * active[:, None] * gap_jacobian
        traction_jacobian = pressure_jacobian[:, None, :] * self.normal[None, :, None]
        slave = np.einsum(
            "ji,jcq->icq",
            MATCHING_QUAD_MASS,
            traction_jacobian,
        ).reshape((12, 24))
        master = -np.einsum(
            "ji,jcq->icq",
            MATCHING_QUAD_MASS,
            traction_jacobian,
        ).reshape((12, 24))
        return np.vstack([slave, master])

    def augment(
        self,
        evaluation: ContactInterfaceEvaluation,
        *,
        tolerance: float,
    ) -> ContactInterfaceUpdate:
        del tolerance
        state, gaps, _ = evaluation.raw
        values = np.maximum(state.multipliers + self.penalty * gaps, 0.0)
        next_state = AugmentedLagrangeState(values, state.augmentation + 1)
        diagnostics = kkt_diagnostics(
            values,
            gaps,
            self.penalty,
            np.ones(4, dtype=bool),
        )
        return ContactInterfaceUpdate(
            next_state,
            values - state.multipliers,
            diagnostics,
        )
