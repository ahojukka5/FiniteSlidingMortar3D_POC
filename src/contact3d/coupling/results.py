"""Immutable coupling results and interface records."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..mechanics import CSRMatrix, FloatArray, IntArray, Tet4SparseEvaluation
from ..mortar.enforcement import AugmentedLagrangeState, KKTDiagnostics


@dataclass(frozen=True, slots=True)
class ContactBranchSignature:
    """Discrete contact branch frozen during one smooth linearization."""

    facet_pairs: tuple[tuple[int, int], ...]
    active_rows: tuple[bool, ...]
    supported_rows: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class ContactInterfaceEvaluation:
    """Local contact contribution in interface-local DOF ordering."""

    residual: FloatArray
    diagnostics: KKTDiagnostics
    signature: ContactBranchSignature
    normal_gaps: FloatArray
    pressure: FloatArray
    raw: object = field(repr=False)


@dataclass(frozen=True, slots=True)
class ContactInterfaceUpdate:
    """Accepted multiplier update for one mapped contact interface."""

    state: AugmentedLagrangeState
    increment: FloatArray
    diagnostics_after: KKTDiagnostics


@dataclass(frozen=True, slots=True)
class CoupledEquilibriumEvaluation:
    """Assembled bulk and contact equilibrium state."""

    displacement: FloatArray
    load_factor: float
    bulk_potential: float
    residual: FloatArray
    tangent: CSRMatrix | None
    free_dofs: IntArray
    free_residual_norm: float
    bulk: Tet4SparseEvaluation
    contacts: tuple[ContactInterfaceEvaluation, ...]

    @property
    def signatures(self) -> tuple[ContactBranchSignature, ...]:
        return tuple(contact.signature for contact in self.contacts)

    @property
    def maximum_penetration(self) -> float:
        return max(
            (contact.diagnostics.maximum_penetration for contact in self.contacts),
            default=0.0,
        )
