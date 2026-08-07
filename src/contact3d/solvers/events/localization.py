"""Topology observations used by event-localized nonlinear solvers."""

from __future__ import annotations

from ...clipping import ClippingTopologyError
from ...coupling import (
    CoupledEquilibriumEvaluation,
    CoupledEquilibriumProblem,
    evaluate_coupled_equilibrium,
)
from ...event_geometry import contact_topology_signatures
from ...mechanics import FloatArray
from ...mortar.enforcement import AugmentedLagrangeState
from ...pallets import PalletTopologyError
from ...parametric import InverseMapTopologyError
from ...topology_model import ContactTopologySignature, EventKind, TopologyObservation

RECOVERABLE_CONTACT_EVENT_ERRORS = (
    ClippingTopologyError,
    PalletTopologyError,
    InverseMapTopologyError,
)


def recoverable_event_kind(error: BaseException) -> EventKind:
    """Map one recoverable geometry failure to its topology event kind."""

    if isinstance(error, ClippingTopologyError):
        return "clipping_vertex_edge"
    if isinstance(error, PalletTopologyError):
        return "pallet_transition"
    if isinstance(error, InverseMapTopologyError):
        return "inverse_map_boundary"
    raise TypeError(f"unsupported recoverable contact event: {type(error).__name__}")


def event_signatures(
    problem: CoupledEquilibriumProblem,
    evaluation: CoupledEquilibriumEvaluation,
    *,
    tolerance: float,
) -> tuple[ContactTopologySignature, ...]:
    """Adapt one coupled evaluation to event-compatible branch signatures."""

    return contact_topology_signatures(
        problem,
        evaluation.displacement,
        evaluation.contacts,
        tolerance=tolerance,
    )


def observe_event_trial(
    problem: CoupledEquilibriumProblem,
    states: tuple[AugmentedLagrangeState, ...],
    displacement: FloatArray,
    step: FloatArray,
    fraction: float,
    *,
    load_factor: float,
    tolerance: float,
) -> TopologyObservation:
    """Evaluate one line-search fraction as a topology observation."""

    try:
        evaluation = evaluate_coupled_equilibrium(
            problem,
            displacement + fraction * step,
            states,
            load_factor=load_factor,
            assemble_tangent=False,
            tolerance=tolerance,
        )
    except RECOVERABLE_CONTACT_EVENT_ERRORS as error:
        return TopologyObservation.recoverable(
            fraction,
            recoverable_event_kind(error),
            str(error),
        )
    signatures = event_signatures(problem, evaluation, tolerance=tolerance)
    return TopologyObservation.valid(fraction, signatures, evaluation)


__all__ = [
    "RECOVERABLE_CONTACT_EVENT_ERRORS",
    "event_signatures",
    "observe_event_trial",
    "recoverable_event_kind",
]
