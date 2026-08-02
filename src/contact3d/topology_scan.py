"""Solver-independent contact-topology scans along prescribed load paths."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .coupled import CoupledEquilibriumProblem
from .event_geometry import contact_topology_signatures
from .load_path import CoupledLoadPath
from .model import FloatArray
from .topology_diff import signature_events
from .topology_model import ContactTopologySignature, EventKind


@dataclass(frozen=True, slots=True)
class KinematicContactTopologySample:
    """One interface signature and its continuous diagnostics at one path state."""

    interface: int
    signature: ContactTopologySignature
    overlap_area: float
    supported_rows: tuple[int, ...]
    active_rows: tuple[int, ...]
    maximum_pressure: float

    @property
    def facet_pair_count(self) -> int:
        return len(self.signature.facet_pairs)

    @property
    def supported_row_count(self) -> int:
        return len(self.supported_rows)

    @property
    def active_row_count(self) -> int:
        return len(self.active_rows)


@dataclass(frozen=True, slots=True)
class KinematicTopologyFrame:
    """All interface observations at one absolute continuation parameter."""

    parameter: float
    phase: str
    phase_parameter: float
    displacement: FloatArray
    contacts: tuple[KinematicContactTopologySample, ...]

    def __post_init__(self) -> None:
        parameter = float(self.parameter)
        phase_parameter = float(self.phase_parameter)
        displacement = np.asarray(self.displacement, dtype=float).reshape(-1)
        if not np.isfinite(parameter) or parameter < 0.0:
            raise ValueError("topology-frame parameter must be finite and nonnegative")
        if not np.isfinite(phase_parameter):
            raise ValueError("topology-frame phase parameter must be finite")
        if not np.all(np.isfinite(displacement)):
            raise ValueError("topology-frame displacement must be finite")
        contacts = tuple(self.contacts)
        if not contacts:
            raise ValueError("topology frame must contain at least one interface")
        if tuple(sample.interface for sample in contacts) != tuple(range(len(contacts))):
            raise ValueError("topology-frame interfaces must be contiguous and ordered")
        object.__setattr__(self, "parameter", parameter)
        object.__setattr__(self, "phase", str(self.phase))
        object.__setattr__(self, "phase_parameter", phase_parameter)
        object.__setattr__(self, "displacement", displacement.copy())
        object.__setattr__(self, "contacts", contacts)

    @property
    def signatures(self) -> tuple[ContactTopologySignature, ...]:
        return tuple(sample.signature for sample in self.contacts)


@dataclass(frozen=True, slots=True)
class KinematicTopologyChange:
    """One atomic branch change known to occur inside a sampled interval."""

    kind: EventKind
    interface: int
    entity: tuple[int, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class KinematicTopologyTransition:
    """A deterministic absolute path interval bracketing topology changes."""

    left_parameter: float
    right_parameter: float
    changes: tuple[KinematicTopologyChange, ...]

    def __post_init__(self) -> None:
        left = float(self.left_parameter)
        right = float(self.right_parameter)
        changes = tuple(self.changes)
        if not np.isfinite(left) or not np.isfinite(right) or right <= left:
            raise ValueError("topology-transition bounds must be finite and ordered")
        if not changes:
            raise ValueError("topology transition must contain at least one change")
        object.__setattr__(self, "left_parameter", left)
        object.__setattr__(self, "right_parameter", right)
        object.__setattr__(self, "changes", changes)

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.left_parameter + self.right_parameter)

    @property
    def width(self) -> float:
        return self.right_parameter - self.left_parameter

    def count(self, kind: EventKind) -> int:
        return sum(change.kind == kind for change in self.changes)


@dataclass(frozen=True, slots=True)
class KinematicTopologyScan:
    """Prescribed path frames and all branch-changing sample intervals."""

    frames: tuple[KinematicTopologyFrame, ...]
    transitions: tuple[KinematicTopologyTransition, ...]
    geometry_tolerance: float

    def __post_init__(self) -> None:
        frames = tuple(self.frames)
        transitions = tuple(self.transitions)
        tolerance = float(self.geometry_tolerance)
        if len(frames) < 2:
            raise ValueError("kinematic topology scan requires at least two frames")
        parameters = np.asarray([frame.parameter for frame in frames], dtype=float)
        if np.any(np.diff(parameters) <= 0.0):
            raise ValueError("kinematic topology parameters must be strictly increasing")
        interface_counts = {len(frame.contacts) for frame in frames}
        if len(interface_counts) != 1:
            raise ValueError("every topology frame must contain the same interfaces")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("geometry tolerance must be finite and positive")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "geometry_tolerance", tolerance)

    @property
    def signature_digest(self) -> str:
        """Return a stable digest of sampled discrete signatures."""

        payload = [
            {
                "parameter": format(frame.parameter, ".17g"),
                "signatures": [
                    {
                        "facet_pairs": sample.signature.facet_pairs,
                        "active_rows": sample.signature.active_rows,
                        "supported_rows": sample.signature.supported_rows,
                        "geometry_tokens": sample.signature.geometry_tokens,
                    }
                    for sample in frame.contacts
                ],
            }
            for frame in self.frames
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def event_count(self, kind: EventKind) -> int:
        return sum(transition.count(kind) for transition in self.transitions)


def _phase_value(values: tuple[tuple[str, float], ...], name: str, default: float) -> float:
    for key, value in values:
        if key == name:
            return float(value)
    return float(default)


def _phase_name(path: CoupledLoadPath, parameter: float) -> str:
    resolver = getattr(path, "phase_name", None)
    return str(resolver(parameter)) if callable(resolver) else "path"


def _validated_parameters(parameters: object) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("scan parameters must be a one-dimensional vector of length two or more")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("scan parameters must be finite and nonnegative")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("scan parameters must be strictly increasing")
    return values


def scan_kinematic_contact_path(
    problem: CoupledEquilibriumProblem,
    path: CoupledLoadPath,
    parameters: object,
    *,
    geometry_tolerance: float = 1.0e-12,
) -> KinematicTopologyScan:
    """Scan contact branches at prescribed states without solving equilibrium.

    Free bulk DOFs are held at zero while the path constraints are applied exactly.
    Every contact interface is evaluated with its zero multiplier state. No Newton,
    line-search, augmentation, or nonlinear stopping option enters this procedure.
    """

    values = _validated_parameters(parameters)
    tolerance = float(geometry_tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("geometry tolerance must be finite and positive")

    frames: list[KinematicTopologyFrame] = []
    for parameter in values:
        path_state = path.evaluate(problem, float(parameter))
        candidate = path_state.problem
        total_dofs = 3 * candidate.mesh.node_count
        displacement = candidate.constraints.apply(np.zeros(total_dofs, dtype=float))
        states = candidate.initial_states()
        contacts = tuple(
            interface.evaluate(displacement, state, tolerance=tolerance)
            for interface, state in zip(candidate.interfaces, states, strict=True)
        )
        signatures = contact_topology_signatures(
            candidate,
            displacement,
            contacts,
            tolerance=tolerance,
        )

        samples: list[KinematicContactTopologySample] = []
        for interface, (contact, signature) in enumerate(
            zip(contacts, signatures, strict=True)
        ):
            raw_contact = getattr(contact.raw, "contact", None)
            weights = getattr(raw_contact, "weights", None)
            if weights is None:
                raise TypeError(
                    "kinematic topology scan requires mortar contact weight diagnostics"
                )
            supported = tuple(
                int(index)
                for index, active in enumerate(signature.supported_rows)
                if active
            )
            active = tuple(
                int(index)
                for index, active in enumerate(signature.active_rows)
                if active
            )
            samples.append(
                KinematicContactTopologySample(
                    interface=interface,
                    signature=signature,
                    overlap_area=float(weights.total_area),
                    supported_rows=supported,
                    active_rows=active,
                    maximum_pressure=float(np.max(contact.pressure, initial=0.0)),
                )
            )
        frames.append(
            KinematicTopologyFrame(
                parameter=float(parameter),
                phase=_phase_name(path, float(parameter)),
                phase_parameter=_phase_value(
                    path_state.values,
                    "phase_parameter",
                    float(parameter),
                ),
                displacement=displacement,
                contacts=tuple(samples),
            )
        )

    transitions: list[KinematicTopologyTransition] = []
    for left, right in zip(frames[:-1], frames[1:], strict=True):
        events = signature_events(
            left.signatures,
            right.signatures,
            fraction=0.5,
            branch="right",
        )
        if not events:
            continue
        transitions.append(
            KinematicTopologyTransition(
                left.parameter,
                right.parameter,
                tuple(
                    KinematicTopologyChange(
                        event.kind,
                        event.interface,
                        event.entity,
                        event.detail,
                    )
                    for event in events
                ),
            )
        )
    return KinematicTopologyScan(tuple(frames), tuple(transitions), tolerance)
