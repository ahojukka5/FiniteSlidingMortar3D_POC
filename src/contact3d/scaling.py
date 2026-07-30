"""Characteristic scales and dimensionless diagnostics for coupled contact."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import numpy as np

from .bulk_material import NeoHookeanMaterial
from .enforcement_state import KKTDiagnostics
from .model import FloatArray


@runtime_checkable
class PenaltyControlledContactInterface(Protocol):
    """Explicit normal-penalty and reference-geometry control contract."""

    @property
    def normal_penalty(self) -> float: ...

    def with_normal_penalty(self, normal_penalty: float) -> object: ...

    def reference_tributary_areas(self) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class ScaleAwareConvergenceOptions:
    """Dimensionless Newton and KKT stopping tolerances.

    ``enabled=False`` preserves the dimensional stopping rules used before the
    scale-aware solver was introduced. Enabling this record interprets all
    tolerances below in normalized units.
    """

    enabled: bool = False
    equilibrium_tolerance: float = 1.0e-10
    gap_tolerance: float = 1.0e-8
    complementarity_tolerance: float = 1.0e-8
    projection_tolerance: float = 1.0e-8
    multiplier_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        for name, value in (
            ("equilibrium_tolerance", self.equilibrium_tolerance),
            ("gap_tolerance", self.gap_tolerance),
            ("complementarity_tolerance", self.complementarity_tolerance),
            ("projection_tolerance", self.projection_tolerance),
            ("multiplier_tolerance", self.multiplier_tolerance),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class NormalizedKKTDiagnostics:
    """Dimensionless unilateral-contact residual maxima."""

    maximum_penetration: float
    maximum_multiplier_violation: float
    maximum_complementarity: float
    maximum_projection_residual: float
    maximum_unsupported_multiplier: float

    def converged(self, options: ScaleAwareConvergenceOptions) -> bool:
        return (
            self.maximum_penetration <= options.gap_tolerance
            and self.maximum_multiplier_violation <= options.multiplier_tolerance
            and self.maximum_complementarity <= options.complementarity_tolerance
            and self.maximum_projection_residual <= options.projection_tolerance
            and self.maximum_unsupported_multiplier <= options.multiplier_tolerance
        )


@dataclass(frozen=True, slots=True)
class ContactInterfaceScales:
    """Reference scales for one mapped contact interface."""

    length: float
    area: float
    pressure: float
    force: float
    energy: float
    penalty: float
    tributary_areas: FloatArray
    row_lengths: FloatArray

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("area", self.area),
            ("pressure", self.pressure),
            ("force", self.force),
            ("energy", self.energy),
            ("penalty", self.penalty),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"interface {name} scale must be finite and positive")
        areas = np.asarray(self.tributary_areas, dtype=float)
        lengths = np.asarray(self.row_lengths, dtype=float)
        if areas.ndim != 1 or lengths.shape != areas.shape or len(areas) == 0:
            raise ValueError("interface tributary scales must be aligned vectors")
        if not np.all(np.isfinite(areas)) or not np.all(np.isfinite(lengths)):
            raise ValueError("interface tributary scales must be finite")
        if np.any(areas <= 0.0) or np.any(lengths <= 0.0):
            raise ValueError("interface tributary scales must be positive")
        object.__setattr__(self, "tributary_areas", areas.copy())
        object.__setattr__(self, "row_lengths", lengths.copy())

    def normalized_penetration(self, value: float) -> float:
        return float(value) / self.length

    def normalized_pressure(self, value: float) -> float:
        return float(value) / self.pressure

    def normalized_force(self, value: float) -> float:
        return float(value) / self.force

    def penalty_ratio(self, value: float) -> float:
        return float(value) / self.penalty

    def normalize_kkt(self, diagnostics: KKTDiagnostics) -> NormalizedKKTDiagnostics:
        return NormalizedKKTDiagnostics(
            maximum_penetration=diagnostics.maximum_penetration / self.length,
            maximum_multiplier_violation=(
                diagnostics.maximum_multiplier_violation / self.pressure
            ),
            maximum_complementarity=(
                diagnostics.maximum_complementarity / (self.pressure * self.length)
            ),
            maximum_projection_residual=(
                diagnostics.maximum_projection_residual / self.pressure
            ),
            maximum_unsupported_multiplier=(
                diagnostics.maximum_unsupported_multiplier / self.pressure
            ),
        )

    def penalty_bounds(
        self,
        *,
        minimum_factor: float,
        maximum_factor: float,
        absolute_maximum: float,
    ) -> tuple[float, float]:
        if minimum_factor <= 0.0 or maximum_factor < minimum_factor:
            raise ValueError("penalty factors must satisfy 0 < minimum <= maximum")
        if absolute_maximum <= 0.0:
            raise ValueError("absolute maximum penalty must be positive")
        lower_natural = self.pressure / float(np.max(self.row_lengths))
        upper_natural = self.pressure / float(np.min(self.row_lengths))
        lower = minimum_factor * lower_natural
        upper = min(absolute_maximum, maximum_factor * upper_natural)
        if upper < lower:
            lower = upper
        return float(lower), float(upper)


@dataclass(frozen=True, slots=True)
class CoupledProblemScales:
    """Characteristic bulk and interface scales for a coupled problem."""

    length: float
    pressure: float
    force: float
    energy: float
    interfaces: tuple[ContactInterfaceScales, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("length", self.length),
            ("pressure", self.pressure),
            ("force", self.force),
            ("energy", self.energy),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"problem {name} scale must be finite and positive")
        if not self.interfaces:
            raise ValueError("coupled problem scales require at least one interface")

    def normalized_equilibrium_residual(self, value: float) -> float:
        return float(value) / self.force


@dataclass(frozen=True, slots=True)
class ContactScaleIndicators:
    """Dimensional and normalized state of one contact interface."""

    penetration: float
    normalized_penetration: float
    pressure: float
    normalized_pressure: float
    penalty: float
    penalty_ratio: float
    active_rows: int
    supported_rows: int


@dataclass(frozen=True, slots=True)
class PenaltyUpdateDecision:
    """One interface-local normal-penalty update decision."""

    interface: int
    old_penalty: float
    new_penalty: float
    penetration: float
    normalized_penetration: float
    old_ratio: float
    new_ratio: float
    reason: str


@dataclass(frozen=True, slots=True)
class PenaltyUpdatePlan:
    """Complete immutable penalty proposal in interface order."""

    penalties: tuple[float, ...]
    decisions: tuple[PenaltyUpdateDecision, ...]

    @property
    def changed(self) -> bool:
        return bool(self.decisions)

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(decision.reason for decision in self.decisions)


def material_pressure_scale(material: NeoHookeanMaterial) -> float:
    """Return Young's modulus reconstructed from shear and bulk moduli."""

    shear = float(material.shear_modulus)
    bulk = float(material.bulk_modulus)
    return 9.0 * bulk * shear / (3.0 * bulk + shear)


def _triangle_area(points: FloatArray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])))


def _surface_tributary_areas(surface: object) -> FloatArray:
    nodes = np.asarray(surface.reference_nodes, dtype=float)
    facets = tuple(np.asarray(facet, dtype=np.int64) for facet in surface.facets)
    areas = np.zeros(len(nodes), dtype=float)
    for facet in facets:
        points = nodes[facet]
        area = sum(
            _triangle_area(points[[0, local, local + 1]])
            for local in range(1, len(points) - 1)
        )
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError("contact reference facet must have positive area")
        areas[facet] += area / len(facet)
    if np.any(areas <= 0.0):
        raise ValueError("every contact slave node must have positive tributary area")
    return areas


def _explicit_mortar_interface(interface: object) -> bool:
    from .coupled import MortarContactInterface

    return isinstance(interface, MortarContactInterface)


def _explicit_frozen_interface(interface: object) -> bool:
    try:
        from .coupled_oracle import FrozenMatchingMortarInterface
    except ImportError:
        return False
    return isinstance(interface, FrozenMatchingMortarInterface)


def interface_normal_penalty(interface: object) -> float:
    """Read a normal penalty through the explicit control protocol or known adapters."""

    if isinstance(interface, PenaltyControlledContactInterface):
        value = float(interface.normal_penalty)
    elif _explicit_mortar_interface(interface):
        value = float(interface.pair.normal_penalty)
    elif _explicit_frozen_interface(interface):
        value = float(interface.penalty)
    else:
        raise TypeError(
            "contact interface must implement PenaltyControlledContactInterface"
        )
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("contact-interface penalty must be finite and positive")
    return value


def with_interface_normal_penalty(interface: object, penalty: float) -> object:
    """Return an immutable interface with a replaced normal penalty."""

    value = float(penalty)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("contact-interface penalty must be finite and positive")
    if isinstance(interface, PenaltyControlledContactInterface):
        return interface.with_normal_penalty(value)
    if _explicit_mortar_interface(interface):
        return replace(interface, pair=replace(interface.pair, normal_penalty=value))
    if _explicit_frozen_interface(interface):
        return replace(interface, penalty=value)
    raise TypeError("contact interface does not implement normal-penalty replacement")


def interface_reference_tributary_areas(interface: object) -> FloatArray:
    """Return positive slave-row tributary areas through an explicit adapter."""

    if isinstance(interface, PenaltyControlledContactInterface):
        values = np.asarray(interface.reference_tributary_areas(), dtype=float)
    elif _explicit_mortar_interface(interface):
        values = _surface_tributary_areas(interface.pair.slave)
    elif _explicit_frozen_interface(interface):
        from .coupled_oracle import MATCHING_QUAD_MASS

        values = np.sum(MATCHING_QUAD_MASS, axis=1)
    else:
        raise TypeError(
            "contact interface must expose reference tributary areas for scaling"
        )
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("reference tributary areas must be a nonempty vector")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("reference tributary areas must be finite and positive")
    return values.copy()


def contact_interface_scales(
    interface: object,
    material: NeoHookeanMaterial,
) -> ContactInterfaceScales:
    areas = interface_reference_tributary_areas(interface)
    total_area = float(np.sum(areas))
    row_lengths = np.sqrt(areas)
    length = float(np.sqrt(total_area))
    pressure = material_pressure_scale(material)
    force = pressure * total_area
    energy = force * length
    penalty = pressure / float(np.median(row_lengths))
    return ContactInterfaceScales(
        length,
        total_area,
        pressure,
        force,
        energy,
        penalty,
        areas,
        row_lengths,
    )


def coupled_problem_scales(problem: object) -> CoupledProblemScales:
    """Construct unit-consistent characteristic scales from reference data."""

    nodes = np.asarray(problem.mesh.reference_nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) == 0:
        raise ValueError("coupled mesh reference nodes must have shape (n, 3)")
    span = np.ptp(nodes, axis=0)
    length = float(np.linalg.norm(span))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("coupled reference mesh must have a positive spatial extent")
    pressure = material_pressure_scale(problem.material)
    interface_scales = tuple(
        contact_interface_scales(interface, problem.material)
        for interface in problem.interfaces
    )
    force = max(
        pressure * length**2,
        sum(scale.force for scale in interface_scales),
    )
    energy = max(
        pressure * length**3,
        sum(scale.energy for scale in interface_scales),
    )
    return CoupledProblemScales(length, pressure, force, energy, interface_scales)


def contact_scale_indicators(
    contact: object,
    scale: ContactInterfaceScales,
    penalty: float,
) -> ContactScaleIndicators:
    diagnostics = contact.diagnostics
    pressure = float(np.max(np.asarray(contact.pressure, dtype=float), initial=0.0))
    return ContactScaleIndicators(
        penetration=diagnostics.maximum_penetration,
        normalized_penetration=(diagnostics.maximum_penetration / scale.length),
        pressure=pressure,
        normalized_pressure=pressure / scale.pressure,
        penalty=float(penalty),
        penalty_ratio=scale.penalty_ratio(penalty),
        active_rows=int(np.count_nonzero(contact.signature.active_rows)),
        supported_rows=int(np.count_nonzero(contact.signature.supported_rows)),
    )


def propose_interface_penalties(
    problem: object,
    contacts: tuple[object, ...],
    *,
    increase_factor: float,
    absolute_maximum: float,
    minimum_scale_factor: float,
    maximum_scale_factor: float,
    dimensional_target: float | None,
    normalized_target: float,
    use_normalized_target: bool,
    interface_local: bool,
) -> PenaltyUpdatePlan:
    """Propose bounded normal-penalty increases only where penetration is unresolved."""

    if len(contacts) != len(problem.interfaces):
        raise ValueError("one contact evaluation is required for every interface")
    if increase_factor <= 1.0:
        raise ValueError("penalty increase factor must exceed one")
    if normalized_target < 0.0:
        raise ValueError("normalized penetration target must be nonnegative")
    if dimensional_target is not None and dimensional_target < 0.0:
        raise ValueError("dimensional penetration target must be nonnegative")

    scales = coupled_problem_scales(problem)
    old = tuple(interface_normal_penalty(interface) for interface in problem.interfaces)
    unresolved: list[bool] = []
    penetrations: list[float] = []
    normalized: list[float] = []
    for contact, scale in zip(contacts, scales.interfaces, strict=True):
        penetration = float(contact.diagnostics.maximum_penetration)
        normalized_penetration = penetration / scale.length
        penetrations.append(penetration)
        normalized.append(normalized_penetration)
        if use_normalized_target:
            unresolved.append(normalized_penetration > normalized_target)
        else:
            target = 0.0 if dimensional_target is None else dimensional_target
            unresolved.append(penetration > target)

    if not interface_local and any(unresolved):
        unresolved = [True] * len(unresolved)

    proposed = list(old)
    decisions: list[PenaltyUpdateDecision] = []
    for index, (needs_update, current, scale) in enumerate(
        zip(unresolved, old, scales.interfaces, strict=True)
    ):
        if not needs_update:
            continue
        lower, upper = scale.penalty_bounds(
            minimum_factor=minimum_scale_factor,
            maximum_factor=maximum_scale_factor,
            absolute_maximum=absolute_maximum,
        )
        candidate = min(upper, max(lower, increase_factor * current))
        if candidate <= current:
            continue
        proposed[index] = candidate
        target_name = "normalized_penetration" if use_normalized_target else "penetration"
        target_value = normalized_target if use_normalized_target else dimensional_target
        reason = (
            f"interface[{index}] {target_name}="
            f"{normalized[index] if use_normalized_target else penetrations[index]:.6e} "
            f"> {float(target_value):.6e}; penalty_ratio "
            f"{scale.penalty_ratio(current):.6e} -> {scale.penalty_ratio(candidate):.6e}"
        )
        decisions.append(
            PenaltyUpdateDecision(
                interface=index,
                old_penalty=current,
                new_penalty=candidate,
                penetration=penetrations[index],
                normalized_penetration=normalized[index],
                old_ratio=scale.penalty_ratio(current),
                new_ratio=scale.penalty_ratio(candidate),
                reason=reason,
            )
        )
    return PenaltyUpdatePlan(tuple(proposed), tuple(decisions))
