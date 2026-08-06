"""Coupled finite-strain bulk and mortar-contact equilibrium drivers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .clipping import ClippingTopologyError
from .enforcement_state import AugmentedLagrangeState, KKTDiagnostics
from .equilibrium import NewtonOptions
from .linear_solver import LinearSolveDiagnostics, solve_reduced_system
from .mechanics import (
    BulkGeometryError,
    CSRMatrix,
    DeadLoad,
    DirichletConstraints,
    FloatArray,
    IntArray,
    NeoHookeanMaterial,
    Tet4Mesh,
    Tet4SparseEvaluation,
    Tet4Sparsity,
    assemble_tet4_sparse,
)
from .pallets import PalletTopologyError
from .parametric import InverseMapTopologyError


@dataclass(frozen=True, slots=True)
class ContactBranchSignature:
    """Discrete contact branch frozen during one smooth Newton linearization."""

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
    raw: Any = field(repr=False)


@dataclass(frozen=True, slots=True)
class ContactInterfaceUpdate:
    """Accepted multiplier update for one mapped contact interface."""

    state: AugmentedLagrangeState
    increment: FloatArray
    diagnostics_after: KKTDiagnostics


@runtime_checkable
class CoupledContactInterface(Protocol):
    """Contact interface contract consumed by the global coupled solver."""

    @property
    def dofs(self) -> IntArray: ...

    def initial_state(self) -> AugmentedLagrangeState: ...

    def evaluate(
        self,
        displacement: FloatArray,
        state: AugmentedLagrangeState,
        *,
        tolerance: float,
    ) -> ContactInterfaceEvaluation: ...

    def tangent(
        self,
        displacement: FloatArray,
        state: AugmentedLagrangeState,
        evaluation: ContactInterfaceEvaluation,
        *,
        tolerance: float,
    ) -> FloatArray: ...

    def augment(
        self,
        evaluation: ContactInterfaceEvaluation,
        *,
        tolerance: float,
    ) -> ContactInterfaceUpdate: ...


@dataclass(frozen=True, slots=True)
class MortarContactInterface:
    """Map one existing mortar contact pair into global bulk-mesh DOFs."""

    pair: Any
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

    def _surface_displacements(self, displacement: FloatArray) -> tuple[FloatArray, FloatArray]:
        values = np.asarray(displacement, dtype=float).reshape((-1, 3))
        return values[self.slave_nodes], values[self.master_nodes]

    def evaluate(
        self,
        displacement: FloatArray,
        state: AugmentedLagrangeState,
        *,
        tolerance: float,
    ) -> ContactInterfaceEvaluation:
        from .enforcement_evaluation import evaluate_augmented_lagrange

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
        from .enforcement_tangent import augmented_lagrange_contact_tangent

        slave, master = self._surface_displacements(displacement)
        raw = evaluation.raw
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
        from .enforcement_evaluation import augment_multipliers

        update = augment_multipliers(self.pair, evaluation.raw, tolerance=tolerance)
        return ContactInterfaceUpdate(
            update.state,
            update.increment.copy(),
            update.diagnostics_after,
        )


@dataclass(frozen=True, slots=True)
class CoupledEquilibriumProblem:
    """Finite-strain bulk problem with one or more mapped contact interfaces."""

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


@dataclass(frozen=True, slots=True)
class CoupledEquilibriumEvaluation:
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


@dataclass(frozen=True, slots=True)
class CoupledNewtonIteration:
    iteration: int
    residual_norm: float
    relative_residual: float
    bulk_potential: float
    minimum_jacobian: float
    maximum_penetration: float
    step_norm: float
    accepted_step: float
    line_search_iterations: int
    contact_branch_changed: bool
    linear_solve: LinearSolveDiagnostics


CoupledTerminationReason = Literal[
    "converged",
    "maximum_iterations",
    "line_search_failed",
    "singular_tangent",
    "linear_solve_failed",
    "contact_linearization_event",
]
ContactEventPolicy = Literal["restart", "reject"]


@dataclass(frozen=True, slots=True)
class CoupledNewtonResult:
    displacement: FloatArray
    load_factor: float
    converged: bool
    termination_reason: CoupledTerminationReason
    evaluation: CoupledEquilibriumEvaluation
    history: tuple[CoupledNewtonIteration, ...]
    contact_event_restarts: int
    linear_solve_failure: LinearSolveDiagnostics | None = None

    @property
    def iteration_count(self) -> int:
        return len(self.history)


@dataclass(frozen=True, slots=True)
class AugmentedContactOptions:
    maximum_augmentations: int = 12
    gap_tolerance: float = 1.0e-8
    complementarity_tolerance: float = 1.0e-8
    projection_tolerance: float = 1.0e-8
    multiplier_tolerance: float = 1.0e-8
    event_policy: ContactEventPolicy = "restart"
    newton: NewtonOptions = field(default_factory=NewtonOptions)

    def __post_init__(self) -> None:
        if self.maximum_augmentations <= 0:
            raise ValueError("maximum_augmentations must be positive")
        for value in (
            self.gap_tolerance,
            self.complementarity_tolerance,
            self.projection_tolerance,
            self.multiplier_tolerance,
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("augmented-contact tolerances must be finite and nonnegative")
        if self.event_policy not in ("restart", "reject"):
            raise ValueError("event_policy must be 'restart' or 'reject'")


@dataclass(frozen=True, slots=True)
class AugmentationIteration:
    augmentation: int
    newton_iterations: int
    contact_event_restarts: int
    equilibrium_residual: float
    maximum_penetration: float
    maximum_complementarity: float
    maximum_projection_residual: float
    maximum_multiplier_increment: float
    active_rows: int
    maximum_pressure: float


AugmentedTerminationReason = Literal[
    "converged",
    "maximum_augmentations",
    "inner_equilibrium_failed",
]


@dataclass(frozen=True, slots=True)
class AugmentedContactResult:
    displacement: FloatArray
    states: tuple[AugmentedLagrangeState, ...]
    converged: bool
    termination_reason: AugmentedTerminationReason
    equilibrium: CoupledNewtonResult
    equilibria: tuple[CoupledNewtonResult, ...]
    history: tuple[AugmentationIteration, ...]


_CONTACT_EVENT_ERRORS = (
    ClippingTopologyError,
    PalletTopologyError,
    InverseMapTopologyError,
)


def _validated_states(
    problem: CoupledEquilibriumProblem,
    states: tuple[AugmentedLagrangeState, ...] | None,
) -> tuple[AugmentedLagrangeState, ...]:
    values = problem.initial_states() if states is None else tuple(states)
    if len(values) != len(problem.interfaces):
        raise ValueError("one multiplier state is required for every contact interface")
    return values


def evaluate_coupled_equilibrium(
    problem: CoupledEquilibriumProblem,
    displacement: FloatArray,
    states: tuple[AugmentedLagrangeState, ...],
    *,
    load_factor: float = 1.0,
    assemble_tangent: bool = True,
    tolerance: float = 1.0e-12,
) -> CoupledEquilibriumEvaluation:
    """Assemble bulk and contact contributions into one global equilibrium system."""

    if not np.isfinite(load_factor) or load_factor < 0.0:
        raise ValueError("load_factor must be finite and nonnegative")
    states = _validated_states(problem, states)
    total_dofs = 3 * problem.mesh.node_count
    values = np.asarray(displacement, dtype=float).reshape(-1)
    if values.shape != (total_dofs,) or not np.all(np.isfinite(values)):
        raise ValueError("displacement must be a finite global DOF vector")
    feasible = problem.constraints.apply(values)
    bulk = assemble_tet4_sparse(
        problem.mesh,
        feasible,
        problem.material,
        sparsity=problem.sparsity,
        tolerance=tolerance,
    )
    external = load_factor * problem.load.force
    residual = bulk.residual.ravel() - external
    tangent_data = bulk.tangent.data.copy() if assemble_tangent else None
    contacts: list[ContactInterfaceEvaluation] = []
    for index, (interface, state) in enumerate(
        zip(problem.interfaces, states, strict=True)
    ):
        contact = interface.evaluate(feasible, state, tolerance=tolerance)
        if contact.residual.shape != interface.dofs.shape:
            raise ValueError("contact residual does not match its mapped DOF ordering")
        if not np.all(np.isfinite(contact.residual)):
            raise ValueError("contact residual must be finite")
        np.add.at(residual, interface.dofs, contact.residual)
        if assemble_tangent:
            local_tangent = interface.tangent(
                feasible,
                state,
                contact,
                tolerance=tolerance,
            )
            expected = (len(interface.dofs), len(interface.dofs))
            if local_tangent.shape != expected:
                raise ValueError("contact tangent does not match its mapped DOF ordering")
            if not np.all(np.isfinite(local_tangent)):
                raise ValueError("contact tangent must be finite")
            np.add.at(
                tangent_data,
                problem.sparsity.additional_positions[index].ravel(),
                local_tangent.ravel(),
            )
        contacts.append(contact)
    free = problem.constraints.free_dofs(total_dofs)
    tangent = problem.sparsity.matrix(tangent_data) if assemble_tangent else None
    return CoupledEquilibriumEvaluation(
        displacement=feasible,
        load_factor=float(load_factor),
        bulk_potential=float(bulk.energy - np.dot(external, feasible)),
        residual=residual,
        tangent=tangent,
        free_dofs=free,
        free_residual_norm=float(np.linalg.norm(residual[free])),
        bulk=bulk,
        contacts=tuple(contacts),
    )


def _relative_residual(norm: float, initial_norm: float) -> float:
    return norm / max(initial_norm, np.finfo(float).tiny)


def _linear_failure_reason(
    diagnostics: LinearSolveDiagnostics,
) -> CoupledTerminationReason:
    if diagnostics.failure_reason in {"singular_matrix", "factorization_failed"}:
        return "singular_tangent"
    return "linear_solve_failed"


def solve_coupled_equilibrium(
    problem: CoupledEquilibriumProblem,
    states: tuple[AugmentedLagrangeState, ...],
    initial_displacement: FloatArray | None = None,
    *,
    load_factor: float = 1.0,
    options: NewtonOptions | None = None,
    event_policy: ContactEventPolicy = "restart",
    tolerance: float = 1.0e-12,
) -> CoupledNewtonResult:
    """Solve one fixed-multiplier equilibrium state with contact-event restarts."""

    if event_policy not in ("restart", "reject"):
        raise ValueError("event_policy must be 'restart' or 'reject'")
    settings = NewtonOptions() if options is None else options
    states = _validated_states(problem, states)
    total_dofs = 3 * problem.mesh.node_count
    displacement = (
        np.zeros(total_dofs, dtype=float)
        if initial_displacement is None
        else np.asarray(initial_displacement, dtype=float).reshape(-1).copy()
    )
    if displacement.shape != (total_dofs,):
        raise ValueError("initial_displacement must match the mesh DOF count")
    displacement = problem.constraints.apply(displacement)
    try:
        evaluation = evaluate_coupled_equilibrium(
            problem,
            displacement,
            states,
            load_factor=load_factor,
            tolerance=tolerance,
        )
    except _CONTACT_EVENT_ERRORS:
        residual_only = evaluate_coupled_equilibrium(
            problem,
            displacement,
            states,
            load_factor=load_factor,
            assemble_tangent=False,
            tolerance=tolerance,
        )
        return CoupledNewtonResult(
            displacement,
            load_factor,
            False,
            "contact_linearization_event",
            residual_only,
            (),
            0,
        )
    initial_norm = evaluation.free_residual_norm
    threshold = max(
        settings.absolute_tolerance,
        settings.relative_tolerance * initial_norm,
    )
    history: list[CoupledNewtonIteration] = []
    event_restarts = 0
    if evaluation.free_residual_norm <= threshold:
        return CoupledNewtonResult(
            displacement, load_factor, True, "converged", evaluation, (), 0
        )

    for iteration in range(settings.maximum_iterations):
        free = evaluation.free_dofs
        assert evaluation.tangent is not None
        linear_result = solve_reduced_system(
            evaluation.tangent,
            free,
            -evaluation.residual[free],
            options=settings.linear_solver,
        )
        if linear_result.solution is None:
            return CoupledNewtonResult(
                displacement,
                load_factor,
                False,
                _linear_failure_reason(linear_result.diagnostics),
                evaluation,
                tuple(history),
                event_restarts,
                linear_result.diagnostics,
            )
        step_free = linear_result.solution
        step = np.zeros(total_dofs, dtype=float)
        step[free] = step_free
        merit = 0.5 * evaluation.free_residual_norm**2
        slope = -evaluation.free_residual_norm**2
        accepted: CoupledEquilibriumEvaluation | None = None
        branch_changed = False
        alpha = 1.0
        line_iteration = 0
        for line_iteration in range(settings.maximum_line_search_iterations):  # noqa: B007 -- read after the loop as line_search_iterations
            try:
                trial = evaluate_coupled_equilibrium(
                    problem,
                    displacement + alpha * step,
                    states,
                    load_factor=load_factor,
                    assemble_tangent=False,
                    tolerance=tolerance,
                )
            except (BulkGeometryError, *_CONTACT_EVENT_ERRORS):
                trial = None
            if trial is not None:
                changed = trial.signatures != evaluation.signatures
                trial_merit = 0.5 * trial.free_residual_norm**2
                armijo = merit + settings.armijo_coefficient * alpha * slope
                acceptable = trial.free_residual_norm <= threshold or trial_merit <= armijo
                if changed and event_policy == "reject":
                    acceptable = False
                if acceptable:
                    accepted = trial
                    branch_changed = changed
                    break
            alpha *= settings.line_search_reduction
            if alpha < settings.minimum_step:
                break
        if accepted is None:
            return CoupledNewtonResult(
                displacement,
                load_factor,
                False,
                "line_search_failed",
                evaluation,
                tuple(history),
                event_restarts,
            )
        displacement = accepted.displacement.copy()
        if branch_changed:
            event_restarts += 1
        try:
            evaluation = evaluate_coupled_equilibrium(
                problem,
                displacement,
                states,
                load_factor=load_factor,
                tolerance=tolerance,
            )
        except _CONTACT_EVENT_ERRORS:
            return CoupledNewtonResult(
                displacement,
                load_factor,
                False,
                "contact_linearization_event",
                accepted,
                tuple(history),
                event_restarts,
            )
        history.append(
            CoupledNewtonIteration(
                iteration=iteration + 1,
                residual_norm=evaluation.free_residual_norm,
                relative_residual=_relative_residual(
                    evaluation.free_residual_norm, initial_norm
                ),
                bulk_potential=evaluation.bulk_potential,
                minimum_jacobian=evaluation.bulk.minimum_jacobian,
                maximum_penetration=evaluation.maximum_penetration,
                step_norm=float(np.linalg.norm(step_free)),
                accepted_step=alpha,
                line_search_iterations=line_iteration,
                contact_branch_changed=branch_changed,
                linear_solve=linear_result.diagnostics,
            )
        )
        if evaluation.free_residual_norm <= threshold:
            return CoupledNewtonResult(
                displacement,
                load_factor,
                True,
                "converged",
                evaluation,
                tuple(history),
                event_restarts,
            )
    return CoupledNewtonResult(
        displacement,
        load_factor,
        False,
        "maximum_iterations",
        evaluation,
        tuple(history),
        event_restarts,
    )


def _all_kkt_converged(
    contacts: tuple[ContactInterfaceEvaluation, ...],
    options: AugmentedContactOptions,
) -> bool:
    return all(
        contact.diagnostics.converged(
            gap_tolerance=options.gap_tolerance,
            complementarity_tolerance=options.complementarity_tolerance,
            projection_tolerance=options.projection_tolerance,
            multiplier_tolerance=options.multiplier_tolerance,
        )
        for contact in contacts
    )


def solve_augmented_contact(
    problem: CoupledEquilibriumProblem,
    initial_displacement: FloatArray | None = None,
    initial_states: tuple[AugmentedLagrangeState, ...] | None = None,
    *,
    load_factor: float = 1.0,
    options: AugmentedContactOptions | None = None,
    tolerance: float = 1.0e-12,
) -> AugmentedContactResult:
    """Alternate fixed-multiplier equilibrium solves and accepted AL updates."""

    settings = AugmentedContactOptions() if options is None else options
    states = _validated_states(problem, initial_states)
    displacement = initial_displacement
    history: list[AugmentationIteration] = []
    last_equilibrium: CoupledNewtonResult | None = None
    equilibria: list[CoupledNewtonResult] = []
    for augmentation in range(settings.maximum_augmentations):
        equilibrium = solve_coupled_equilibrium(
            problem,
            states,
            displacement,
            load_factor=load_factor,
            options=settings.newton,
            event_policy=settings.event_policy,
            tolerance=tolerance,
        )
        last_equilibrium = equilibrium
        equilibria.append(equilibrium)
        if not equilibrium.converged:
            return AugmentedContactResult(
                equilibrium.displacement,
                states,
                False,
                "inner_equilibrium_failed",
                equilibrium,
                tuple(equilibria),
                tuple(history),
            )
        displacement = equilibrium.displacement
        contacts = equilibrium.evaluation.contacts
        if _all_kkt_converged(contacts, settings):
            history.append(
                AugmentationIteration(
                    augmentation=augmentation,
                    newton_iterations=equilibrium.iteration_count,
                    contact_event_restarts=equilibrium.contact_event_restarts,
                    equilibrium_residual=equilibrium.evaluation.free_residual_norm,
                    maximum_penetration=max(
                        (c.diagnostics.maximum_penetration for c in contacts), default=0.0
                    ),
                    maximum_complementarity=max(
                        (c.diagnostics.maximum_complementarity for c in contacts), default=0.0
                    ),
                    maximum_projection_residual=max(
                        (c.diagnostics.maximum_projection_residual for c in contacts),
                        default=0.0,
                    ),
                    maximum_multiplier_increment=0.0,
                    active_rows=sum(
                        int(np.count_nonzero(contact.signature.active_rows))
                        for contact in contacts
                    ),
                    maximum_pressure=max(
                        (
                            float(np.max(contact.pressure, initial=0.0))
                            for contact in contacts
                        ),
                        default=0.0,
                    ),
                )
            )
            return AugmentedContactResult(
                displacement,
                states,
                True,
                "converged",
                equilibrium,
                tuple(equilibria),
                tuple(history),
            )
        updates = tuple(
            interface.augment(contact, tolerance=tolerance)
            for interface, contact in zip(problem.interfaces, contacts, strict=True)
        )
        increment = max(
            (float(np.max(np.abs(update.increment), initial=0.0)) for update in updates),
            default=0.0,
        )
        history.append(
            AugmentationIteration(
                augmentation=augmentation,
                newton_iterations=equilibrium.iteration_count,
                contact_event_restarts=equilibrium.contact_event_restarts,
                equilibrium_residual=equilibrium.evaluation.free_residual_norm,
                maximum_penetration=max(
                    (c.diagnostics.maximum_penetration for c in contacts), default=0.0
                ),
                maximum_complementarity=max(
                    (c.diagnostics.maximum_complementarity for c in contacts), default=0.0
                ),
                maximum_projection_residual=max(
                    (c.diagnostics.maximum_projection_residual for c in contacts),
                    default=0.0,
                ),
                maximum_multiplier_increment=increment,
                active_rows=sum(
                    int(np.count_nonzero(contact.signature.active_rows))
                    for contact in contacts
                ),
                maximum_pressure=max(
                    (
                        float(np.max(contact.pressure, initial=0.0))
                        for contact in contacts
                    ),
                    default=0.0,
                ),
            )
        )
        if augmentation + 1 == settings.maximum_augmentations:
            break
        states = tuple(update.state for update in updates)
    assert last_equilibrium is not None
    return AugmentedContactResult(
        last_equilibrium.displacement,
        states,
        False,
        "maximum_augmentations",
        last_equilibrium,
        tuple(equilibria),
        tuple(history),
    )
