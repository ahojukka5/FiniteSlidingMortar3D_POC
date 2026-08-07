"""Global assembly of mechanics and mapped contact contributions."""

from __future__ import annotations

import numpy as np

from ..mechanics import FloatArray, assemble_tet4_sparse
from ..mortar.enforcement import AugmentedLagrangeState
from .problem import CoupledEquilibriumProblem
from .results import ContactInterfaceEvaluation, CoupledEquilibriumEvaluation


def evaluate_coupled_equilibrium(
    problem: CoupledEquilibriumProblem,
    displacement: FloatArray,
    states: tuple[AugmentedLagrangeState, ...],
    *,
    load_factor: float = 1.0,
    assemble_tangent: bool = True,
    tolerance: float = 1.0e-12,
) -> CoupledEquilibriumEvaluation:
    """Assemble bulk and contact contributions into one global system."""

    if not np.isfinite(load_factor) or load_factor < 0.0:
        raise ValueError("load_factor must be finite and nonnegative")
    states = problem.validate_states(states)
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
            assert tangent_data is not None
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
