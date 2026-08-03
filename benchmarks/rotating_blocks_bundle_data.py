"""Checkpoint selection and tabular data for rotating-blocks artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rotating_blocks_model import RotatingBlocksModel
from rotating_blocks_solver import RotatingBlocksSolverRun

from contact3d import build_facet_overlap
from contact3d.coupled import evaluate_coupled_equilibrium


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """One selected physical state for table and VTK export."""

    name: str
    target: float
    parameter: float
    displacement: np.ndarray
    states: tuple[object, ...]
    evaluation: object
    reaction: np.ndarray
    path_state: object

    @property
    def selection_error(self) -> float:
        return abs(self.parameter - self.target)


def _reaction(evaluation: object) -> np.ndarray:
    residual = np.asarray(evaluation.residual, dtype=float).reshape(-1)
    free = np.asarray(evaluation.free_dofs, dtype=np.int64).reshape(-1)
    values = np.zeros_like(residual)
    constrained = np.ones(len(residual), dtype=bool)
    constrained[free] = False
    values[constrained] = residual[constrained]
    return values


def _initial_checkpoint(model: RotatingBlocksModel) -> Checkpoint:
    state = model.path.evaluate(model.problem, 0.0)
    displacement = np.zeros(3 * state.problem.mesh.node_count, dtype=float)
    displacement[state.prescribed_dofs] = state.prescribed_values
    states = tuple(state.problem.initial_states())
    evaluation = evaluate_coupled_equilibrium(
        state.problem,
        displacement,
        states,
        load_factor=state.solver_load_factor,
        assemble_tangent=False,
    )
    return Checkpoint(
        "pre-contact",
        0.0,
        0.0,
        evaluation.displacement.copy(),
        states,
        evaluation,
        _reaction(evaluation),
        state,
    )


def _accepted_checkpoint(name: str, target: float, step: object) -> Checkpoint:
    result = step.result
    evaluation = result.equilibrium.evaluation
    reaction = getattr(step, "reaction", None)
    return Checkpoint(
        name,
        target,
        float(step.parameter),
        np.asarray(result.displacement, dtype=float).reshape(-1).copy(),
        tuple(result.states),
        evaluation,
        _reaction(evaluation)
        if reaction is None
        else np.asarray(reaction, dtype=float).reshape(-1).copy(),
        step.path_state,
    )


def select_checkpoints(
    model: RotatingBlocksModel,
    completed: RotatingBlocksSolverRun,
) -> tuple[Checkpoint, ...]:
    """Select pre-contact, compressed, intermediate, and final states."""

    steps = tuple(getattr(completed.result, "accepted_steps", ()))
    if not steps:
        raise ValueError("rotating-blocks result contains no accepted steps")
    compression = float(model.geometry.compression_end)
    targets = (
        ("compressed", compression),
        ("mid-rotation", compression + 0.5 * (1.0 - compression)),
        ("final", float(model.path.end_parameter)),
    )
    checkpoints = [_initial_checkpoint(model)]
    for name, target in targets:
        step = min(
            steps,
            key=lambda item: (abs(float(item.parameter) - target), float(item.parameter)),
        )
        checkpoints.append(_accepted_checkpoint(name, target, step))
    return tuple(checkpoints)


def contact(checkpoint: Checkpoint) -> object:
    """Return the single contact evaluation retained by a checkpoint."""

    contacts = tuple(checkpoint.evaluation.contacts)
    if len(contacts) != 1:
        raise ValueError("rotating-blocks bundle requires one contact interface")
    return contacts[0]


def multiplier(checkpoint: Checkpoint) -> np.ndarray:
    """Return the accepted multiplier vector for a checkpoint."""

    if len(checkpoint.states) != 1:
        raise ValueError("rotating-blocks bundle requires one multiplier state")
    return np.asarray(checkpoint.states[0].multipliers, dtype=float)


def bulk_fields(checkpoint: Checkpoint) -> dict[str, np.ndarray]:
    """Return element-quality fields used by volume checkpoint output."""

    evaluations = tuple(checkpoint.evaluation.bulk.element_evaluations)
    return {
        "jacobian": np.asarray([item.jacobian for item in evaluations]),
        "energy_density": np.asarray([item.energy_density for item in evaluations]),
    }


def global_contact_force(checkpoint: Checkpoint) -> np.ndarray:
    """Scatter the local interface force into global displacement ordering."""

    force = np.zeros_like(checkpoint.displacement)
    interface = checkpoint.path_state.problem.interfaces[0]
    np.add.at(force, interface.dofs, contact(checkpoint).residual)
    return force


def checkpoint_rows(
    checkpoints: Sequence[Checkpoint],
) -> tuple[dict[str, object], ...]:
    """Return one summary row for every selected physical checkpoint."""

    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        displacement = checkpoint.displacement.reshape((-1, 3))
        evaluation = contact(checkpoint)
        rows.append(
            {
                "checkpoint": checkpoint.name,
                "target_parameter": checkpoint.target,
                "selected_parameter": checkpoint.parameter,
                "selection_error": checkpoint.selection_error,
                "maximum_displacement": float(
                    np.max(np.linalg.norm(displacement, axis=1), initial=0.0)
                ),
                "maximum_pressure": float(
                    np.max(evaluation.pressure, initial=0.0)
                ),
                "overlap_area": float(
                    evaluation.raw.contact.weights.total_area
                ),
            }
        )
    return tuple(rows)


def interface_rows(
    checkpoints: Sequence[Checkpoint],
) -> tuple[dict[str, object], ...]:
    """Return nodal gap, pressure, multiplier, support, and force rows."""

    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        evaluation = contact(checkpoint)
        multipliers = multiplier(checkpoint)
        forces = np.asarray(evaluation.residual, dtype=float).reshape((-1, 3))
        slave_forces = forces[: len(evaluation.pressure)]
        row_areas = np.asarray(
            evaluation.raw.contact.weights.row_areas,
            dtype=float,
        )
        for node in range(len(evaluation.pressure)):
            rows.append(
                {
                    "checkpoint": checkpoint.name,
                    "parameter": checkpoint.parameter,
                    "node": node,
                    "normal_gap": float(evaluation.normal_gaps[node]),
                    "pressure": float(evaluation.pressure[node]),
                    "multiplier": float(multipliers[node]),
                    "row_area": float(row_areas[node]),
                    "supported": bool(evaluation.signature.supported_rows[node]),
                    "active": bool(evaluation.signature.active_rows[node]),
                    "contact_force_x": float(slave_forces[node, 0]),
                    "contact_force_y": float(slave_forces[node, 1]),
                    "contact_force_z": float(slave_forces[node, 2]),
                }
            )
    return tuple(rows)


def pair_rows(checkpoints: Sequence[Checkpoint]) -> tuple[dict[str, object], ...]:
    """Return facet-pair overlap areas at every checkpoint."""

    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        evaluation = contact(checkpoint)
        pairs = tuple(evaluation.signature.facet_pairs)
        areas = np.asarray(
            evaluation.raw.contact.weights.overlap_areas,
            dtype=float,
        )
        if len(pairs) != len(areas):
            raise ValueError("facet pairs and overlap areas are not aligned")
        values = zip(pairs, areas, strict=True) if pairs else [((-1, -1), 0.0)]
        for index, ((slave, master), area) in enumerate(values):
            rows.append(
                {
                    "checkpoint": checkpoint.name,
                    "parameter": checkpoint.parameter,
                    "pair": index if pairs else -1,
                    "slave_facet": int(slave),
                    "master_facet": int(master),
                    "overlap_area": float(area),
                }
            )
    return tuple(rows)


def _area(polygon: np.ndarray) -> float:
    shifted = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon[:, 1]
    return 0.5 * abs(float(np.sum(cross)))


def projected_regions(
    model: RotatingBlocksModel,
    checkpoint: Checkpoint,
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[dict[str, object], ...],
    tuple[tuple[np.ndarray, str], ...],
]:
    """Return projected slave, master, and clipped polygons."""

    interface = checkpoint.path_state.problem.interfaces[0]
    pair = interface.pair
    displacement = checkpoint.displacement.reshape((-1, 3))
    slave = pair.slave.current_nodes(displacement[model.slave_nodes])
    master = pair.master.current_nodes(displacement[model.master_nodes])
    pairs = tuple(contact(checkpoint).signature.facet_pairs)
    if not pairs:
        pairs = tuple((index, -1) for index in range(len(pair.slave.facets)))

    points: list[np.ndarray] = []
    facets: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    overlays: list[tuple[np.ndarray, str]] = []
    for pair_index, (slave_index, master_index) in enumerate(pairs):
        slave_points = slave[pair.slave.facets[slave_index]]
        if master_index >= 0:
            master_points = master[pair.master.facets[master_index]]
            overlap = build_facet_overlap(slave_points, master_points)
            regions = (
                ("slave", 0, overlap.slave_polygon),
                ("master", 1, overlap.master_polygon),
                ("intersection", 2, overlap.intersection_polygon),
            )
        else:
            overlap = build_facet_overlap(slave_points, slave_points)
            regions = (("slave", 0, overlap.slave_polygon),)
        for region, kind, polygon in regions:
            polygon = np.asarray(polygon, dtype=float)
            if len(polygon) < 3:
                continue
            start = len(points)
            points.extend(np.column_stack([polygon, np.zeros(len(polygon))]))
            facets.append(np.arange(start, start + len(polygon), dtype=np.int64))
            rows.append(
                {
                    "checkpoint": checkpoint.name,
                    "parameter": checkpoint.parameter,
                    "pair": pair_index,
                    "slave_facet": int(slave_index),
                    "master_facet": int(master_index),
                    "region": region,
                    "region_kind": kind,
                    "vertex_count": len(polygon),
                    "projected_area": _area(polygon),
                }
            )
            overlays.append((polygon, f"{region} {pair_index}"))
    if not facets:
        raise ValueError(f"checkpoint {checkpoint.name} has no projected polygons")
    return np.asarray(points), tuple(facets), tuple(rows), tuple(overlays)
