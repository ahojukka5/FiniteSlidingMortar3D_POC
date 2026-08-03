"""Checkpoint selection and tabular data for rotating-blocks artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from rotating_blocks_model import RotatingBlocksModel
from rotating_blocks_solver import RotatingBlocksSolverRun

from contact3d import build_facet_overlap
from contact3d.coupled import evaluate_coupled_equilibrium

CHECKPOINT_NAMES = (
    "pre-contact",
    "first-contact",
    "compressed",
    "quarter-rotation",
    "half-rotation",
    "final",
)


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
    accepted_step: int | None
    selection_rule: str

    @property
    def selection_error(self) -> float:
        return abs(self.parameter - self.target)


@dataclass(frozen=True, slots=True)
class CheckpointSelection:
    """Selected checkpoints plus explicit records for missing regimes."""

    checkpoints: tuple[Checkpoint, ...]
    requests: tuple[dict[str, object], ...]

    @property
    def complete(self) -> bool:
        return all(bool(request["present"]) for request in self.requests)

    @property
    def missing(self) -> tuple[dict[str, object], ...]:
        return tuple(request for request in self.requests if not request["present"])


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
        None,
        "reference state before the staged motion",
    )


def _accepted_checkpoint(
    name: str,
    target: float,
    source: Mapping[str, object],
    step: object,
    selection_rule: str,
) -> Checkpoint:
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
        int(source["accepted_step"]),
        selection_rule,
    )


def _accepted_records(
    completed: RotatingBlocksSolverRun,
) -> tuple[tuple[dict[str, object], object], ...]:
    rows = tuple(dict(row) for row in getattr(completed, "accepted_rows", ()))
    steps = tuple(getattr(getattr(completed, "result", None), "accepted_steps", ()))
    if len(rows) != len(steps):
        raise ValueError("accepted-step rows and solver states must be aligned")
    records: list[tuple[dict[str, object], object]] = []
    for ordinal, (row, step) in enumerate(zip(rows, steps, strict=True), start=1):
        row.setdefault("accepted_step", ordinal)
        row_parameter = float(row["parameter"])
        step_parameter = float(step.parameter)
        if not np.isclose(row_parameter, step_parameter, rtol=0.0, atol=1.0e-12):
            raise ValueError("accepted-step table does not match solver state parameter")
        records.append((row, step))
    return tuple(records)


def _nearest_record(
    records: Sequence[tuple[dict[str, object], object]],
    target: float,
) -> tuple[dict[str, object], object] | None:
    if not records:
        return None
    return min(
        records,
        key=lambda record: (
            abs(float(record[0]["parameter"]) - target),
            float(record[0]["parameter"]),
            int(record[0]["accepted_step"]),
        ),
    )


def _request(
    name: str,
    target: float,
    selection_rule: str,
    checkpoint: Checkpoint | None,
    missing_reason: str = "",
) -> dict[str, object]:
    return {
        "checkpoint": name,
        "target_parameter": target,
        "selection_rule": selection_rule,
        "present": checkpoint is not None,
        "missing_reason": "" if checkpoint is not None else missing_reason,
    }


def select_checkpoint_regimes(
    model: RotatingBlocksModel,
    completed: RotatingBlocksSolverRun,
) -> CheckpointSelection:
    """Select all requested physical regimes and retain missing-state records."""

    records = _accepted_records(completed)
    compression = float(model.geometry.compression_end)
    end_parameter = float(model.path.end_parameter)
    quarter_rotation = compression + 0.25 * (end_parameter - compression)
    half_rotation = compression + 0.50 * (end_parameter - compression)
    checkpoints: list[Checkpoint] = [_initial_checkpoint(model)]
    requests: list[dict[str, object]] = [
        _request(
            "pre-contact",
            0.0,
            "reference state before the staged motion",
            checkpoints[0],
        )
    ]

    first_rule = "earliest accepted state with overlap and active pressure"
    first_candidates = tuple(
        record
        for record in records
        if float(record[0].get("overlap_area", 0.0)) > 0.0
        and int(record[0].get("supported_rows", 0)) > 0
        and (
            int(record[0].get("active_rows", 0)) > 0
            or float(record[0].get("maximum_pressure", 0.0)) > 0.0
        )
    )
    first_record = min(
        first_candidates,
        key=lambda record: (
            float(record[0]["parameter"]),
            int(record[0]["accepted_step"]),
        ),
        default=None,
    )
    first = (
        None
        if first_record is None
        else _accepted_checkpoint(
            "first-contact",
            float(first_record[0]["parameter"]),
            first_record[0],
            first_record[1],
            first_rule,
        )
    )
    if first is not None:
        checkpoints.append(first)
    requests.append(
        _request(
            "first-contact",
            float(first_record[0]["parameter"]) if first_record is not None else 0.0,
            first_rule,
            first,
            "no accepted state contains overlap with active contact pressure",
        )
    )

    compressed_rule = "nearest accepted state to the compression phase boundary"
    compressed_record = _nearest_record(records, compression)
    compressed = (
        None
        if compressed_record is None
        else _accepted_checkpoint(
            "compressed",
            compression,
            compressed_record[0],
            compressed_record[1],
            compressed_rule,
        )
    )
    if compressed is not None:
        checkpoints.append(compressed)
    requests.append(
        _request(
            "compressed",
            compression,
            compressed_rule,
            compressed,
            "no accepted nonlinear state is available",
        )
    )

    rotation_records = tuple(
        record for record in records if int(record[0].get("phase_index", -1)) == 1
    )
    for name, fraction, target in (
        ("quarter-rotation", 0.25, quarter_rotation),
        ("half-rotation", 0.50, half_rotation),
    ):
        rule = f"nearest accepted rotation state to {fraction:.0%} rotation"
        record = _nearest_record(rotation_records, target)
        checkpoint = (
            None
            if record is None
            else _accepted_checkpoint(
                name,
                target,
                record[0],
                record[1],
                rule,
            )
        )
        if checkpoint is not None:
            checkpoints.append(checkpoint)
        requests.append(
            _request(
                name,
                target,
                rule,
                checkpoint,
                "no accepted rotation-phase state is available",
            )
        )

    final_rule = "accepted state at the completed prescribed motion"
    final_record = _nearest_record(records, end_parameter)
    final = None
    if final_record is not None:
        tolerance = max(
            float(getattr(getattr(completed, "profile", None), "minimum_step", 0.0)),
            1.0e-12,
        )
        if np.isclose(
            float(final_record[0]["parameter"]),
            end_parameter,
            rtol=0.0,
            atol=tolerance,
        ):
            final = _accepted_checkpoint(
                "final",
                end_parameter,
                final_record[0],
                final_record[1],
                final_rule,
            )
    if final is not None:
        checkpoints.append(final)
    requests.append(
        _request(
            "final",
            end_parameter,
            final_rule,
            final,
            "the accepted path does not reach the prescribed final parameter",
        )
    )

    ordered = tuple(
        checkpoint
        for name in CHECKPOINT_NAMES
        for checkpoint in checkpoints
        if checkpoint.name == name
    )
    return CheckpointSelection(ordered, tuple(requests))


def select_checkpoints(
    model: RotatingBlocksModel,
    completed: RotatingBlocksSolverRun,
) -> tuple[Checkpoint, ...]:
    """Return the available requested checkpoints in deterministic order."""

    return select_checkpoint_regimes(model, completed).checkpoints


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
                "selection_rule": checkpoint.selection_rule,
                "target_parameter": checkpoint.target,
                "accepted_step": checkpoint.accepted_step,
                "selected_parameter": checkpoint.parameter,
                "selection_error": checkpoint.selection_error,
                "maximum_displacement": float(
                    np.max(np.linalg.norm(displacement, axis=1), initial=0.0)
                ),
                "maximum_pressure": float(
                    np.max(evaluation.pressure, initial=0.0)
                ),
                "overlap_area": float(evaluation.raw.contact.weights.total_area),
            }
        )
    return tuple(rows)


def checkpoint_selection_rows(
    selection: CheckpointSelection,
) -> tuple[dict[str, object], ...]:
    """Return complete checkpoint evidence including unavailable regimes."""

    physical = {row["checkpoint"]: row for row in checkpoint_rows(selection.checkpoints)}
    rows: list[dict[str, object]] = []
    for request in selection.requests:
        checkpoint = str(request["checkpoint"])
        selected = physical.get(checkpoint)
        rows.append(
            {
                "checkpoint": checkpoint,
                "selection_rule": str(request["selection_rule"]),
                "target_parameter": float(request["target_parameter"]),
                "present": bool(request["present"]),
                "missing_reason": str(request["missing_reason"]),
                "accepted_step": None if selected is None else selected["accepted_step"],
                "selected_parameter": (
                    None if selected is None else selected["selected_parameter"]
                ),
                "selection_error": (
                    None if selected is None else selected["selection_error"]
                ),
                "maximum_displacement": (
                    None if selected is None else selected["maximum_displacement"]
                ),
                "maximum_pressure": (
                    None if selected is None else selected["maximum_pressure"]
                ),
                "overlap_area": None if selected is None else selected["overlap_area"],
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
