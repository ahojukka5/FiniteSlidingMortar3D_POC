"""Pressure redistribution evidence for the rotating-blocks benchmark."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from rotating_blocks_model import RotatingBlocksModel
from rotating_blocks_refinement import RotatingBlocksRefinement
from rotating_blocks_solver import RotatingBlocksSolverRun

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.benchmark_plots import write_line_chart

SCHEMA = "contact3d-rotating-blocks-pressure/v1"
NODE_SCHEMA = "contact3d-rotating-blocks-pressure-nodes/v1"
AGGREGATE_SCHEMA = "contact3d-rotating-blocks-pressure-aggregates/v1"
REFINEMENT_NODE_SCHEMA = "contact3d-rotating-blocks-pressure-refinement-nodes/v1"
REFINEMENT_AGGREGATE_SCHEMA = (
    "contact3d-rotating-blocks-pressure-refinement-aggregates/v1"
)
AGGREGATE_FIELDS = (
    "pressure_resultant",
    "pressure_mean",
    "pressure_rms",
    "pressure_l2_area",
    "pressure_variance",
    "supported_area",
)
CENTROID_FIELDS = (
    "pressure_centroid_x",
    "pressure_centroid_y",
    "pressure_centroid_z",
)
NODAL_FIELDS = ("pressure", "multiplier", "normal_gap", "row_area")


@dataclass(frozen=True, slots=True)
class PressureHistory:
    """Accepted-state nodal and aggregate pressure evidence."""

    nodal_rows: tuple[dict[str, object], ...]
    aggregate_rows: tuple[dict[str, object], ...]
    summary: dict[str, object]

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


@dataclass(frozen=True, slots=True)
class PressureRefinement:
    """Medium/fine pressure comparisons on one common path grid."""

    nodal_rows: tuple[dict[str, object], ...]
    aggregate_rows: tuple[dict[str, object], ...]
    summary: dict[str, object]

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


@dataclass(frozen=True, slots=True)
class PressureArtifacts:
    """Files, row counts, and assessments added to the result bundle."""

    required: tuple[str, ...]
    row_counts: dict[str, int]
    summary: dict[str, object]


def _surface_normal(
    current_nodes: np.ndarray,
    facets: Iterable[np.ndarray],
    normal_sign: float,
) -> np.ndarray:
    total = np.zeros(3)
    for facet in facets:
        indices = np.asarray(facet, dtype=np.int64)
        points = current_nodes[indices]
        for index in range(1, len(points) - 1):
            total += np.cross(points[index] - points[0], points[index + 1] - points[0])
    magnitude = float(np.linalg.norm(total))
    if magnitude <= np.finfo(float).eps:
        raise ValueError("rotating-blocks slave surface has zero current area")
    return float(normal_sign) * total / magnitude


def _characteristic_length(model: RotatingBlocksModel) -> float:
    points = model.problem.mesh.reference_nodes[model.slave_nodes]
    extent = np.max(points, axis=0) - np.min(points, axis=0)
    value = float(np.linalg.norm(extent))
    if value <= np.finfo(float).eps:
        raise ValueError("rotating-blocks slave surface has zero characteristic length")
    return value


def _phase_values(step: object) -> tuple[int, float, float]:
    value = step.path_state.value
    return (
        int(round(float(value("phase_index")))),
        float(value("phase_parameter")),
        float(value("rotation_angle")),
    )


def _pressure_state(
    model: RotatingBlocksModel,
    step: object,
    accepted_step: int,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    result = step.result
    evaluation = result.equilibrium.evaluation
    if len(evaluation.contacts) != 1 or len(result.states) != 1:
        raise ValueError("rotating-blocks pressure audit requires one contact interface")
    contact = evaluation.contacts[0]
    interface = step.path_state.problem.interfaces[0]
    pressure = np.asarray(contact.pressure, dtype=float)
    gaps = np.asarray(contact.normal_gaps, dtype=float)
    multipliers = np.asarray(result.states[0].multipliers, dtype=float)
    areas = np.asarray(contact.raw.contact.weights.row_areas, dtype=float)
    supported = np.asarray(contact.signature.supported_rows, dtype=bool)
    active = np.asarray(contact.signature.active_rows, dtype=bool)
    node_count = len(interface.slave_nodes)
    for name, values in (
        ("pressure", pressure),
        ("normal gaps", gaps),
        ("multipliers", multipliers),
        ("row areas", areas),
        ("support flags", supported),
        ("activity flags", active),
    ):
        if values.shape != (node_count,):
            raise ValueError(f"{name} must match the slave-node count")
    if np.any(areas < 0.0) or np.any(pressure < -1.0e-14):
        raise ValueError("pressure and row areas must be nonnegative")

    displacement = np.asarray(result.displacement, dtype=float).reshape((-1, 3))
    reference = step.path_state.problem.mesh.reference_nodes
    current = reference + displacement
    slave_nodes = np.asarray(interface.slave_nodes, dtype=np.int64)
    slave_coordinates = current[slave_nodes]
    normal = _surface_normal(
        slave_coordinates,
        interface.pair.slave.facets,
        interface.pair.slave.normal_sign,
    )
    local_force = np.asarray(contact.residual, dtype=float).reshape((-1, 3))
    expected = len(interface.slave_nodes) + len(interface.master_nodes)
    if local_force.shape != (expected, 3):
        raise ValueError("contact residual does not match interface-local node ordering")
    slave_force = np.sum(local_force[:node_count], axis=0)
    master_force = np.sum(local_force[node_count:], axis=0)

    pressure_measure = pressure * areas
    resultant = float(np.sum(pressure_measure))
    support_area = float(np.sum(areas))
    area_scale = max(support_area, np.finfo(float).tiny)
    pressure_mean = resultant / area_scale
    pressure_l2 = float(np.sqrt(np.dot(areas, pressure * pressure)))
    pressure_rms = pressure_l2 / np.sqrt(area_scale)
    pressure_variance = float(
        np.dot(areas, (pressure - pressure_mean) ** 2) / area_scale
    )
    centroid_defined = resultant > 1.0e-14
    centroid = (
        np.sum(slave_coordinates * pressure_measure[:, None], axis=0) / resultant
        if centroid_defined
        else np.zeros(3)
    )

    normal_component = float(np.dot(slave_force, normal))
    normal_resultant = abs(normal_component)
    force_scale = max(resultant, float(np.linalg.norm(slave_force)), 1.0e-14)
    resultant_error = abs(normal_resultant - resultant) / force_scale
    tangential = slave_force - normal_component * normal
    tangential_error = float(np.linalg.norm(tangential)) / force_scale
    pair_balance_error = float(np.linalg.norm(slave_force + master_force)) / max(
        float(np.linalg.norm(slave_force)),
        float(np.linalg.norm(master_force)),
        1.0e-14,
    )
    pressure_scale = max(float(np.max(pressure, initial=0.0)), 1.0)
    multiplier_scale = max(float(np.max(multipliers, initial=0.0)), 1.0)
    unsupported_pressure = float(np.max(pressure[~supported], initial=0.0))
    unsupported_multiplier = float(np.max(multipliers[~supported], initial=0.0))
    phase_index, phase_parameter, rotation_angle = _phase_values(step)

    nodal_rows = tuple(
        {
            "accepted_step": accepted_step,
            "parameter": float(step.parameter),
            "phase_index": phase_index,
            "phase_parameter": phase_parameter,
            "rotation_angle": rotation_angle,
            "node": index,
            "global_node": int(slave_nodes[index]),
            "current_x": float(slave_coordinates[index, 0]),
            "current_y": float(slave_coordinates[index, 1]),
            "current_z": float(slave_coordinates[index, 2]),
            "row_area": float(areas[index]),
            "normal_gap": float(gaps[index]),
            "pressure": float(pressure[index]),
            "multiplier": float(multipliers[index]),
            "supported": bool(supported[index]),
            "active": bool(active[index]),
            "pressure_measure": float(pressure_measure[index]),
        }
        for index in range(node_count)
    )
    aggregate = {
        "accepted_step": accepted_step,
        "parameter": float(step.parameter),
        "phase_index": phase_index,
        "phase_parameter": phase_parameter,
        "rotation_angle": rotation_angle,
        "supported_rows": int(np.count_nonzero(supported)),
        "active_rows": int(np.count_nonzero(active)),
        "supported_area": support_area,
        "pressure_resultant": resultant,
        "contact_normal_resultant": normal_resultant,
        "resultant_relative_error": resultant_error,
        "tangential_relative_force": tangential_error,
        "slave_master_relative_error": pair_balance_error,
        "pressure_mean": pressure_mean,
        "pressure_variance": pressure_variance,
        "pressure_rms": pressure_rms,
        "pressure_l2_area": pressure_l2,
        "pressure_centroid_defined": centroid_defined,
        "pressure_centroid_x": float(centroid[0]) if centroid_defined else None,
        "pressure_centroid_y": float(centroid[1]) if centroid_defined else None,
        "pressure_centroid_z": float(centroid[2]) if centroid_defined else None,
        "maximum_unsupported_pressure": unsupported_pressure,
        "maximum_unsupported_multiplier": unsupported_multiplier,
        "normalized_unsupported_pressure": unsupported_pressure / pressure_scale,
        "normalized_unsupported_multiplier": unsupported_multiplier / multiplier_scale,
        "centroid_interval_crosses_event": False,
        "centroid_jump": None,
        "normalized_centroid_jump": None,
    }
    return nodal_rows, aggregate


def _event_parameters(completed: RotatingBlocksSolverRun) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                float(row["continuation_parameter"])
                for row in completed.event_rows
                if row.get("continuation_parameter") is not None
            }
        )
    )


def _crosses_event(start: float, stop: float, events: tuple[float, ...]) -> bool:
    tolerance = 1.0e-12
    return any(start - tolerance < value <= stop + tolerance for value in events)


def collect_pressure_history(
    model: RotatingBlocksModel,
    completed: RotatingBlocksSolverRun,
) -> PressureHistory:
    """Collect nodal and aggregate pressure evidence at every accepted state."""

    steps = tuple(getattr(completed.result, "accepted_steps", ()))
    if not steps:
        raise ValueError("rotating-blocks pressure audit requires accepted states")
    nodal_rows: list[dict[str, object]] = []
    aggregates: list[dict[str, object]] = []
    for index, step in enumerate(steps, start=1):
        nodal, aggregate = _pressure_state(model, step, index)
        nodal_rows.extend(nodal)
        aggregates.append(aggregate)

    length = _characteristic_length(model)
    events = _event_parameters(completed)
    previous: dict[str, object] | None = None
    for row in aggregates:
        if not bool(row["pressure_centroid_defined"]):
            previous = None
            continue
        if previous is not None:
            start = float(previous["parameter"])
            stop = float(row["parameter"])
            crosses = _crosses_event(start, stop, events)
            row["centroid_interval_crosses_event"] = crosses
            if not crosses:
                before = np.asarray(
                    [
                        previous["pressure_centroid_x"],
                        previous["pressure_centroid_y"],
                        previous["pressure_centroid_z"],
                    ],
                    dtype=float,
                )
                after = np.asarray(
                    [
                        row["pressure_centroid_x"],
                        row["pressure_centroid_y"],
                        row["pressure_centroid_z"],
                    ],
                    dtype=float,
                )
                jump = float(np.linalg.norm(after - before))
                row["centroid_jump"] = jump
                row["normalized_centroid_jump"] = jump / length
        previous = row

    maximum_resultant_error = max(
        float(row["resultant_relative_error"]) for row in aggregates
    )
    maximum_tangential_error = max(
        float(row["tangential_relative_force"]) for row in aggregates
    )
    maximum_pair_error = max(
        float(row["slave_master_relative_error"]) for row in aggregates
    )
    maximum_unsupported_pressure = max(
        float(row["normalized_unsupported_pressure"]) for row in aggregates
    )
    maximum_unsupported_multiplier = max(
        float(row["normalized_unsupported_multiplier"]) for row in aggregates
    )
    maximum_centroid_jump = max(
        (
            float(row["normalized_centroid_jump"])
            for row in aggregates
            if row["normalized_centroid_jump"] is not None
        ),
        default=0.0,
    )
    expected_nodal_rows = sum(
        len(step.path_state.problem.interfaces[0].slave_nodes) for step in steps
    )
    criteria = {
        "all_accepted_states_recorded": len(aggregates) == len(steps),
        "all_nodal_states_recorded": len(nodal_rows) == expected_nodal_rows,
        "pressure_resultant_matches_contact_force": maximum_resultant_error <= 1.0e-8,
        "frictionless_tangential_force_satisfied": maximum_tangential_error <= 1.0e-8,
        "slave_master_contact_balance_satisfied": maximum_pair_error <= 1.0e-8,
        "unsupported_pressure_zero": maximum_unsupported_pressure <= 1.0e-12,
        "unsupported_multiplier_zero": maximum_unsupported_multiplier <= 1.0e-12,
        "pressure_centroid_defined_in_contact": any(
            bool(row["pressure_centroid_defined"]) for row in aggregates
        ),
        "centroid_continuous_between_events": maximum_centroid_jump <= 0.5,
    }
    summary = {
        "schema_version": SCHEMA,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "accepted_state_count": len(aggregates),
        "nodal_row_count": len(nodal_rows),
        "maximum_resultant_relative_error": maximum_resultant_error,
        "maximum_tangential_relative_force": maximum_tangential_error,
        "maximum_slave_master_relative_error": maximum_pair_error,
        "maximum_normalized_unsupported_pressure": maximum_unsupported_pressure,
        "maximum_normalized_unsupported_multiplier": maximum_unsupported_multiplier,
        "maximum_non_event_centroid_jump": maximum_centroid_jump,
        "centroid_jump_limit": 0.5,
        "characteristic_length": length,
    }
    return PressureHistory(tuple(nodal_rows), tuple(aggregates), summary)


def _interpolate(
    rows: tuple[dict[str, object], ...],
    parameters: np.ndarray,
    field: str,
    *,
    node: int | None = None,
    defined_field: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = tuple(
        row
        for row in rows
        if (node is None or int(row["node"]) == node)
        and (defined_field is None or bool(row[defined_field]))
    )
    if not selected:
        return np.zeros(len(parameters)), np.zeros(len(parameters), dtype=bool)
    source_parameters = np.asarray([float(row["parameter"]) for row in selected])
    source_values = np.asarray([float(row[field]) for row in selected])
    order = np.argsort(source_parameters)
    source_parameters = source_parameters[order]
    source_values = source_values[order]
    unique, indices = np.unique(source_parameters, return_index=True)
    values = np.interp(parameters, unique, source_values[indices])
    defined = (parameters >= unique[0] - 1.0e-12) & (
        parameters <= unique[-1] + 1.0e-12
    )
    return values, defined


def _nearest_flag(
    rows: tuple[dict[str, object], ...],
    parameters: np.ndarray,
    node: int,
    field: str,
) -> np.ndarray:
    selected = tuple(row for row in rows if int(row["node"]) == node)
    source_parameters = np.asarray([float(row["parameter"]) for row in selected])
    source_values = np.asarray([bool(row[field]) for row in selected])
    order = np.argsort(source_parameters)
    source_parameters = source_parameters[order]
    source_values = source_values[order]
    result = []
    for parameter in parameters:
        distance = np.abs(source_parameters - parameter)
        result.append(bool(source_values[int(np.argmin(distance))]))
    return np.asarray(result, dtype=bool)


def _relative_scale(values: np.ndarray, floor: float = 1.0e-14) -> float:
    return max(float(np.max(np.abs(values), initial=0.0)), floor)


def _aggregate_comparison(
    medium: PressureHistory,
    fine: PressureHistory,
    parameters: np.ndarray,
    length: float,
) -> tuple[dict[str, object], ...]:
    fields = (*AGGREGATE_FIELDS, *CENTROID_FIELDS)
    interpolated: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for field in fields:
        defined_field = "pressure_centroid_defined" if field in CENTROID_FIELDS else None
        medium_values, medium_defined = _interpolate(
            medium.aggregate_rows,
            parameters,
            field,
            defined_field=defined_field,
        )
        fine_values, fine_defined = _interpolate(
            fine.aggregate_rows,
            parameters,
            field,
            defined_field=defined_field,
        )
        interpolated[field] = (
            medium_values,
            fine_values,
            medium_defined & fine_defined,
        )
    rows = []
    for index, parameter in enumerate(parameters):
        row: dict[str, object] = {"parameter": float(parameter)}
        for field, (medium_values, fine_values, defined) in interpolated.items():
            available = bool(defined[index])
            medium_value = float(medium_values[index]) if available else None
            fine_value = float(fine_values[index]) if available else None
            error = (
                abs(float(medium_values[index] - fine_values[index]))
                if available
                else None
            )
            scale = (
                length
                if field in CENTROID_FIELDS
                else _relative_scale(fine_values[defined])
            )
            row[f"medium_{field}"] = medium_value
            row[f"fine_{field}"] = fine_value
            row[f"absolute_error_{field}"] = error
            row[f"relative_error_{field}"] = error / scale if available else None
        rows.append(row)
    return tuple(rows)


def _nodal_comparison(
    medium: PressureHistory,
    fine: PressureHistory,
    parameters: np.ndarray,
) -> tuple[dict[str, object], ...]:
    nodes = sorted({int(row["node"]) for row in fine.nodal_rows})
    fields: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, float]] = {}
    flags: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for node in nodes:
        for field in NODAL_FIELDS:
            medium_values, _ = _interpolate(
                medium.nodal_rows,
                parameters,
                field,
                node=node,
            )
            fine_values, _ = _interpolate(
                fine.nodal_rows,
                parameters,
                field,
                node=node,
            )
            fields[(node, field)] = (
                medium_values,
                fine_values,
                _relative_scale(fine_values),
            )
        for field in ("supported", "active"):
            flags[(node, field)] = (
                _nearest_flag(medium.nodal_rows, parameters, node, field),
                _nearest_flag(fine.nodal_rows, parameters, node, field),
            )
    rows = []
    for index, parameter in enumerate(parameters):
        for node in nodes:
            row: dict[str, object] = {
                "parameter": float(parameter),
                "node": node,
            }
            for field in NODAL_FIELDS:
                medium_values, fine_values, scale = fields[(node, field)]
                error = abs(float(medium_values[index] - fine_values[index]))
                row[f"medium_{field}"] = float(medium_values[index])
                row[f"fine_{field}"] = float(fine_values[index])
                row[f"absolute_error_{field}"] = error
                row[f"relative_error_{field}"] = error / scale
            for field in ("supported", "active"):
                medium_values, fine_values = flags[(node, field)]
                row[f"medium_{field}"] = bool(medium_values[index])
                row[f"fine_{field}"] = bool(fine_values[index])
                row[f"{field}_matches"] = bool(
                    medium_values[index] == fine_values[index]
                )
            rows.append(row)
    return tuple(rows)


def compare_pressure_refinement(
    model: RotatingBlocksModel,
    refinement: RotatingBlocksRefinement,
) -> PressureRefinement:
    """Compare medium and fine nodal and aggregate pressure histories."""

    levels = tuple(refinement.levels)
    if len(levels) < 2:
        raise ValueError("pressure refinement requires at least two path levels")
    medium = collect_pressure_history(model, levels[-2].run)
    fine = collect_pressure_history(model, levels[-1].run)
    parameters = np.asarray(refinement.comparison_parameters, dtype=float)
    if parameters.ndim != 1 or len(parameters) == 0:
        raise ValueError("pressure refinement requires a common parameter grid")
    length = _characteristic_length(model)
    aggregate_rows = _aggregate_comparison(medium, fine, parameters, length)
    nodal_rows = _nodal_comparison(medium, fine, parameters)

    aggregate_errors = {
        field: max(
            (
                float(row[f"relative_error_{field}"])
                for row in aggregate_rows
                if row[f"relative_error_{field}"] is not None
            ),
            default=0.0,
        )
        for field in (*AGGREGATE_FIELDS, *CENTROID_FIELDS)
    }
    nodal_errors = {
        field: max(float(row[f"relative_error_{field}"]) for row in nodal_rows)
        for field in NODAL_FIELDS
    }
    mismatch_rows = tuple(
        row
        for row in nodal_rows
        if not bool(row["supported_matches"]) or not bool(row["active_matches"])
    )
    event_parameters = tuple(
        sorted(
            float(row["fine_parameter"])
            for row in refinement.event_rows
            if row.get("fine_parameter") is not None
        )
    )
    event_tolerance = 2.0 / levels[-1].requested_steps
    mismatches_localized = all(
        event_parameters
        and min(abs(float(row["parameter"]) - value) for value in event_parameters)
        <= event_tolerance + 1.0e-12
        for row in mismatch_rows
    )
    pressure_aggregate_fields = (
        "pressure_resultant",
        "pressure_mean",
        "pressure_rms",
        "pressure_l2_area",
        "pressure_variance",
    )
    criteria = {
        "medium_fine_pressure_histories_passed": medium.passed and fine.passed,
        "aggregate_pressure_history_converged": max(
            aggregate_errors[field] for field in pressure_aggregate_fields
        )
        <= 5.0e-2,
        "pressure_centroid_history_converged": max(
            aggregate_errors[field] for field in CENTROID_FIELDS
        )
        <= 5.0e-2,
        "nodal_pressure_history_converged": nodal_errors["pressure"] <= 1.0e-1,
        "nodal_multiplier_history_converged": nodal_errors["multiplier"] <= 1.0e-1,
        "nodal_gap_history_converged": nodal_errors["normal_gap"] <= 1.0e-1,
        "nodal_support_area_history_converged": nodal_errors["row_area"] <= 5.0e-2,
        "discrete_state_mismatches_localized": mismatches_localized,
    }
    summary = {
        "schema_version": SCHEMA,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "comparison_parameter_count": len(parameters),
        "aggregate_row_count": len(aggregate_rows),
        "nodal_row_count": len(nodal_rows),
        "maximum_relative_aggregate_errors": aggregate_errors,
        "maximum_relative_nodal_errors": nodal_errors,
        "discrete_state_mismatch_count": len(mismatch_rows),
        "event_localization_tolerance": event_tolerance,
    }
    return PressureRefinement(nodal_rows, aggregate_rows, summary)


def _write_pressure_plots(
    writer: BenchmarkArtifactWriter,
    output: Path,
    history: PressureHistory,
    refinement: PressureRefinement,
) -> tuple[str, ...]:
    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        "plots/pressure-nodal-history.svg",
        "plots/pressure-aggregate-history.svg",
        "plots/pressure-centroid-history.svg",
        "plots/pressure-refinement-errors.svg",
        "plots/pressure-nodal-refinement-errors.svg",
    )
    aggregate_parameters = np.asarray(
        [float(row["parameter"]) for row in history.aggregate_rows]
    )
    nodes = sorted({int(row["node"]) for row in history.nodal_rows})
    write_line_chart(
        output / paths[0],
        title="Rotating-blocks nodal pressure histories",
        x_label="continuation parameter",
        y_label="pressure",
        x_values=aggregate_parameters,
        series=tuple(
            (
                np.asarray(
                    [
                        float(row["pressure"])
                        for row in history.nodal_rows
                        if int(row["node"]) == node
                    ]
                ),
                f"node {node}",
            )
            for node in nodes
        ),
        show_markers=True,
    )
    write_line_chart(
        output / paths[1],
        title="Rotating-blocks aggregate pressure measures",
        x_label="continuation parameter",
        y_label="pressure measure",
        x_values=aggregate_parameters,
        series=tuple(
            (
                np.asarray([float(row[field]) for row in history.aggregate_rows]),
                field,
            )
            for field in (
                "pressure_resultant",
                "pressure_mean",
                "pressure_rms",
                "pressure_l2_area",
            )
        ),
        show_markers=True,
    )
    centroid_rows = tuple(
        row for row in history.aggregate_rows if row["pressure_centroid_defined"]
    )
    write_line_chart(
        output / paths[2],
        title="Rotating-blocks pressure centroid",
        x_label="continuation parameter",
        y_label="centroid coordinate",
        x_values=np.asarray([float(row["parameter"]) for row in centroid_rows]),
        series=tuple(
            (
                np.asarray([float(row[field]) for row in centroid_rows]),
                field,
            )
            for field in CENTROID_FIELDS
        ),
        show_markers=True,
    )
    refinement_parameters = np.asarray(
        [float(row["parameter"]) for row in refinement.aggregate_rows]
    )
    write_line_chart(
        output / paths[3],
        title="Rotating-blocks aggregate pressure refinement",
        x_label="continuation parameter",
        y_label="relative error",
        x_values=refinement_parameters,
        series=tuple(
            (
                np.asarray(
                    [
                        0.0
                        if row[f"relative_error_{field}"] is None
                        else float(row[f"relative_error_{field}"])
                        for row in refinement.aggregate_rows
                    ]
                ),
                field,
            )
            for field in (
                "pressure_resultant",
                "pressure_rms",
                "pressure_l2_area",
                "pressure_centroid_x",
                "pressure_centroid_y",
            )
        ),
        show_markers=True,
    )
    nodal_by_parameter: dict[float, list[dict[str, object]]] = {}
    for row in refinement.nodal_rows:
        nodal_by_parameter.setdefault(float(row["parameter"]), []).append(row)
    nodal_parameters = np.asarray(sorted(nodal_by_parameter))
    write_line_chart(
        output / paths[4],
        title="Rotating-blocks nodal pressure refinement",
        x_label="continuation parameter",
        y_label="maximum relative error",
        x_values=nodal_parameters,
        series=tuple(
            (
                np.asarray(
                    [
                        max(
                            float(row[f"relative_error_{field}"])
                            for row in nodal_by_parameter[parameter]
                        )
                        for parameter in nodal_parameters
                    ]
                ),
                field,
            )
            for field in NODAL_FIELDS
        ),
        show_markers=True,
    )
    for relative in paths:
        ElementTree.parse(output / relative)
        writer.register(relative, "svg")
    return paths


def write_pressure_artifacts(
    writer: BenchmarkArtifactWriter,
    output: Path,
    model: RotatingBlocksModel,
    completed: RotatingBlocksSolverRun,
    refinement: RotatingBlocksRefinement,
) -> PressureArtifacts:
    """Write pressure histories, refinement evidence, plots, and summary."""

    history = collect_pressure_history(model, completed)
    comparison = compare_pressure_refinement(model, refinement)
    paths = (
        "tables/pressure-nodes.csv",
        "tables/pressure-aggregates.csv",
        "tables/refinement-pressure-nodes.csv",
        "tables/refinement-pressure-aggregates.csv",
        "pressure-summary.json",
    )
    writer.write_csv(paths[0], history.nodal_rows, schema=NODE_SCHEMA)
    writer.write_csv(paths[1], history.aggregate_rows, schema=AGGREGATE_SCHEMA)
    writer.write_csv(
        paths[2],
        comparison.nodal_rows,
        schema=REFINEMENT_NODE_SCHEMA,
    )
    writer.write_csv(
        paths[3],
        comparison.aggregate_rows,
        schema=REFINEMENT_AGGREGATE_SCHEMA,
    )
    summary = {
        "schema_version": SCHEMA,
        "passed": history.passed and comparison.passed,
        "history": history.summary,
        "refinement": comparison.summary,
    }
    writer.write_json(paths[4], summary, schema=SCHEMA)
    plots = _write_pressure_plots(writer, Path(output), history, comparison)
    row_counts = {
        "pressure_nodes": len(history.nodal_rows),
        "pressure_aggregates": len(history.aggregate_rows),
        "refinement_pressure_nodes": len(comparison.nodal_rows),
        "refinement_pressure_aggregates": len(comparison.aggregate_rows),
    }
    return PressureArtifacts((*paths, *plots), row_counts, summary)
