"""Force and moment balance diagnostics for the rotating-blocks benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rotating_blocks_model import RotatingBlocksModel
from rotating_blocks_solver import RotatingBlocksSolverRun

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.benchmark_plots import write_category_timeline, write_line_chart

SCHEMA = "contact3d-rotating-blocks-balance/v1"
FORCE_FIELDS = (
    "normalized_global_force_error",
    "normalized_contact_force_error",
)
MOMENT_FIELDS = (
    "normalized_global_moment_origin_error",
    "normalized_global_moment_pivot_error",
    "normalized_contact_moment_origin_error",
    "normalized_contact_moment_pivot_error",
)


@dataclass(frozen=True, slots=True)
class RotatingBlocksBalance:
    """Complete accepted-state balance rows and their aggregate assessment."""

    rows: tuple[dict[str, object], ...]
    summary: dict[str, object]

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


def _points(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"{name} must have shape (node_count, 3)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _nodal_vectors(value: object, node_count: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape == (3 * node_count,):
        array = array.reshape((-1, 3))
    if array.shape != (node_count, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must match the global node count")
    return array


def _indices(value: object, node_count: int, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional index vector")
    if np.any(array < 0) or np.any(array >= node_count):
        raise ValueError(f"{name} contains an out-of-range node index")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} must contain unique node indices")
    return array


def _moment(
    coordinates: np.ndarray,
    forces: np.ndarray,
    point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    nodal = np.cross(coordinates - point, forces)
    return nodal, np.sum(nodal, axis=0)


def _variation(values: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(values, axis=1)))


def _components(row: dict[str, object], name: str, value: np.ndarray) -> None:
    row[f"{name}_x"] = float(value[0])
    row[f"{name}_y"] = float(value[1])
    row[f"{name}_z"] = float(value[2])


def evaluate_balance(
    current_coordinates: object,
    reaction: object,
    applied: object,
    local_contact_force: object,
    slave_nodes: object,
    master_nodes: object,
    pivot: object,
    *,
    accepted_step: int,
    parameter: float,
) -> dict[str, object]:
    """Evaluate one accepted state's force and moment balance."""

    coordinates = _points(current_coordinates, name="current_coordinates")
    node_count = len(coordinates)
    reaction_values = _nodal_vectors(reaction, node_count, name="reaction")
    applied_values = _nodal_vectors(applied, node_count, name="applied")
    slave = _indices(slave_nodes, node_count, name="slave_nodes")
    master = _indices(master_nodes, node_count, name="master_nodes")
    if np.intersect1d(slave, master).size:
        raise ValueError("slave and master node mappings must be disjoint")
    local = np.asarray(local_contact_force, dtype=float)
    if local.shape == (3 * (len(slave) + len(master)),):
        local = local.reshape((-1, 3))
    expected = (len(slave) + len(master), 3)
    if local.shape != expected or not np.all(np.isfinite(local)):
        raise ValueError("local_contact_force must match slave and master nodes")
    pivot_value = np.asarray(pivot, dtype=float)
    if pivot_value.shape != (3,) or not np.all(np.isfinite(pivot_value)):
        raise ValueError("pivot must be a finite three-vector")
    parameter_value = float(parameter)
    if not np.isfinite(parameter_value):
        raise ValueError("parameter must be finite")
    if accepted_step <= 0:
        raise ValueError("accepted_step must be positive")

    origin = np.zeros(3)
    slave_force = local[: len(slave)]
    master_force = local[len(slave) :]
    applied_resultant = np.sum(applied_values, axis=0)
    reaction_resultant = np.sum(reaction_values, axis=0)
    slave_resultant = np.sum(slave_force, axis=0)
    master_resultant = np.sum(master_force, axis=0)
    global_force_error = applied_resultant + reaction_resultant
    contact_force_error = slave_resultant + master_resultant

    applied_origin_nodal, applied_origin = _moment(
        coordinates,
        applied_values,
        origin,
    )
    reaction_origin_nodal, reaction_origin = _moment(
        coordinates,
        reaction_values,
        origin,
    )
    applied_pivot_nodal, applied_pivot = _moment(
        coordinates,
        applied_values,
        pivot_value,
    )
    reaction_pivot_nodal, reaction_pivot = _moment(
        coordinates,
        reaction_values,
        pivot_value,
    )
    slave_origin_nodal, slave_origin = _moment(
        coordinates[slave],
        slave_force,
        origin,
    )
    master_origin_nodal, master_origin = _moment(
        coordinates[master],
        master_force,
        origin,
    )
    slave_pivot_nodal, slave_pivot = _moment(
        coordinates[slave],
        slave_force,
        pivot_value,
    )
    master_pivot_nodal, master_pivot = _moment(
        coordinates[master],
        master_force,
        pivot_value,
    )
    global_origin_error = applied_origin + reaction_origin
    global_pivot_error = applied_pivot + reaction_pivot
    contact_origin_error = slave_origin + master_origin
    contact_pivot_error = slave_pivot + master_pivot

    tiny = np.finfo(float).tiny
    force_scale = max(
        _variation(applied_values),
        _variation(reaction_values),
        _variation(local),
        tiny,
    )
    contact_force_scale = max(
        _variation(slave_force),
        _variation(master_force),
        tiny,
    )
    length_scale = max(
        float(np.max(np.linalg.norm(coordinates, axis=1), initial=0.0)),
        float(
            np.max(
                np.linalg.norm(coordinates - pivot_value, axis=1),
                initial=0.0,
            )
        ),
        tiny,
    )
    moment_scale = max(
        _variation(applied_origin_nodal),
        _variation(reaction_origin_nodal),
        _variation(applied_pivot_nodal),
        _variation(reaction_pivot_nodal),
        force_scale * length_scale,
        tiny,
    )
    contact_moment_scale = max(
        _variation(slave_origin_nodal),
        _variation(master_origin_nodal),
        _variation(slave_pivot_nodal),
        _variation(master_pivot_nodal),
        contact_force_scale * length_scale,
        tiny,
    )

    row: dict[str, object] = {
        "accepted_step": int(accepted_step),
        "parameter": parameter_value,
        "force_scale": force_scale,
        "moment_scale": moment_scale,
        "contact_force_scale": contact_force_scale,
        "contact_moment_scale": contact_moment_scale,
        "normalized_global_force_error": float(
            np.linalg.norm(global_force_error) / force_scale
        ),
        "normalized_contact_force_error": float(
            np.linalg.norm(contact_force_error) / contact_force_scale
        ),
        "normalized_global_moment_origin_error": float(
            np.linalg.norm(global_origin_error) / moment_scale
        ),
        "normalized_global_moment_pivot_error": float(
            np.linalg.norm(global_pivot_error) / moment_scale
        ),
        "normalized_contact_moment_origin_error": float(
            np.linalg.norm(contact_origin_error) / contact_moment_scale
        ),
        "normalized_contact_moment_pivot_error": float(
            np.linalg.norm(contact_pivot_error) / contact_moment_scale
        ),
    }
    _components(row, "pivot", pivot_value)
    _components(row, "applied_force", applied_resultant)
    _components(row, "reaction_force", reaction_resultant)
    _components(row, "global_force_error", global_force_error)
    _components(row, "slave_contact_force", slave_resultant)
    _components(row, "master_contact_force", master_resultant)
    _components(row, "contact_force_error", contact_force_error)
    _components(row, "applied_moment_origin", applied_origin)
    _components(row, "reaction_moment_origin", reaction_origin)
    _components(row, "global_moment_origin_error", global_origin_error)
    _components(row, "applied_moment_pivot", applied_pivot)
    _components(row, "reaction_moment_pivot", reaction_pivot)
    _components(row, "global_moment_pivot_error", global_pivot_error)
    _components(row, "slave_contact_moment_origin", slave_origin)
    _components(row, "master_contact_moment_origin", master_origin)
    _components(row, "contact_moment_origin_error", contact_origin_error)
    _components(row, "slave_contact_moment_pivot", slave_pivot)
    _components(row, "master_contact_moment_pivot", master_pivot)
    _components(row, "contact_moment_pivot_error", contact_pivot_error)
    return row


def _worst(
    rows: tuple[dict[str, object], ...],
    fields: tuple[str, ...],
) -> dict[str, object]:
    candidates = tuple(
        (float(row[field]), int(row["accepted_step"]), field, row)
        for row in rows
        for field in fields
    )
    value, _, field, row = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return {
        "metric": field,
        "accepted_step": int(row["accepted_step"]),
        "parameter": float(row["parameter"]),
        "value": value,
    }


def summarize_balance(
    rows: tuple[dict[str, object], ...],
    *,
    force_tolerance: float = 1.0e-7,
    moment_tolerance: float = 1.0e-7,
) -> dict[str, object]:
    """Assess complete accepted-state balance histories."""

    if not rows:
        raise ValueError("balance assessment requires at least one accepted state")
    for value in (force_tolerance, moment_tolerance):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("balance tolerances must be finite and nonnegative")
    maxima = {
        field: max(float(row[field]) for row in rows)
        for field in (*FORCE_FIELDS, *MOMENT_FIELDS)
    }
    criteria = {
        field: value
        <= (force_tolerance if field in FORCE_FIELDS else moment_tolerance)
        for field, value in maxima.items()
    }
    return {
        "schema_version": SCHEMA,
        "passed": all(criteria.values()),
        "accepted_state_count": len(rows),
        "force_tolerance": float(force_tolerance),
        "moment_tolerance": float(moment_tolerance),
        "criteria": criteria,
        "maximum_normalized_errors": maxima,
        "worst_force_state": _worst(rows, FORCE_FIELDS),
        "worst_moment_state": _worst(rows, MOMENT_FIELDS),
    }


def _reaction(evaluation: object) -> np.ndarray:
    residual = np.asarray(evaluation.residual, dtype=float).reshape(-1)
    free = np.asarray(evaluation.free_dofs, dtype=np.int64).reshape(-1)
    values = np.zeros_like(residual)
    constrained = np.ones(len(residual), dtype=bool)
    constrained[free] = False
    values[constrained] = residual[constrained]
    return values


def audit_accepted_states(
    model: RotatingBlocksModel,
    completed: RotatingBlocksSolverRun,
    *,
    force_tolerance: float = 1.0e-7,
    moment_tolerance: float = 1.0e-7,
) -> RotatingBlocksBalance:
    """Audit every production accepted state in continuation order."""

    rows: list[dict[str, object]] = []
    steps = tuple(getattr(completed.result, "accepted_steps", ()))
    for accepted_step, step in enumerate(steps, start=1):
        evaluation = step.result.equilibrium.evaluation
        problem = step.path_state.problem
        interface = problem.interfaces[0]
        if len(evaluation.contacts) != 1:
            raise ValueError("rotating-blocks balance requires one contact interface")
        displacement = np.asarray(evaluation.displacement, dtype=float).reshape((-1, 3))
        current = problem.mesh.reference_nodes + displacement
        reaction = getattr(step, "reaction", None)
        reaction_values = (
            _reaction(evaluation)
            if reaction is None
            else np.asarray(reaction, dtype=float).reshape(-1)
        )
        applied = float(evaluation.load_factor) * problem.load.force
        translation = np.asarray(
            [
                step.path_state.value("translation_x"),
                step.path_state.value("translation_y"),
                step.path_state.value("translation_z"),
            ],
            dtype=float,
        )
        pivot = np.asarray(model.geometry.pivot, dtype=float) + translation
        rows.append(
            evaluate_balance(
                current,
                reaction_values,
                applied,
                evaluation.contacts[0].residual,
                interface.slave_nodes,
                interface.master_nodes,
                pivot,
                accepted_step=accepted_step,
                parameter=float(step.parameter),
            )
        )
    values = tuple(rows)
    return RotatingBlocksBalance(
        values,
        summarize_balance(
            values,
            force_tolerance=force_tolerance,
            moment_tolerance=moment_tolerance,
        ),
    )


def write_balance_plots(
    writer: BenchmarkArtifactWriter,
    output: Path,
    balance: RotatingBlocksBalance,
) -> tuple[str, ...]:
    """Write deterministic force, moment, and worst-state visualizations."""

    output = Path(output)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    parameters = np.asarray([float(row["parameter"]) for row in balance.rows])
    force_path = "plots/force-balance.svg"
    write_line_chart(
        output / force_path,
        title="Rotating-blocks normalized force balance",
        x_label="continuation parameter",
        y_label="normalized error",
        x_values=parameters,
        series=tuple(
            (
                np.asarray([float(row[field]) for row in balance.rows]),
                field.removeprefix("normalized_").replace("_", " "),
            )
            for field in FORCE_FIELDS
        ),
        show_markers=True,
    )
    writer.register(force_path, "svg")

    moment_path = "plots/moment-balance.svg"
    write_line_chart(
        output / moment_path,
        title="Rotating-blocks normalized moment balance",
        x_label="continuation parameter",
        y_label="normalized error",
        x_values=parameters,
        series=tuple(
            (
                np.asarray([float(row[field]) for row in balance.rows]),
                field.removeprefix("normalized_").replace("_", " "),
            )
            for field in MOMENT_FIELDS
        ),
        show_markers=True,
    )
    writer.register(moment_path, "svg")

    worst_force = balance.summary["worst_force_state"]
    worst_moment = balance.summary["worst_moment_state"]
    assert isinstance(worst_force, dict) and isinstance(worst_moment, dict)
    worst_path = "plots/balance-worst-states.svg"
    write_category_timeline(
        output / worst_path,
        title="Worst accepted balance states",
        x_label="continuation parameter",
        categories=(
            str(worst_force["metric"]).removeprefix("normalized_"),
            str(worst_moment["metric"]).removeprefix("normalized_"),
        ),
        x_values=np.asarray(
            [float(worst_force["parameter"]), float(worst_moment["parameter"])]
        ),
    )
    writer.register(worst_path, "svg")
    return force_path, moment_path, worst_path
