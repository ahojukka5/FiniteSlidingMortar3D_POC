"""Mesh-quality evidence for the rotating-blocks benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rotating_blocks_model import RotatingBlocksModel, build_rotating_blocks_model
from rotating_blocks_profiles import (
    RotatingBlocksExecutionProfile,
    rotating_blocks_execution_profile,
)

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.benchmark_plots import write_line_chart

SCHEMA = "contact3d-rotating-blocks-mesh-quality/v1"
ROW_SCHEMA = "contact3d-rotating-blocks-mesh-quality-rows/v1"
REFINEMENT_SCHEMA = "contact3d-rotating-blocks-mesh-quality-refinement/v1"


@dataclass(frozen=True, slots=True)
class MeshQualityThresholds:
    """Scale-independent warning and failure limits."""

    profile: str
    warning_minimum_jacobian: float
    failure_minimum_jacobian: float
    warning_normalized_energy_density: float
    failure_normalized_energy_density: float
    maximum_refinement_jacobian_difference: float
    maximum_refinement_energy_difference: float

    def __post_init__(self) -> None:
        if self.profile not in ("quick", "full"):
            raise ValueError("mesh-quality profile must be 'quick' or 'full'")
        values = (
            self.warning_minimum_jacobian,
            self.failure_minimum_jacobian,
            self.warning_normalized_energy_density,
            self.failure_normalized_energy_density,
            self.maximum_refinement_jacobian_difference,
            self.maximum_refinement_energy_difference,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("mesh-quality thresholds must be finite and nonnegative")
        if self.failure_minimum_jacobian >= self.warning_minimum_jacobian:
            raise ValueError("Jacobian failure limit must be below warning limit")
        if (
            self.failure_normalized_energy_density
            <= self.warning_normalized_energy_density
        ):
            raise ValueError("energy failure limit must exceed warning limit")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MeshQualityHistory:
    """Accepted-state element-quality rows and aggregate assessment."""

    rows: tuple[dict[str, object], ...]
    summary: dict[str, object]

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


@dataclass(frozen=True, slots=True)
class MeshQualityRefinement:
    """Medium/fine quality-history comparison on one path grid."""

    rows: tuple[dict[str, object], ...]
    summary: dict[str, object]

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


@dataclass(frozen=True, slots=True)
class RotatingBlocksMeshQuality:
    """Production and refinement mesh-quality evidence."""

    history: MeshQualityHistory
    refinement: MeshQualityRefinement
    summary: dict[str, object]

    @property
    def rows(self) -> tuple[dict[str, object], ...]:
        return self.history.rows

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


def mesh_quality_thresholds(
    profile: str | RotatingBlocksExecutionProfile,
) -> MeshQualityThresholds:
    """Return immutable quick/full quality limits."""

    selected = rotating_blocks_execution_profile(profile)
    return MeshQualityThresholds(
        profile=selected.name,
        warning_minimum_jacobian=0.50,
        failure_minimum_jacobian=0.05,
        warning_normalized_energy_density=0.50,
        failure_normalized_energy_density=5.0,
        maximum_refinement_jacobian_difference=5.0e-2,
        maximum_refinement_energy_difference=1.0e-1,
    )


def _element_location(
    model: RotatingBlocksModel,
    element: int,
) -> tuple[str, int]:
    lower_count = len(model.lower_elements)
    if element < lower_count:
        return "lower", element
    return "upper", element - lower_count


def _quality_row(
    model: RotatingBlocksModel,
    source: Mapping[str, object],
    step: object,
    thresholds: MeshQualityThresholds,
) -> dict[str, object]:
    evaluations = tuple(step.result.equilibrium.evaluation.bulk.element_evaluations)
    if len(evaluations) != model.problem.mesh.element_count:
        raise ValueError("mesh-quality evaluations must match the element count")
    jacobians = np.asarray([item.jacobian for item in evaluations], dtype=float)
    energy = np.asarray([item.energy_density for item in evaluations], dtype=float)
    if not np.all(np.isfinite(jacobians)) or not np.all(np.isfinite(energy)):
        raise ValueError("mesh-quality fields must be finite")
    shear = float(model.problem.material.shear_modulus)
    normalized_energy = energy / shear
    minimum_index = int(np.argmin(jacobians))
    maximum_index = int(np.argmax(normalized_energy))
    minimum_body, minimum_local = _element_location(model, minimum_index)
    maximum_body, maximum_local = _element_location(model, maximum_index)
    minimum_jacobian = float(jacobians[minimum_index])
    maximum_normalized_energy = float(normalized_energy[maximum_index])
    inverted = minimum_jacobian <= 0.0
    failed = (
        minimum_jacobian <= thresholds.failure_minimum_jacobian
        or maximum_normalized_energy
        > thresholds.failure_normalized_energy_density
    )
    warning = (
        not failed
        and (
            minimum_jacobian <= thresholds.warning_minimum_jacobian
            or maximum_normalized_energy
            > thresholds.warning_normalized_energy_density
        )
    )
    status = "failed" if failed else "warning" if warning else "accepted"
    return {
        "accepted_step": int(source["accepted_step"]),
        "parameter": float(source["parameter"]),
        "phase_index": int(source.get("phase_index", -1)),
        "phase_parameter": float(source.get("phase_parameter", 0.0)),
        "rotation_angle": float(source.get("rotation_angle", 0.0)),
        "element_count": len(evaluations),
        "minimum_jacobian": minimum_jacobian,
        "minimum_jacobian_element": minimum_index,
        "minimum_jacobian_body": minimum_body,
        "minimum_jacobian_body_element": minimum_local,
        "minimum_energy_density": float(np.min(energy)),
        "maximum_energy_density": float(energy[maximum_index]),
        "maximum_normalized_energy_density": maximum_normalized_energy,
        "maximum_energy_element": maximum_index,
        "maximum_energy_body": maximum_body,
        "maximum_energy_body_element": maximum_local,
        "inverted": inverted,
        "warning": warning,
        "passed": not failed and not inverted,
        "status": status,
    }


def collect_mesh_quality_rows(
    model: RotatingBlocksModel,
    completed: object,
    *,
    thresholds: MeshQualityThresholds | None = None,
) -> tuple[dict[str, object], ...]:
    """Collect one quality row for every accepted nonlinear state."""

    limits = (
        mesh_quality_thresholds(completed.profile)
        if thresholds is None
        else thresholds
    )
    accepted = tuple(getattr(completed, "accepted_rows", ()))
    steps = tuple(getattr(getattr(completed, "result", None), "accepted_steps", ()))
    if not accepted or len(accepted) != len(steps):
        raise ValueError("accepted rows and states must be nonempty and aligned")
    return tuple(
        _quality_row(model, source, step, limits)
        for source, step in zip(accepted, steps, strict=True)
    )


def summarize_mesh_quality(
    rows: Sequence[Mapping[str, object]],
    thresholds: MeshQualityThresholds,
) -> dict[str, object]:
    """Assess complete accepted-state quality histories."""

    values = tuple(rows)
    if not values:
        raise ValueError("mesh-quality summary requires accepted states")
    worst_jacobian = min(
        values,
        key=lambda row: (
            float(row["minimum_jacobian"]),
            int(row["accepted_step"]),
        ),
    )
    worst_energy = max(
        values,
        key=lambda row: (
            float(row["maximum_normalized_energy_density"]),
            -int(row["accepted_step"]),
        ),
    )
    minimum_jacobian = float(worst_jacobian["minimum_jacobian"])
    maximum_energy = float(worst_energy["maximum_normalized_energy_density"])
    criteria = {
        "no_inverted_elements": all(not bool(row["inverted"]) for row in values),
        "minimum_jacobian_above_failure_limit": (
            minimum_jacobian > thresholds.failure_minimum_jacobian
        ),
        "normalized_energy_below_failure_limit": (
            maximum_energy <= thresholds.failure_normalized_energy_density
        ),
        "all_states_classified": all(
            str(row["status"]) in ("accepted", "warning", "failed")
            for row in values
        ),
    }
    return {
        "schema_version": SCHEMA,
        "profile": thresholds.profile,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "thresholds": thresholds.as_dict(),
        "accepted_state_count": len(values),
        "warning_state_count": sum(bool(row["warning"]) for row in values),
        "failed_state_count": sum(not bool(row["passed"]) for row in values),
        "minimum_jacobian": minimum_jacobian,
        "maximum_normalized_energy_density": maximum_energy,
        "worst_jacobian_state": {
            "accepted_step": int(worst_jacobian["accepted_step"]),
            "parameter": float(worst_jacobian["parameter"]),
            "element": int(worst_jacobian["minimum_jacobian_element"]),
            "body": str(worst_jacobian["minimum_jacobian_body"]),
            "body_element": int(worst_jacobian["minimum_jacobian_body_element"]),
            "value": minimum_jacobian,
        },
        "worst_energy_state": {
            "accepted_step": int(worst_energy["accepted_step"]),
            "parameter": float(worst_energy["parameter"]),
            "element": int(worst_energy["maximum_energy_element"]),
            "body": str(worst_energy["maximum_energy_body"]),
            "body_element": int(worst_energy["maximum_energy_body_element"]),
            "value": maximum_energy,
        },
    }


def audit_mesh_quality(
    model: RotatingBlocksModel,
    completed: object,
    *,
    thresholds: MeshQualityThresholds | None = None,
) -> MeshQualityHistory:
    """Collect and assess one production quality history."""

    limits = (
        mesh_quality_thresholds(completed.profile)
        if thresholds is None
        else thresholds
    )
    rows = collect_mesh_quality_rows(model, completed, thresholds=limits)
    return MeshQualityHistory(rows, summarize_mesh_quality(rows, limits))


def _interpolate(
    rows: Sequence[Mapping[str, object]],
    parameters: np.ndarray,
    field: str,
) -> np.ndarray:
    source_parameters = np.asarray([float(row["parameter"]) for row in rows])
    source_values = np.asarray([float(row[field]) for row in rows])
    order = np.argsort(source_parameters)
    unique, indices = np.unique(source_parameters[order], return_index=True)
    return np.interp(parameters, unique, source_values[order][indices])


def compare_mesh_quality_refinement(
    refinement: object,
    *,
    thresholds: MeshQualityThresholds | None = None,
) -> MeshQualityRefinement:
    """Compare medium/fine quality histories on the refinement path grid."""

    profile = rotating_blocks_execution_profile(refinement.profile)
    limits = mesh_quality_thresholds(profile) if thresholds is None else thresholds
    levels = tuple(refinement.levels)
    if len(levels) < 2:
        raise ValueError("mesh-quality refinement requires two levels")
    histories = []
    for level in levels:
        model = build_rotating_blocks_model(level.run.profile.model_profile)
        histories.append(audit_mesh_quality(model, level.run, thresholds=limits))
    medium = histories[-2]
    fine = histories[-1]
    parameters = np.asarray(refinement.comparison_parameters, dtype=float)
    medium_jacobian = _interpolate(medium.rows, parameters, "minimum_jacobian")
    fine_jacobian = _interpolate(fine.rows, parameters, "minimum_jacobian")
    medium_energy = _interpolate(
        medium.rows,
        parameters,
        "maximum_normalized_energy_density",
    )
    fine_energy = _interpolate(
        fine.rows,
        parameters,
        "maximum_normalized_energy_density",
    )
    rows = tuple(
        {
            "parameter": float(parameter),
            "medium_minimum_jacobian": float(medium_jacobian[index]),
            "fine_minimum_jacobian": float(fine_jacobian[index]),
            "absolute_jacobian_difference": float(
                abs(medium_jacobian[index] - fine_jacobian[index])
            ),
            "medium_maximum_normalized_energy_density": float(
                medium_energy[index]
            ),
            "fine_maximum_normalized_energy_density": float(fine_energy[index]),
            "absolute_normalized_energy_difference": float(
                abs(medium_energy[index] - fine_energy[index])
            ),
        }
        for index, parameter in enumerate(parameters)
    )
    maximum_jacobian = max(
        float(row["absolute_jacobian_difference"]) for row in rows
    )
    maximum_energy = max(
        float(row["absolute_normalized_energy_difference"]) for row in rows
    )
    criteria = {
        "all_levels_passed": all(history.passed for history in histories),
        "minimum_jacobian_history_agrees": (
            maximum_jacobian
            <= limits.maximum_refinement_jacobian_difference
        ),
        "normalized_energy_history_agrees": (
            maximum_energy <= limits.maximum_refinement_energy_difference
        ),
    }
    summary = {
        "schema_version": REFINEMENT_SCHEMA,
        "profile": profile.name,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "requested_steps": [int(level.requested_steps) for level in levels],
        "maximum_jacobian_difference": maximum_jacobian,
        "maximum_normalized_energy_difference": maximum_energy,
        "level_summaries": [history.summary for history in histories],
    }
    return MeshQualityRefinement(rows, summary)


def evaluate_mesh_quality(
    model: RotatingBlocksModel,
    completed: object,
    refinement: object,
) -> RotatingBlocksMeshQuality:
    """Evaluate production and refinement mesh-quality evidence."""

    limits = mesh_quality_thresholds(completed.profile)
    history = audit_mesh_quality(model, completed, thresholds=limits)
    compared = compare_mesh_quality_refinement(refinement, thresholds=limits)
    criteria = {
        "production_mesh_quality_passed": history.passed,
        "refinement_mesh_quality_passed": compared.passed,
        "production_evidence_complete": len(history.rows)
        == len(tuple(completed.accepted_rows)),
    }
    summary = {
        "schema_version": SCHEMA,
        "profile": limits.profile,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "thresholds": limits.as_dict(),
        "production": history.summary,
        "refinement": compared.summary,
    }
    return RotatingBlocksMeshQuality(history, compared, summary)


def write_mesh_quality_artifacts(
    writer: BenchmarkArtifactWriter,
    output: Path,
    quality: RotatingBlocksMeshQuality,
) -> tuple[str, ...]:
    """Write manifest-managed tables, summary, and deterministic plots."""

    paths = (
        "tables/mesh-quality.csv",
        "tables/refinement-mesh-quality.csv",
        "mesh-quality.json",
        "plots/mesh-quality.svg",
        "plots/mesh-quality-refinement.svg",
    )
    writer.write_csv(paths[0], quality.history.rows, schema=ROW_SCHEMA)
    writer.write_csv(
        paths[1],
        quality.refinement.rows,
        schema=REFINEMENT_SCHEMA,
    )
    writer.write_json(paths[2], quality.summary, schema=SCHEMA)
    root = Path(output)
    (root / "plots").mkdir(parents=True, exist_ok=True)
    parameters = np.asarray(
        [float(row["parameter"]) for row in quality.history.rows]
    )
    write_line_chart(
        root / paths[3],
        title="Rotating-blocks mesh quality",
        x_label="continuation parameter",
        y_label="scale-independent quality",
        x_values=parameters,
        series=(
            (
                np.asarray(
                    [float(row["minimum_jacobian"]) for row in quality.history.rows]
                ),
                "minimum Jacobian",
            ),
            (
                np.asarray(
                    [
                        float(row["maximum_normalized_energy_density"])
                        for row in quality.history.rows
                    ]
                ),
                "maximum energy / shear modulus",
            ),
        ),
    )
    refinement_parameters = np.asarray(
        [float(row["parameter"]) for row in quality.refinement.rows]
    )
    write_line_chart(
        root / paths[4],
        title="Rotating-blocks mesh-quality refinement",
        x_label="continuation parameter",
        y_label="medium/fine absolute difference",
        x_values=refinement_parameters,
        series=(
            (
                np.asarray(
                    [
                        float(row["absolute_jacobian_difference"])
                        for row in quality.refinement.rows
                    ]
                ),
                "minimum Jacobian",
            ),
            (
                np.asarray(
                    [
                        float(row["absolute_normalized_energy_difference"])
                        for row in quality.refinement.rows
                    ]
                ),
                "normalized energy density",
            ),
        ),
    )
    return paths
