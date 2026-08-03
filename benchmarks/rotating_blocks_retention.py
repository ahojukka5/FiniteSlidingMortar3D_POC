"""Contact-retention evidence for the rotating-blocks benchmark."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from rotating_blocks_profiles import (
    RotatingBlocksExecutionProfile,
    rotating_blocks_execution_profile,
)

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.benchmark_plots import write_category_timeline, write_line_chart
from contact3d.topology_signature import topology_signature_hash

SCHEMA = "contact3d-rotating-blocks-retention/v1"
ROW_SCHEMA = "contact3d-rotating-blocks-retention-rows/v1"


@dataclass(frozen=True, slots=True)
class RetentionThresholds:
    """Profile-aware limits for accepted rotation-phase contact states."""

    profile: str
    minimum_overlap_area: float
    minimum_supported_rows: int
    minimum_active_rows: int
    minimum_normal_reaction: float
    maximum_localized_interval: float

    def __post_init__(self) -> None:
        if self.profile not in ("quick", "full"):
            raise ValueError("retention profile must be 'quick' or 'full'")
        numeric = (
            self.minimum_overlap_area,
            self.minimum_normal_reaction,
            self.maximum_localized_interval,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError("retention thresholds must be finite and nonnegative")
        if self.minimum_supported_rows <= 0 or self.minimum_active_rows <= 0:
            raise ValueError("retention row thresholds must be positive")
        if self.maximum_localized_interval <= 0.0:
            raise ValueError("localized interval limit must be positive")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RotatingBlocksRetention:
    """Classified rotation states and aggregate contact-retention result."""

    rows: tuple[dict[str, object], ...]
    summary: dict[str, object]

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])


def retention_thresholds(
    profile: str | RotatingBlocksExecutionProfile,
) -> RetentionThresholds:
    selected = rotating_blocks_execution_profile(profile)
    return RetentionThresholds(
        profile=selected.name,
        minimum_overlap_area=1.0e-12,
        minimum_supported_rows=1,
        minimum_active_rows=1,
        minimum_normal_reaction=1.0e-12,
        maximum_localized_interval=2.0 / selected.requested_path_steps,
    )


def _event_parameters(completed: object) -> tuple[float, ...]:
    values = {
        float(row["continuation_parameter"])
        for row in tuple(getattr(completed, "event_rows", ()))
        if "continuation_parameter" in row
    }
    return tuple(sorted(values))


def _contact_state(step: object) -> tuple[float, float, str]:
    evaluation = step.result.equilibrium.evaluation
    contacts = tuple(evaluation.contacts)
    if len(contacts) != 1:
        raise ValueError("rotating-blocks retention requires one contact interface")
    contact = contacts[0]
    gaps = np.asarray(contact.normal_gaps, dtype=float)
    pressure = np.asarray(contact.pressure, dtype=float)
    areas = np.asarray(contact.raw.contact.weights.row_areas, dtype=float)
    if gaps.ndim != 1 or pressure.shape != gaps.shape or areas.shape != gaps.shape:
        raise ValueError("retention contact fields must share one nodal shape")
    if not np.all(np.isfinite(gaps)) or not np.all(np.isfinite(pressure)):
        raise ValueError("retention contact fields must be finite")
    if not np.all(np.isfinite(areas)) or np.any(areas < 0.0):
        raise ValueError("retention row areas must be finite and nonnegative")
    normal_reaction = float(np.dot(pressure, areas))
    maximum_gap = float(np.max(gaps, initial=0.0))
    return normal_reaction, maximum_gap, topology_signature_hash(contact.signature)


def collect_retention_rows(completed: object) -> tuple[dict[str, object], ...]:
    """Collect physical and topology fields for accepted rotation states."""

    accepted = tuple(getattr(completed, "accepted_rows", ()))
    steps = tuple(getattr(getattr(completed, "result", None), "accepted_steps", ()))
    if len(accepted) != len(steps):
        raise ValueError("accepted-step rows and solver states must have equal length")
    rows: list[dict[str, object]] = []
    for source, step in zip(accepted, steps, strict=True):
        if int(source.get("phase_index", -1)) != 1:
            continue
        normal_reaction, maximum_gap, signature = _contact_state(step)
        contact = step.result.equilibrium.evaluation.contacts[0]
        gaps = np.asarray(contact.normal_gaps, dtype=float)
        rows.append(
            {
                "accepted_step": int(source["accepted_step"]),
                "parameter": float(source["parameter"]),
                "phase_parameter": float(source["phase_parameter"]),
                "rotation_angle": float(source["rotation_angle"]),
                "overlap_area": float(source["overlap_area"]),
                "supported_rows": int(source["supported_rows"]),
                "active_rows": int(source["active_rows"]),
                "normal_reaction": normal_reaction,
                "maximum_gap": maximum_gap,
                "maximum_separation": float(np.max(-gaps, initial=0.0)),
                "signature": signature,
            }
        )
    return tuple(rows)


def _structural_contact(row: Mapping[str, object], limits: RetentionThresholds) -> bool:
    return (
        float(row["overlap_area"]) >= limits.minimum_overlap_area
        and int(row["supported_rows"]) >= limits.minimum_supported_rows
    )


def _load_bearing(row: Mapping[str, object], limits: RetentionThresholds) -> bool:
    return (
        int(row["active_rows"]) >= limits.minimum_active_rows
        and float(row["normal_reaction"]) >= limits.minimum_normal_reaction
    )


def _crosses_event(
    left: float,
    right: float,
    events: Sequence[float],
) -> bool:
    return any(left <= value <= right for value in events)


def classify_retention_rows(
    rows: Sequence[Mapping[str, object]],
    event_parameters: Sequence[float],
    thresholds: RetentionThresholds,
) -> tuple[dict[str, object], ...]:
    """Classify retained, localized-transition, and failed contact states."""

    source = tuple(dict(row) for row in rows)
    unordered = any(
        float(right["parameter"]) <= float(left["parameter"])
        for left, right in zip(source, source[1:], strict=False)
    )
    if unordered:
        raise ValueError("retention parameters must be strictly increasing")
    classified: list[dict[str, object]] = []
    for index, row in enumerate(source):
        previous = source[index - 1] if index > 0 else None
        following = source[index + 1] if index + 1 < len(source) else None
        structural = _structural_contact(row, thresholds)
        load_bearing = _load_bearing(row, thresholds)
        isolated = (
            previous is not None
            and following is not None
            and _structural_contact(previous, thresholds)
            and _structural_contact(following, thresholds)
            and _load_bearing(previous, thresholds)
            and _load_bearing(following, thresholds)
            and not load_bearing
        )
        interval = (
            float(following["parameter"]) - float(previous["parameter"])
            if previous is not None and following is not None
            else float("inf")
        )
        event_bracketed = (
            previous is not None
            and following is not None
            and _crosses_event(
                float(previous["parameter"]),
                float(following["parameter"]),
                event_parameters,
            )
        )
        localized = (
            structural
            and isolated
            and event_bracketed
            and interval <= thresholds.maximum_localized_interval
        )
        passed = structural and (load_bearing or localized)
        reasons: list[str] = []
        if float(row["overlap_area"]) < thresholds.minimum_overlap_area:
            reasons.append("overlap_below_limit")
        if int(row["supported_rows"]) < thresholds.minimum_supported_rows:
            reasons.append("support_below_limit")
        if int(row["active_rows"]) < thresholds.minimum_active_rows:
            reasons.append("active_rows_below_limit")
        if float(row["normal_reaction"]) < thresholds.minimum_normal_reaction:
            reasons.append("normal_reaction_below_limit")
        if not passed and structural and not load_bearing:
            if not isolated:
                reasons.append("not_isolated")
            if not event_bracketed:
                reasons.append("not_event_bracketed")
            if interval > thresholds.maximum_localized_interval:
                reasons.append("localized_interval_too_wide")
        status = "retained" if load_bearing and structural else "localized_transition"
        if not passed:
            status = "failed"
        classified.append(
            {
                **row,
                "status": status,
                "passed": passed,
                "failure_reasons": ";".join(reasons),
                "localized_exception": localized,
                "event_bracketed": event_bracketed,
                "bracket_interval": None if not np.isfinite(interval) else interval,
                "previous_parameter": (
                    None if previous is None else float(previous["parameter"])
                ),
                "previous_signature": (
                    None if previous is None else str(previous["signature"])
                ),
                "following_parameter": (
                    None if following is None else float(following["parameter"])
                ),
                "following_signature": (
                    None if following is None else str(following["signature"])
                ),
            }
        )
    return tuple(classified)


def summarize_retention(
    rows: Sequence[Mapping[str, object]],
    thresholds: RetentionThresholds,
) -> dict[str, object]:
    """Summarize every classified state without hiding sustained loss."""

    values = tuple(dict(row) for row in rows)
    failed = tuple(row for row in values if not bool(row["passed"]))
    localized = tuple(row for row in values if bool(row["localized_exception"]))
    consecutive_failures = any(
        not bool(left["passed"]) and not bool(right["passed"])
        for left, right in zip(values, values[1:], strict=False)
    )
    criteria = {
        "rotation_states_present": bool(values),
        "all_rotation_states_classified": bool(values)
        and all(str(row["status"]) for row in values),
        "no_unexplained_contact_loss": not failed,
        "no_sustained_contact_loss": not consecutive_failures,
        "localized_exceptions_bounded": all(
            float(row["bracket_interval"]) <= thresholds.maximum_localized_interval
            for row in localized
        ),
    }
    worst = None
    if failed:
        row = failed[0]
        worst = {
            "accepted_step": int(row["accepted_step"]),
            "parameter": float(row["parameter"]),
            "failure_reasons": str(row["failure_reasons"]),
            "previous_signature": row["previous_signature"],
            "following_signature": row["following_signature"],
        }
    return {
        "schema_version": SCHEMA,
        "profile": thresholds.profile,
        "passed": all(criteria.values()),
        "thresholds": thresholds.as_dict(),
        "criteria": criteria,
        "rotation_state_count": len(values),
        "retained_state_count": sum(row["status"] == "retained" for row in values),
        "localized_exception_count": len(localized),
        "failed_state_count": len(failed),
        "minimum_overlap_area": min(
            (float(row["overlap_area"]) for row in values),
            default=0.0,
        ),
        "minimum_supported_rows": min(
            (int(row["supported_rows"]) for row in values),
            default=0,
        ),
        "minimum_active_rows": min(
            (int(row["active_rows"]) for row in values),
            default=0,
        ),
        "minimum_normal_reaction": min(
            (float(row["normal_reaction"]) for row in values),
            default=0.0,
        ),
        "maximum_gap": max(
            (float(row["maximum_gap"]) for row in values),
            default=0.0,
        ),
        "maximum_separation": max(
            (float(row["maximum_separation"]) for row in values),
            default=0.0,
        ),
        "first_failure": worst,
    }


def audit_contact_retention(completed: object) -> RotatingBlocksRetention:
    """Collect and classify contact retention after the compression phase."""

    limits = retention_thresholds(completed.profile)
    collected = collect_retention_rows(completed)
    rows = classify_retention_rows(collected, _event_parameters(completed), limits)
    return RotatingBlocksRetention(rows, summarize_retention(rows, limits))


def write_retention_artifacts(
    writer: BenchmarkArtifactWriter,
    output: Path,
    retention: RotatingBlocksRetention,
) -> tuple[str, ...]:
    """Write manifest-validated retention tables, summary, and plots."""

    paths = (
        "tables/contact-retention.csv",
        "contact-retention.json",
        "plots/contact-retention-metrics.svg",
        "plots/contact-retention-status.svg",
    )
    (Path(output) / "plots").mkdir(parents=True, exist_ok=True)
    writer.write_csv(paths[0], retention.rows, schema=ROW_SCHEMA)
    writer.write_json(paths[1], retention.summary, schema=SCHEMA)
    parameters = np.asarray([float(row["parameter"]) for row in retention.rows])
    write_line_chart(
        output / paths[2],
        title="Rotating-blocks contact retention",
        x_label="continuation parameter",
        y_label="normalized metric",
        x_values=parameters,
        series=(
            (
                np.asarray([float(row["overlap_area"]) for row in retention.rows]),
                "overlap area",
            ),
            (
                np.asarray([float(row["normal_reaction"]) for row in retention.rows]),
                "normal reaction",
            ),
            (
                np.asarray([float(row["active_rows"]) for row in retention.rows]),
                "active rows",
            ),
        ),
        show_markers=True,
    )
    write_category_timeline(
        output / paths[3],
        title="Rotating-blocks retention classifications",
        x_label="continuation parameter",
        categories=tuple(str(row["status"]) for row in retention.rows),
        x_values=parameters,
        groups=tuple(row["accepted_step"] for row in retention.rows),
    )
    for path in paths[2:]:
        ElementTree.parse(output / path)
        writer.register(path, "svg")
    return paths
