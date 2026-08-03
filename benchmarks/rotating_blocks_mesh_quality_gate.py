"""Acceptance-gate integration for rotating-blocks mesh quality."""

from __future__ import annotations

from rotating_blocks_gate import RotatingBlocksAcceptanceGate
from rotating_blocks_mesh_quality import RotatingBlocksMeshQuality


def _criterion(
    name: str,
    observed: object,
    relation: str,
    limit: object,
    passed: bool,
    detail: str,
) -> dict[str, object]:
    return {
        "criterion": name,
        "category": "mesh_quality",
        "observed": observed,
        "relation": relation,
        "limit": limit,
        "passed": bool(passed),
        "message": (
            f"{name}: observed={observed!r}; required {relation} {limit!r}; "
            f"{detail}"
        ),
    }


def include_mesh_quality_in_gate(
    gate: RotatingBlocksAcceptanceGate,
    quality: RotatingBlocksMeshQuality,
) -> RotatingBlocksAcceptanceGate:
    """Return a gate extended with complete mesh-quality criteria."""

    production = quality.history.summary
    refinement = quality.refinement.summary
    thresholds = quality.summary["thresholds"]
    assert isinstance(thresholds, dict)
    worst = production["worst_jacobian_state"]
    assert isinstance(worst, dict)
    minimum_jacobian = float(production["minimum_jacobian"])
    minimum_limit = float(thresholds["failure_minimum_jacobian"])
    detail = (
        f"element={worst['element']}; body={worst['body']}; "
        f"body_element={worst['body_element']}; parameter={worst['parameter']}"
    )
    added = (
        _criterion(
            "mesh_quality_production_passed",
            quality.history.passed,
            "==",
            True,
            quality.history.passed,
            detail,
        ),
        _criterion(
            "mesh_minimum_jacobian",
            minimum_jacobian,
            ">",
            minimum_limit,
            minimum_jacobian > minimum_limit,
            detail,
        ),
        _criterion(
            "mesh_quality_refinement_passed",
            quality.refinement.passed,
            "==",
            True,
            quality.refinement.passed,
            (
                "maximum_jacobian_difference="
                f"{refinement['maximum_jacobian_difference']}; "
                "maximum_normalized_energy_difference="
                f"{refinement['maximum_normalized_energy_difference']}"
            ),
        ),
    )
    rows = (*gate.rows, *added)
    failed = tuple(row for row in rows if not bool(row["passed"]))
    summary = dict(gate.summary)
    summary.update(
        {
            "passed": not failed,
            "criterion_count": len(rows),
            "failed_count": len(failed),
            "criteria": list(rows),
            "failed_criteria": [str(row["criterion"]) for row in failed],
            "failure_messages": [str(row["message"]) for row in failed],
            "mesh_quality": quality.summary,
        }
    )
    return RotatingBlocksAcceptanceGate(tuple(rows), summary)
