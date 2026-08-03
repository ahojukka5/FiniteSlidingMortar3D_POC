"""Acceptance-gate integration for rotating-blocks contact retention."""

from __future__ import annotations

from dataclasses import replace


def _failure_context(retention: object) -> str:
    summary = retention.summary
    failure = summary.get("first_failure")
    if not isinstance(failure, dict):
        return ""
    return (
        "; first_failure="
        f"parameter={failure.get('parameter')!r}, "
        f"reasons={failure.get('failure_reasons')!r}, "
        f"previous_signature={failure.get('previous_signature')!r}, "
        f"following_signature={failure.get('following_signature')!r}"
    )


def include_retention_in_gate(gate: object, retention: object) -> object:
    """Return the gate with one complete contact-retention criterion."""

    if any(row.get("criterion") == "contact_retention_monitor" for row in gate.rows):
        raise ValueError("contact-retention criterion is already present")
    passed = bool(retention.passed)
    row = {
        "criterion": "contact_retention_monitor",
        "category": "contact_retention",
        "observed": passed,
        "relation": "==",
        "limit": True,
        "passed": passed,
        "message": (
            "contact_retention_monitor: "
            f"observed={passed!r}; required == True"
            + _failure_context(retention)
        ),
    }
    rows = (*gate.rows, row)
    failed = tuple(value for value in rows if not value["passed"])
    summary = {
        **gate.summary,
        "passed": not failed,
        "criterion_count": len(rows),
        "failed_count": len(failed),
        "criteria": list(rows),
        "failed_criteria": [str(value["criterion"]) for value in failed],
        "failure_messages": [str(value["message"]) for value in failed],
        "contact_retention": retention.summary,
    }
    return replace(gate, rows=rows, summary=summary)
