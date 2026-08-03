from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))


def _load_module(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, BENCHMARKS / filename)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


GATE = _load_module("rotating_blocks_gate", "rotating_blocks_gate.py")
INTEGRATION = _load_module(
    "rotating_blocks_retention_gate",
    "rotating_blocks_retention_gate.py",
)


def _gate() -> object:
    row = {
        "criterion": "solver_converged",
        "category": "convergence",
        "observed": True,
        "relation": "==",
        "limit": True,
        "passed": True,
        "message": "solver_converged: observed=True; required == True",
    }
    summary = {
        "passed": True,
        "criterion_count": 1,
        "failed_count": 0,
        "criteria": [row],
        "failed_criteria": [],
        "failure_messages": [],
    }
    return GATE.RotatingBlocksAcceptanceGate((row,), summary)


def test_passing_retention_extends_gate_once() -> None:
    retention = SimpleNamespace(
        passed=True,
        summary={"passed": True, "first_failure": None},
    )

    result = INTEGRATION.include_retention_in_gate(_gate(), retention)

    assert result.passed
    assert result.summary["criterion_count"] == 2
    assert result.rows[-1]["criterion"] == "contact_retention_monitor"
    assert result.rows[-1]["category"] == "contact_retention"
    assert result.summary["contact_retention"]["passed"]


def test_retention_failure_reports_neighboring_signatures() -> None:
    retention = SimpleNamespace(
        passed=False,
        summary={
            "passed": False,
            "first_failure": {
                "parameter": 0.625,
                "failure_reasons": "active_rows_below_limit;not_isolated",
                "previous_signature": "left-signature",
                "following_signature": "right-signature",
            },
        },
    )

    result = INTEGRATION.include_retention_in_gate(_gate(), retention)

    assert not result.passed
    assert result.summary["failed_criteria"] == ["contact_retention_monitor"]
    message = result.summary["failure_messages"][0]
    assert "left-signature" in message
    assert "right-signature" in message
    assert "active_rows_below_limit" in message
