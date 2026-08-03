from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from contact3d.benchmark_artifacts import (
    BenchmarkArtifactWriter,
    validate_benchmark_manifest,
)

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))


def _load_module(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, BENCHMARKS / filename)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


_load_module("rotating_blocks_model", "rotating_blocks_model.py")
PROFILES = _load_module("rotating_blocks_profiles", "rotating_blocks_profiles.py")
GATE = _load_module("rotating_blocks_gate", "rotating_blocks_gate.py")


def _completed(*, passed: bool = True) -> object:
    profile = PROFILES.rotating_blocks_execution_profile("quick")
    accepted = (
        {
            "parameter": 0.25,
            "phase_index": 1,
            "overlap_area": 0.4 if passed else 0.0,
            "supported_rows": 3 if passed else 0,
        },
        {
            "parameter": 1.0 if passed else 0.8,
            "phase_index": 1,
            "overlap_area": 0.2 if passed else 0.0,
            "supported_rows": 2 if passed else 0,
        },
    )
    attempts = (
        {
            "action": "accepted",
            "normalized_equilibrium_residual": 1.0e-10 if passed else 1.0e-2,
            "normalized_maximum_penetration": 1.0e-10 if passed else 1.0e-2,
        },
    )
    return SimpleNamespace(
        profile=profile,
        passed=passed,
        accepted_rows=accepted,
        attempt_rows=attempts,
        summary={
            "criteria": {"solver_converged": passed},
            "final_parameter": 1.0 if passed else 0.8,
            "maximum_normalized_equilibrium_residual": (
                1.0e-10 if passed else 1.0e-2
            ),
            "maximum_normalized_penetration": 1.0e-10 if passed else 1.0e-2,
        },
    )


def _refinement(*, passed: bool = True) -> object:
    return SimpleNamespace(
        passed=passed,
        comparison_rows=(),
        event_rows=(),
        summary={
            "passed": passed,
            "criteria": {"event_counts_match": passed},
            "maximum_relative_field_errors": {
                "reaction_x": 1.0e-3 if passed else 2.0e-1
            },
            "maximum_event_location_error": 1.0e-3 if passed else 2.0e-1,
        },
    )


def _balance(*, passed: bool = True) -> dict[str, object]:
    value = 1.0e-10 if passed else 1.0e-2
    return {
        "passed": passed,
        "maximum_normalized_errors": {
            "normalized_global_force_error": value,
            "normalized_contact_force_error": value,
            "normalized_global_moment_origin_error": value,
            "normalized_global_moment_pivot_error": value,
            "normalized_contact_moment_origin_error": value,
            "normalized_contact_moment_pivot_error": value,
        },
    }


def _determinism(*, passed: bool = True) -> dict[str, object]:
    return {
        "passed": passed,
        "sample_count": 65,
        "maximum_absolute_error": 0.0 if passed else 1.0e-3,
        "maximum_relative_error": 0.0 if passed else 1.0e-3,
        "first_divergence": None if passed else {"field": "topology_signature"},
    }


def test_acceptance_gate_passes_complete_evidence() -> None:
    gate = GATE.evaluate_acceptance_gate(
        _completed(),
        _refinement(),
        _balance(),
        {"passed": True},
        determinism=_determinism(),
    )

    assert gate.passed
    assert gate.summary["failed_count"] == 0
    assert gate.summary["criterion_count"] == len(gate.rows)
    assert all(row["passed"] for row in gate.rows)
    assert {row["category"] for row in gate.rows} >= {
        "convergence",
        "contact_retention",
        "kkt",
        "balance",
        "determinism",
        "refinement",
        "pressure",
    }


def test_acceptance_gate_reports_every_failure_with_values_and_limits() -> None:
    gate = GATE.evaluate_acceptance_gate(
        _completed(passed=False),
        _refinement(passed=False),
        _balance(passed=False),
        {"passed": False},
        determinism=_determinism(passed=False),
    )

    assert not gate.passed
    assert gate.summary["failed_count"] >= 12
    assert "final_motion_reached" in gate.summary["failed_criteria"]
    assert "rotation_overlap_retained" in gate.summary["failed_criteria"]
    assert "event_history_deterministic" in gate.summary["failed_criteria"]
    message = GATE.acceptance_failure_message(gate.summary)
    assert "observed=" in message
    assert "required" in message
    assert "normalized_force_balance" in message
    assert "refinement_field_agreement" in message


def test_profile_thresholds_are_explicit_and_versioned() -> None:
    quick = GATE.acceptance_thresholds("quick")
    full = GATE.acceptance_thresholds("full")

    assert quick.profile == "quick"
    assert full.profile == "full"
    assert full.maximum_event_location_error < quick.maximum_event_location_error
    assert quick.as_dict()["maximum_normalized_force_error"] == 1.0e-7
    assert GATE.SCHEMA.endswith("/v1")


def test_gate_artifacts_validate_through_manifest(tmp_path: Path) -> None:
    gate = GATE.evaluate_acceptance_gate(
        _completed(),
        _refinement(),
        _balance(),
        {"passed": True},
        determinism=_determinism(),
    )
    writer = BenchmarkArtifactWriter(
        tmp_path,
        "rotating-blocks-gate-test",
        seed=0,
        solver_settings={},
    )

    required = GATE.write_gate_artifacts(writer, gate)
    writer.finalize(required=required)

    validate_benchmark_manifest(
        json.loads((tmp_path / "manifest.json").read_text()),
        root=tmp_path,
    )
    with (tmp_path / "tables/acceptance-gate.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == gate.summary["criterion_count"]
    assert json.loads((tmp_path / "acceptance-gate.json").read_text())["passed"]
