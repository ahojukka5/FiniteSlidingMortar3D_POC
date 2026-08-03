from __future__ import annotations

import json
from pathlib import Path

import pytest

from contact3d.benchmark_artifacts import BenchmarkArtifactError
from contact3d.benchmark_goldens import (
    GOLDEN_SCHEMA_VERSION,
    evaluate_golden_spec,
    load_golden_directory,
    load_golden_spec,
    validate_golden_spec,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "benchmark": "unit-golden",
        "source": "summary.json",
        "profiles": ["full", "quick"],
        "metrics": {
            "metrics.force": {
                "reference": 10.0,
                "absolute": 1.0e-3,
                "relative": 0.0,
            },
            "metrics.iterations": {
                "reference": 4,
                "absolute": 0.0,
                "relative": 0.0,
            },
        },
    }


def test_golden_spec_evaluates_selected_metrics(tmp_path: Path) -> None:
    specification = validate_golden_spec(_payload())
    output = tmp_path / "unit-golden"
    output.mkdir()
    (output / "summary.json").write_text(
        json.dumps({"metrics": {"force": 10.0005, "iterations": 4}}),
        encoding="utf-8",
    )

    report = evaluate_golden_spec(specification, output, profile="quick")

    assert report["status"] == "passed"
    assert report["metric_count"] == 2
    assert report["metrics"]["metrics.force"]["absolute_error"] == pytest.approx(
        5.0e-4
    )


def test_golden_spec_skips_unselected_profile(tmp_path: Path) -> None:
    payload = _payload()
    payload["profiles"] = ["full"]
    specification = validate_golden_spec(payload)

    report = evaluate_golden_spec(specification, tmp_path, profile="quick")

    assert report["status"] == "skipped_profile"
    assert report["metric_count"] == 0


def test_golden_spec_reports_numeric_regression(tmp_path: Path) -> None:
    specification = validate_golden_spec(_payload())
    output = tmp_path / "unit-golden"
    output.mkdir()
    (output / "summary.json").write_text(
        json.dumps({"metrics": {"force": 10.1, "iterations": 4}}),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkArtifactError, match="metrics.force"):
        evaluate_golden_spec(specification, output, profile="full")


def test_golden_loader_rejects_invalid_schema_and_paths(tmp_path: Path) -> None:
    invalid = _payload()
    invalid["source"] = "../summary.json"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(BenchmarkArtifactError, match="stay under"):
        load_golden_spec(path)

    invalid = _payload()
    invalid["metrics"] = {
        "metrics.force": {"reference": True, "absolute": 0.0, "relative": 0.0}
    }
    with pytest.raises(BenchmarkArtifactError, match="must be numeric"):
        validate_golden_spec(invalid)


def test_golden_directory_loads_specs_and_rejects_duplicates(tmp_path: Path) -> None:
    first = _payload()
    second = _payload()
    second["benchmark"] = "another-golden"
    (tmp_path / "first.json").write_text(json.dumps(first), encoding="utf-8")
    (tmp_path / "second.json").write_text(json.dumps(second), encoding="utf-8")

    loaded = load_golden_directory(tmp_path)
    assert set(loaded) == {"unit-golden", "another-golden"}

    second["benchmark"] = "unit-golden"
    (tmp_path / "second.json").write_text(json.dumps(second), encoding="utf-8")
    with pytest.raises(BenchmarkArtifactError, match="duplicate golden"):
        load_golden_directory(tmp_path)


def test_checked_goldens_match_committed_full_results() -> None:
    repository = Path(__file__).resolve().parents[1]
    specifications = load_golden_directory(repository / "benchmarks" / "goldens")
    result_root = repository / "results"
    committed = {
        name: specification
        for name, specification in specifications.items()
        if (result_root / name / specification.source).is_file()
    }

    reports = {
        name: evaluate_golden_spec(
            specification,
            result_root / name,
            profile="full",
        )
        for name, specification in committed.items()
    }

    assert set(reports) == {
        "adaptive-topology-events",
        "broad-phase-scaling",
        "mixed-load-path",
        "nonlinear-equilibrium",
        "scale-aware-penalty",
        "topology-events",
    }
    assert "rotating-blocks" in specifications
    assert all(report["status"] == "passed" for report in reports.values())
