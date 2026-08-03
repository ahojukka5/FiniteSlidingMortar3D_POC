from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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


_load_module("rotating_blocks_profiles", "rotating_blocks_profiles.py")
RETENTION = _load_module("rotating_blocks_retention", "rotating_blocks_retention.py")


def _row(
    step: int,
    parameter: float,
    *,
    active: int = 2,
    supported: int = 3,
    overlap: float = 0.4,
    reaction: float = 1.5,
) -> dict[str, object]:
    return {
        "accepted_step": step,
        "parameter": parameter,
        "phase_parameter": parameter,
        "rotation_angle": parameter,
        "overlap_area": overlap,
        "supported_rows": supported,
        "active_rows": active,
        "normal_reaction": reaction,
        "maximum_gap": 1.0e-9,
        "maximum_separation": 0.1,
        "signature": f"signature-{step}",
    }


def test_retained_rotation_states_pass() -> None:
    limits = RETENTION.retention_thresholds("quick")
    source = (_row(1, 0.50), _row(2, 0.55), _row(3, 0.60))

    rows = RETENTION.classify_retention_rows(source, (), limits)
    summary = RETENTION.summarize_retention(rows, limits)

    assert summary["passed"]
    assert [row["status"] for row in rows] == ["retained"] * 3
    assert summary["localized_exception_count"] == 0


def test_single_event_bracketed_zero_active_state_is_localized() -> None:
    limits = RETENTION.retention_thresholds("quick")
    source = (
        _row(1, 0.50),
        _row(2, 0.55, active=0, reaction=0.0),
        _row(3, 0.60),
    )

    rows = RETENTION.classify_retention_rows(source, (0.55,), limits)
    summary = RETENTION.summarize_retention(rows, limits)

    assert summary["passed"]
    assert rows[1]["status"] == "localized_transition"
    assert rows[1]["localized_exception"]
    assert rows[1]["previous_signature"] == "signature-1"
    assert rows[1]["following_signature"] == "signature-3"


def test_sustained_contact_loss_reports_neighboring_signatures() -> None:
    limits = RETENTION.retention_thresholds("quick")
    source = (
        _row(1, 0.50),
        _row(2, 0.55, active=0, reaction=0.0),
        _row(3, 0.60, active=0, reaction=0.0),
        _row(4, 0.65),
    )

    rows = RETENTION.classify_retention_rows(source, (0.575,), limits)
    summary = RETENTION.summarize_retention(rows, limits)

    assert not summary["passed"]
    assert summary["failed_state_count"] == 2
    assert not summary["criteria"]["no_sustained_contact_loss"]
    assert rows[1]["status"] == "failed"
    assert "not_isolated" in rows[1]["failure_reasons"]
    assert rows[1]["previous_signature"] == "signature-1"
    assert rows[1]["following_signature"] == "signature-3"
    assert summary["first_failure"]["following_signature"] == "signature-3"


def test_missing_overlap_cannot_use_localized_exception() -> None:
    limits = RETENTION.retention_thresholds("quick")
    source = (
        _row(1, 0.50),
        _row(2, 0.55, active=0, overlap=0.0, reaction=0.0),
        _row(3, 0.60),
    )

    rows = RETENTION.classify_retention_rows(source, (0.55,), limits)

    assert not rows[1]["passed"]
    assert not rows[1]["localized_exception"]
    assert "overlap_below_limit" in rows[1]["failure_reasons"]


def test_retention_artifacts_are_manifest_validated(tmp_path: Path) -> None:
    limits = RETENTION.retention_thresholds("quick")
    rows = RETENTION.classify_retention_rows(
        (_row(1, 0.50), _row(2, 0.55), _row(3, 0.60)),
        (),
        limits,
    )
    result = RETENTION.RotatingBlocksRetention(
        rows,
        RETENTION.summarize_retention(rows, limits),
    )
    writer = BenchmarkArtifactWriter(
        tmp_path,
        "retention-test",
        seed=0,
        solver_settings={"profile": "quick"},
    )

    required = RETENTION.write_retention_artifacts(writer, tmp_path, result)
    writer.finalize(required=required)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    validate_benchmark_manifest(manifest, root=tmp_path)
    paths = {record["path"] for record in manifest["artifacts"]}
    assert "tables/contact-retention.csv" in paths
    assert "contact-retention.json" in paths
    assert "plots/contact-retention-metrics.svg" in paths
    assert "plots/contact-retention-status.svg" in paths
