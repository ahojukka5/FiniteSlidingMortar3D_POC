from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

from contact3d.benchmark_artifacts import validate_benchmark_manifest


def _load_standardized_runner(repository: Path):
    path = repository / "benchmarks" / "run_standardized.py"
    specification = importlib.util.spec_from_file_location(
        "run_standardized_profile_test",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_rotating_blocks_uses_profile_specific_runners() -> None:
    repository = Path(__file__).resolve().parents[1]
    runner = _load_standardized_runner(repository)

    assert runner._benchmark_script("rotating-blocks", "quick") == (
        "rotating_blocks_quick.py"
    )
    assert runner._benchmark_script("rotating-blocks", "full") == (
        "rotating_blocks_bundle.py"
    )


def test_standardized_benchmarks_write_valid_artifacts(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "standardized"
    subprocess.run(
        [
            sys.executable,
            str(repository / "benchmarks" / "run_standardized.py"),
            "--quick",
            "--output",
            str(output),
        ],
        cwd=repository,
        check=True,
    )

    summary = json.loads(
        (output / "suite-summary.json").read_text(encoding="utf-8")
    )
    required = {
        "tet4-patch",
        "nonlinear-equilibrium",
        "coupled-mortar-patch",
        "adaptive-contact-policy",
        "adaptive-topology-events",
        "mixed-load-path",
        "mixed-contact-onset",
        "scale-aware-penalty",
        "warped-nonmatching-adapter",
        "warped-nonmatching-contact-onset",
        "topology-events",
        "broad-phase-scaling",
        "linear-solver-scaling",
        "rotating-blocks",
    }
    benchmark_rows = {
        row["benchmark"]: row for row in summary["benchmarks"]
    }
    expected = set(benchmark_rows)
    assert summary["profile"] == "quick"
    assert summary["benchmark_count"] == len(expected)
    assert required <= expected

    golden = json.loads(
        (output / summary["golden_report"]).read_text(encoding="utf-8")
    )
    statuses = {row["benchmark"]: row["status"] for row in golden["reports"]}
    passed = [row for row in golden["reports"] if row["status"] == "passed"]
    evaluated_metrics = sum(int(row["metric_count"]) for row in passed)
    assert golden["verification_enabled"]
    assert golden["configured_benchmarks"] >= 7
    assert golden["evaluated_benchmarks"] == len(passed)
    assert golden["evaluated_metrics"] == evaluated_metrics
    assert summary["golden_evaluated_benchmarks"] == len(passed)
    assert summary["golden_evaluated_metrics"] == evaluated_metrics
    assert statuses["nonlinear-equilibrium"] == "passed"
    assert statuses["adaptive-topology-events"] == "passed"
    assert statuses["mixed-load-path"] == "passed"
    assert statuses["scale-aware-penalty"] == "passed"
    assert statuses["topology-events"] == "passed"
    assert statuses["rotating-blocks"] == "passed"
    assert statuses["broad-phase-scaling"] == "skipped_profile"
    assert statuses["linear-solver-scaling"] == "not_configured"

    for name in expected:
        directory = output / name
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        validate_benchmark_manifest(manifest, root=directory)
        assert manifest["benchmark"] == name
        assert manifest["artifacts"]

    summary_schemas = {
        "adaptive-topology-events": "contact3d-adaptive-topology-events/v1",
        "mixed-load-path": "contact3d-mixed-load-path/v1",
        "mixed-contact-onset": "contact3d-mixed-contact-onset/v1",
        "scale-aware-penalty": "contact3d-scale-aware-penalty/v1",
        "warped-nonmatching-contact-onset": (
            "contact3d-warped-contact-onset/v1"
        ),
        "topology-events": "contact3d-topology-events/v1",
        "broad-phase-scaling": "contact3d-broad-phase-scaling/v1",
        "linear-solver-scaling": "contact3d-linear-solver-scaling/v1",
        "rotating-blocks": "contact3d-rotating-blocks-quick/v1",
    }
    for name, schema in summary_schemas.items():
        payload = json.loads(
            (output / name / "summary.json").read_text(encoding="utf-8")
        )
        assert payload["schema_version"] == schema

    adaptive_events = json.loads(
        (output / "adaptive-topology-events" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert adaptive_events["metrics"]["all_right_branch"]
    assert adaptive_events["metrics"]["continuation_parameters"] == [
        0.75,
        0.375,
        0.75,
        1.0,
    ]
    assert adaptive_events["metrics"]["solver_load_factors"] == [1.0] * 4

    linear = json.loads(
        (output / "linear-solver-scaling" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert linear["acceptance"]["passed"]
    assert linear["settings"]["minimum_free_dofs"] == 0

    broad_phase = json.loads(
        (output / "broad-phase-scaling" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert broad_phase["metrics"]["all_pair_sets_equal"]

    topology = json.loads(
        (output / "topology-events" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert topology["metrics"]["branch_selection"] == "right"

    rotating = json.loads(
        (output / "rotating-blocks" / "summary.json").read_text(encoding="utf-8")
    )
    assert rotating["passed"]
    assert rotating["acceptance_gate"]["passed"]
    assert all(rotating["acceptance_gate"]["criteria"].values())
    assert not rotating["optional_evidence"]["refinement_executed"]
    assert not rotating["optional_evidence"]["repetition_executed"]
    assert rotating["table_row_counts"]["checkpoint_requests"] == 6
    assert rotating["table_row_counts"]["checkpoint_exports"] == 1

    vtk_files = (
        output / "tet4-patch" / "affine-patch.vtu",
        output / "nonlinear-equilibrium" / "deformed.vtu",
        output / "coupled-mortar-patch" / "deformed.vtu",
        output / "coupled-mortar-patch" / "slave-contact.vtp",
        output / "coupled-mortar-patch" / "master-contact.vtp",
        output / "mixed-contact-onset" / "deformed.vtu",
        output / "mixed-contact-onset" / "slave-contact.vtp",
        output / "mixed-contact-onset" / "master-contact.vtp",
        output / "warped-nonmatching-adapter" / "projected-overlap.vtp",
        output / "warped-nonmatching-contact-onset" / "deformed.vtu",
        output / "warped-nonmatching-contact-onset" / "slave-contact.vtp",
        output / "warped-nonmatching-contact-onset" / "master-contact.vtp",
        output / "warped-nonmatching-contact-onset" / "projected-overlap.vtp",
    )
    for path in vtk_files:
        root = ElementTree.parse(path).getroot()
        assert root.tag == "VTKFile"

    final_directory = output / "rotating-blocks" / "checkpoints" / "00-final"
    for filename in (
        "volume.vtu",
        "slave-contact.vtp",
        "master-contact.vtp",
        "projected-overlap.vtp",
    ):
        root = ElementTree.parse(final_directory / filename).getroot()
        assert root.tag == "VTKFile"
