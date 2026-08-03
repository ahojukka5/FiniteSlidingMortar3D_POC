from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from contact3d.benchmark_artifacts import (
    BenchmarkArtifactWriter,
    validate_benchmark_manifest,
)

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))

import rotating_blocks_mesh_quality as QUALITY  # noqa: E402
import rotating_blocks_mesh_quality_gate as QUALITY_GATE  # noqa: E402
from rotating_blocks_gate import RotatingBlocksAcceptanceGate  # noqa: E402
from rotating_blocks_model import build_rotating_blocks_model  # noqa: E402
from rotating_blocks_profiles import QUICK_PROFILE  # noqa: E402


def _completed(
    model,
    *,
    minimum_jacobian: float = 0.9,
    normalized_energy: float = 0.1,
):
    count = model.problem.mesh.element_count
    rows = []
    steps = []
    for accepted_step, parameter in enumerate((0.25, 0.625, 1.0), start=1):
        jacobians = np.ones(count)
        jacobians[accepted_step - 1] = minimum_jacobian
        energy = np.zeros(count)
        energy[accepted_step - 1] = (
            normalized_energy * model.problem.material.shear_modulus
        )
        evaluations = tuple(
            SimpleNamespace(jacobian=float(jacobian), energy_density=float(density))
            for jacobian, density in zip(jacobians, energy, strict=True)
        )
        steps.append(
            SimpleNamespace(
                result=SimpleNamespace(
                    equilibrium=SimpleNamespace(
                        evaluation=SimpleNamespace(
                            bulk=SimpleNamespace(element_evaluations=evaluations)
                        )
                    )
                )
            )
        )
        rows.append(
            {
                "accepted_step": accepted_step,
                "parameter": parameter,
                "phase_index": 0 if accepted_step == 1 else 1,
                "phase_parameter": 0.5 * accepted_step,
                "rotation_angle": 0.25 * accepted_step,
            }
        )
    return SimpleNamespace(
        profile=QUICK_PROFILE,
        accepted_rows=tuple(rows),
        result=SimpleNamespace(accepted_steps=tuple(steps)),
    )


def _refinement(completed):
    return SimpleNamespace(
        profile=QUICK_PROFILE,
        levels=(
            SimpleNamespace(requested_steps=16, run=completed),
            SimpleNamespace(requested_steps=32, run=completed),
        ),
        comparison_parameters=(0.25, 0.625, 1.0),
    )


def test_quality_history_identifies_worst_element_and_body() -> None:
    model = build_rotating_blocks_model("quick")
    completed = _completed(model, minimum_jacobian=0.4, normalized_energy=0.7)

    history = QUALITY.audit_mesh_quality(model, completed)

    assert history.passed
    assert history.summary["warning_state_count"] == 3
    assert history.summary["minimum_jacobian"] == 0.4
    worst = history.summary["worst_jacobian_state"]
    assert worst["element"] == 0
    assert worst["body"] == "lower"
    assert worst["parameter"] == 0.25
    assert all(row["status"] == "warning" for row in history.rows)


def test_quality_history_rejects_near_inversion_and_reports_state() -> None:
    model = build_rotating_blocks_model("quick")
    completed = _completed(model, minimum_jacobian=0.01)

    history = QUALITY.audit_mesh_quality(model, completed)

    assert not history.passed
    assert not history.summary["criteria"][
        "minimum_jacobian_above_failure_limit"
    ]
    assert history.summary["failed_state_count"] == 3
    assert history.summary["worst_jacobian_state"]["element"] == 0


def test_refinement_quality_compares_complete_histories() -> None:
    model = build_rotating_blocks_model("quick")
    completed = _completed(model)

    compared = QUALITY.compare_mesh_quality_refinement(_refinement(completed))

    assert compared.passed
    assert len(compared.rows) == 3
    assert compared.summary["maximum_jacobian_difference"] == 0.0
    assert compared.summary["maximum_normalized_energy_difference"] == 0.0


def test_quality_artifacts_are_manifest_validated(tmp_path: Path) -> None:
    model = build_rotating_blocks_model("quick")
    completed = _completed(model)
    quality = QUALITY.evaluate_mesh_quality(
        model,
        completed,
        _refinement(completed),
    )
    writer = BenchmarkArtifactWriter(
        tmp_path,
        "mesh-quality-test",
        seed=0,
        solver_settings={"profile": "quick"},
    )

    required = QUALITY.write_mesh_quality_artifacts(writer, tmp_path, quality)
    writer.finalize(required=required)

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    validate_benchmark_manifest(manifest, root=tmp_path)
    paths = {record["path"] for record in manifest["artifacts"]}
    assert set(required) <= paths
    assert json.loads((tmp_path / "mesh-quality.json").read_text())["passed"]


def test_quality_gate_reports_worst_element_and_parameter() -> None:
    model = build_rotating_blocks_model("quick")
    completed = _completed(model, minimum_jacobian=0.01)
    quality = QUALITY.evaluate_mesh_quality(
        model,
        completed,
        _refinement(completed),
    )
    base_row = {
        "criterion": "base",
        "category": "test",
        "observed": True,
        "relation": "==",
        "limit": True,
        "passed": True,
        "message": "base passed",
    }
    base = RotatingBlocksAcceptanceGate(
        (base_row,),
        {
            "passed": True,
            "criterion_count": 1,
            "failed_count": 0,
            "criteria": [base_row],
            "failed_criteria": [],
            "failure_messages": [],
        },
    )

    gate = QUALITY_GATE.include_mesh_quality_in_gate(base, quality)

    assert not gate.passed
    assert "mesh_minimum_jacobian" in gate.summary["failed_criteria"]
    message = "\n".join(gate.summary["failure_messages"])
    assert "element=0" in message
    assert "body=lower" in message
    assert "parameter=0.25" in message
