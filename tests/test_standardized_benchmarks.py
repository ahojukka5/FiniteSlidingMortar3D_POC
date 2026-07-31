from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

from contact3d.benchmark_artifacts import validate_benchmark_manifest


def test_standardized_benchmarks_write_valid_artifacts(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    output = tmp_path / "standardized"
    subprocess.run(
        [
            sys.executable,
            str(repository / "benchmarks" / "run_standardized.py"),
            "--output",
            str(output),
        ],
        cwd=repository,
        check=True,
    )

    summary = json.loads(
        (output / "suite-summary.json").read_text(encoding="utf-8")
    )
    expected = {
        "tet4-patch",
        "nonlinear-equilibrium",
        "coupled-mortar-patch",
        "adaptive-contact-policy",
        "mixed-load-path",
        "mixed-contact-onset",
        "scale-aware-penalty",
        "warped-nonmatching-adapter",
    }
    assert summary["benchmark_count"] == len(expected)
    assert {row["benchmark"] for row in summary["benchmarks"]} == expected

    for name in expected:
        directory = output / name
        manifest = json.loads(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        validate_benchmark_manifest(manifest, root=directory)
        assert manifest["benchmark"] == name
        assert manifest["artifacts"]

    summary_schemas = {
        "mixed-load-path": "contact3d-mixed-load-path/v1",
        "mixed-contact-onset": "contact3d-mixed-contact-onset/v1",
        "scale-aware-penalty": "contact3d-scale-aware-penalty/v1",
    }
    for name, schema in summary_schemas.items():
        payload = json.loads(
            (output / name / "summary.json").read_text(encoding="utf-8")
        )
        assert payload["schema_version"] == schema

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
    )
    for path in vtk_files:
        root = ElementTree.parse(path).getroot()
        assert root.tag == "VTKFile"
