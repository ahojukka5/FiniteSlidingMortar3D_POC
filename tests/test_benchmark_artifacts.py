from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from contact3d.benchmark_artifacts import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkArtifactError,
    BenchmarkArtifactWriter,
    NumericTolerance,
    compare_numeric_metrics,
    validate_benchmark_manifest,
)


def _tet4() -> tuple[np.ndarray, np.ndarray]:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return nodes, np.array([[0, 1, 2, 3]], dtype=np.int64)


def test_writer_finalizes_versioned_manifest(tmp_path: Path) -> None:
    writer = BenchmarkArtifactWriter(
        tmp_path,
        "unit-contract",
        seed=4102,
        solver_settings={"newton": {"maximum_iterations": 12}},
        repo_root=tmp_path,
    )
    writer.write_json(
        "summary.json",
        {"converged": True, "residual": 1.0e-12},
        schema="contact3d-summary/v1",
    )
    writer.write_csv(
        "iterations.csv",
        [
            {"iteration": 1, "residual": 1.0e-3},
            {"iteration": 2, "residual": 1.0e-8},
        ],
        schema="contact3d-newton-iterations/v1",
    )

    manifest = writer.finalize(required=("summary.json", "iterations.csv"))
    stored = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest == stored
    assert stored["schema_version"] == BENCHMARK_SCHEMA_VERSION
    assert stored["benchmark"] == "unit-contract"
    assert stored["provenance"]["seed"] == 4102
    assert [item["path"] for item in stored["artifacts"]] == [
        "iterations.csv",
        "summary.json",
    ]
    validate_benchmark_manifest(stored, root=tmp_path)


def test_manifest_validator_rejects_missing_provenance() -> None:
    with pytest.raises(BenchmarkArtifactError, match="missing fields"):
        validate_benchmark_manifest(
            {
                "schema_version": BENCHMARK_SCHEMA_VERSION,
                "benchmark": "broken",
                "provenance": {"benchmark": "broken"},
                "artifacts": [],
            }
        )


def test_numeric_regression_uses_explicit_tolerances() -> None:
    actual = {"response": {"force": 10.0004, "pressure": 2.0}}
    reference = {"response": {"force": 10.0, "pressure": 2.0}}
    report = compare_numeric_metrics(
        actual,
        reference,
        {"response.force": NumericTolerance(absolute=5.0e-4)},
    )

    assert report["response.force"]["absolute_error"] == pytest.approx(4.0e-4)
    with pytest.raises(BenchmarkArtifactError, match="pressure"):
        compare_numeric_metrics(
            {"response": {"pressure": 2.1}},
            reference,
            {"response.pressure": NumericTolerance(relative=1.0e-3)},
        )


def test_writer_exports_parseable_tet4_vtu(tmp_path: Path) -> None:
    nodes, elements = _tet4()
    displacement = 0.1 * nodes
    writer = BenchmarkArtifactWriter(
        tmp_path,
        "tet4-vtu",
        seed=0,
        solver_settings={},
        repo_root=tmp_path,
    )
    writer.write_tet4_vtu(
        "deformed.vtu",
        nodes,
        elements,
        displacement,
        point_data={"reaction": np.ones((4, 3))},
        cell_data={"jacobian": np.array([1.331])},
    )
    writer.finalize(required=("deformed.vtu",))

    root = ElementTree.parse(tmp_path / "deformed.vtu").getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    assert piece is not None
    assert piece.attrib["NumberOfPoints"] == "4"
    assert piece.attrib["NumberOfCells"] == "1"
    names = {
        element.attrib.get("Name")
        for element in root.findall(".//DataArray")
        if "Name" in element.attrib
    }
    assert {"reference_coordinates", "displacement", "reaction", "jacobian"} <= names
    cell_types = root.find(".//Cells/DataArray[@Name='types']")
    assert cell_types is not None and cell_types.text.strip() == "10"


def test_writer_exports_contact_surface_fields(tmp_path: Path) -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    writer = BenchmarkArtifactWriter(
        tmp_path,
        "surface-vtp",
        seed=0,
        solver_settings={},
        repo_root=tmp_path,
    )
    writer.write_surface_vtp(
        "contact.vtp",
        points,
        (np.array([0, 1, 2]), np.array([0, 2, 3])),
        point_data={
            "pressure": np.array([1.0, 2.0, 3.0, 4.0]),
            "active": np.array([0, 1, 1, 0]),
        },
        cell_data={"overlap_area": np.array([0.5, 0.5])},
    )
    writer.finalize(required=("contact.vtp",))

    root = ElementTree.parse(tmp_path / "contact.vtp").getroot()
    piece = root.find("./PolyData/Piece")
    assert piece is not None
    assert piece.attrib["NumberOfPolys"] == "2"
    names = {
        element.attrib.get("Name")
        for element in root.findall(".//DataArray")
        if "Name" in element.attrib
    }
    assert {"pressure", "active", "overlap_area"} <= names
