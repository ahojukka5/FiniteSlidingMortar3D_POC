"""Versioned benchmark artifacts, provenance, validation, and VTK output."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from xml.etree import ElementTree

import numpy as np

BENCHMARK_SCHEMA_VERSION = "contact3d-benchmark/v1"
ArtifactKind = Literal["json", "csv", "vtu", "vtp", "svg", "other"]


class BenchmarkArtifactError(ValueError):
    """Raised when a benchmark artifact violates the versioned contract."""


@dataclass(frozen=True, slots=True)
class NumericTolerance:
    """Absolute and relative tolerance for one golden regression metric."""

    absolute: float = 0.0
    relative: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.absolute) or self.absolute < 0.0:
            raise ValueError("absolute tolerance must be finite and nonnegative")
        if not np.isfinite(self.relative) or self.relative < 0.0:
            raise ValueError("relative tolerance must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One file registered in a benchmark manifest."""

    path: str
    kind: ArtifactKind
    schema: str | None = None

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"path": self.path, "kind": self.kind}
        if self.schema is not None:
            result["schema"] = self.schema
        return result


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _git_sha(repo_root: Path | None) -> str:
    environment_sha = os.environ.get("GITHUB_SHA")
    if environment_sha:
        return environment_sha
    root = Path.cwd() if repo_root is None else Path(repo_root)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = result.stdout.strip()
    return sha if sha else "unknown"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def benchmark_provenance(
    benchmark: str,
    *,
    seed: int,
    solver_settings: Mapping[str, object] | object,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Collect deterministic runtime and solver metadata for one benchmark run."""

    if not benchmark.strip():
        raise ValueError("benchmark name must be nonempty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return {
        "benchmark": benchmark,
        "git_sha": _git_sha(repo_root),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "contact3d": _package_version("contact3d"),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
        },
        "seed": seed,
        "solver_settings": _json_value(solver_settings),
    }


def _relative_artifact_path(path: str) -> PurePosixPath:
    value = PurePosixPath(path)
    if value.is_absolute() or not value.parts or ".." in value.parts:
        raise BenchmarkArtifactError("artifact paths must be relative and stay under output")
    return value


def validate_benchmark_manifest(
    manifest: Mapping[str, object],
    *,
    root: Path | None = None,
) -> None:
    """Validate a v1 benchmark manifest and optionally verify registered files."""

    if manifest.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkArtifactError("unsupported benchmark schema version")
    benchmark = manifest.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise BenchmarkArtifactError("manifest benchmark must be a nonempty string")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise BenchmarkArtifactError("manifest provenance must be an object")
    required_provenance = {
        "benchmark",
        "git_sha",
        "python",
        "platform",
        "packages",
        "seed",
        "solver_settings",
    }
    missing = sorted(required_provenance - set(provenance))
    if missing:
        raise BenchmarkArtifactError(
            f"manifest provenance is missing fields: {', '.join(missing)}"
        )
    if provenance.get("benchmark") != benchmark:
        raise BenchmarkArtifactError("manifest and provenance benchmark names differ")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise BenchmarkArtifactError("manifest artifacts must be a list")
    seen: set[str] = set()
    valid_kinds = {"json", "csv", "vtu", "vtp", "svg", "other"}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise BenchmarkArtifactError("each artifact record must be an object")
        path = artifact.get("path")
        kind = artifact.get("kind")
        if not isinstance(path, str):
            raise BenchmarkArtifactError("artifact path must be a string")
        relative = _relative_artifact_path(path)
        normalized = relative.as_posix()
        if normalized in seen:
            raise BenchmarkArtifactError(f"duplicate artifact path: {normalized}")
        seen.add(normalized)
        if kind not in valid_kinds:
            raise BenchmarkArtifactError(f"unsupported artifact kind: {kind}")
        schema = artifact.get("schema")
        if schema is not None and not isinstance(schema, str):
            raise BenchmarkArtifactError("artifact schema must be a string when present")
        if root is not None and not (Path(root) / Path(*relative.parts)).is_file():
            raise BenchmarkArtifactError(f"registered artifact does not exist: {normalized}")
    try:
        json.dumps(_json_value(manifest), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise BenchmarkArtifactError("manifest is not strict JSON") from error


def _format_ascii(values: np.ndarray) -> str:
    flat = np.asarray(values).reshape(-1)
    if np.issubdtype(flat.dtype, np.integer) or flat.dtype == np.bool_:
        return " ".join(str(int(value)) for value in flat)
    return " ".join(f"{float(value):.17g}" for value in flat)


def _validated_data_array(
    name: str,
    values: object,
    count: int,
) -> tuple[np.ndarray, int]:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape((-1, 1))
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] == 0:
        raise BenchmarkArtifactError(
            f"field {name!r} must have shape ({count},) or ({count}, components)"
        )
    if not (
        np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)
    ):
        raise BenchmarkArtifactError(f"field {name!r} must be numeric or boolean")
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise BenchmarkArtifactError(f"field {name!r} must be finite")
    return array, array.shape[1]


def _append_data_arrays(
    parent: ElementTree.Element,
    fields: Mapping[str, object],
    count: int,
) -> None:
    for name, values in fields.items():
        array, components = _validated_data_array(name, values, count)
        integer = np.issubdtype(array.dtype, np.integer) or array.dtype == np.bool_
        element = ElementTree.SubElement(
            parent,
            "DataArray",
            type="Int64" if integer else "Float64",
            Name=name,
            NumberOfComponents=str(components),
            format="ascii",
        )
        element.text = _format_ascii(array)


def _write_xml(path: Path, root: ElementTree.Element) -> None:
    ElementTree.indent(root, space="  ")
    ElementTree.ElementTree(root).write(
        path,
        encoding="utf-8",
        xml_declaration=True,
    )


def write_tet4_vtu(
    path: Path,
    reference_nodes: object,
    elements: object,
    displacement: object,
    *,
    point_data: Mapping[str, object] | None = None,
    cell_data: Mapping[str, object] | None = None,
) -> None:
    """Write a deformed TET4 mesh as an ASCII VTK unstructured grid."""

    reference = np.asarray(reference_nodes, dtype=float)
    cells = np.asarray(elements, dtype=np.int64)
    values = np.asarray(displacement, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) == 0:
        raise BenchmarkArtifactError("reference_nodes must have shape (node_count, 3)")
    if cells.ndim != 2 or cells.shape[1] != 4 or len(cells) == 0:
        raise BenchmarkArtifactError("TET4 elements must have shape (element_count, 4)")
    if values.shape == (3 * len(reference),):
        values = values.reshape((-1, 3))
    if values.shape != reference.shape:
        raise BenchmarkArtifactError("displacement must match reference_nodes")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(values)):
        raise BenchmarkArtifactError("mesh coordinates and displacement must be finite")
    if np.any(cells < 0) or np.any(cells >= len(reference)):
        raise BenchmarkArtifactError("TET4 connectivity contains an invalid node index")
    current = reference + values
    vtk = ElementTree.Element(
        "VTKFile",
        type="UnstructuredGrid",
        version="1.0",
        byte_order="LittleEndian",
    )
    grid = ElementTree.SubElement(vtk, "UnstructuredGrid")
    piece = ElementTree.SubElement(
        grid,
        "Piece",
        NumberOfPoints=str(len(reference)),
        NumberOfCells=str(len(cells)),
    )
    point_fields = {
        "reference_coordinates": reference,
        "displacement": values,
        **({} if point_data is None else dict(point_data)),
    }
    points_data = ElementTree.SubElement(piece, "PointData")
    _append_data_arrays(points_data, point_fields, len(reference))
    cells_data = ElementTree.SubElement(piece, "CellData")
    _append_data_arrays(cells_data, {} if cell_data is None else cell_data, len(cells))
    points = ElementTree.SubElement(piece, "Points")
    coordinates = ElementTree.SubElement(
        points,
        "DataArray",
        type="Float64",
        NumberOfComponents="3",
        format="ascii",
    )
    coordinates.text = _format_ascii(current)
    cell_section = ElementTree.SubElement(piece, "Cells")
    connectivity = ElementTree.SubElement(
        cell_section,
        "DataArray",
        type="Int64",
        Name="connectivity",
        format="ascii",
    )
    connectivity.text = _format_ascii(cells)
    offsets = ElementTree.SubElement(
        cell_section,
        "DataArray",
        type="Int64",
        Name="offsets",
        format="ascii",
    )
    offsets.text = _format_ascii(4 * np.arange(1, len(cells) + 1, dtype=np.int64))
    cell_types = ElementTree.SubElement(
        cell_section,
        "DataArray",
        type="UInt8",
        Name="types",
        format="ascii",
    )
    cell_types.text = _format_ascii(np.full(len(cells), 10, dtype=np.uint8))
    _write_xml(Path(path), vtk)


def write_surface_vtp(
    path: Path,
    reference_nodes: object,
    facets: Sequence[object],
    displacement: object | None = None,
    *,
    point_data: Mapping[str, object] | None = None,
    cell_data: Mapping[str, object] | None = None,
) -> None:
    """Write TRI3/QUAD4 or general polygon contact fields as VTK PolyData."""

    reference = np.asarray(reference_nodes, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) == 0:
        raise BenchmarkArtifactError("reference_nodes must have shape (node_count, 3)")
    if displacement is None:
        values = np.zeros_like(reference)
    else:
        values = np.asarray(displacement, dtype=float)
        if values.shape == (3 * len(reference),):
            values = values.reshape((-1, 3))
        if values.shape != reference.shape:
            raise BenchmarkArtifactError("surface displacement must match reference_nodes")
    normalized: list[np.ndarray] = []
    for facet in facets:
        indices = np.asarray(facet, dtype=np.int64)
        if indices.ndim != 1 or len(indices) < 3:
            raise BenchmarkArtifactError("surface facets must contain at least three nodes")
        if np.any(indices < 0) or np.any(indices >= len(reference)):
            raise BenchmarkArtifactError("surface facet contains an invalid node index")
        normalized.append(indices)
    if not normalized:
        raise BenchmarkArtifactError("surface must contain at least one facet")
    current = reference + values
    vtk = ElementTree.Element(
        "VTKFile",
        type="PolyData",
        version="1.0",
        byte_order="LittleEndian",
    )
    data = ElementTree.SubElement(vtk, "PolyData")
    piece = ElementTree.SubElement(
        data,
        "Piece",
        NumberOfPoints=str(len(reference)),
        NumberOfVerts="0",
        NumberOfLines="0",
        NumberOfStrips="0",
        NumberOfPolys=str(len(normalized)),
    )
    point_fields = {
        "reference_coordinates": reference,
        "displacement": values,
        **({} if point_data is None else dict(point_data)),
    }
    point_section = ElementTree.SubElement(piece, "PointData")
    _append_data_arrays(point_section, point_fields, len(reference))
    cell_section = ElementTree.SubElement(piece, "CellData")
    _append_data_arrays(
        cell_section,
        {} if cell_data is None else cell_data,
        len(normalized),
    )
    points = ElementTree.SubElement(piece, "Points")
    coordinates = ElementTree.SubElement(
        points,
        "DataArray",
        type="Float64",
        NumberOfComponents="3",
        format="ascii",
    )
    coordinates.text = _format_ascii(current)
    polygons = ElementTree.SubElement(piece, "Polys")
    connectivity = ElementTree.SubElement(
        polygons,
        "DataArray",
        type="Int64",
        Name="connectivity",
        format="ascii",
    )
    connectivity.text = _format_ascii(np.concatenate(normalized))
    offsets = ElementTree.SubElement(
        polygons,
        "DataArray",
        type="Int64",
        Name="offsets",
        format="ascii",
    )
    offsets.text = _format_ascii(np.cumsum([len(facet) for facet in normalized]))
    _write_xml(Path(path), vtk)


def _metric_at_path(values: Mapping[str, object], path: str) -> float:
    current: object = values
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise BenchmarkArtifactError(f"missing numeric metric: {path}")
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float, np.generic)):
        raise BenchmarkArtifactError(f"metric is not numeric: {path}")
    result = float(current)
    if not np.isfinite(result):
        raise BenchmarkArtifactError(f"metric is not finite: {path}")
    return result


def compare_numeric_metrics(
    actual: Mapping[str, object],
    reference: Mapping[str, object],
    tolerances: Mapping[str, NumericTolerance],
) -> dict[str, dict[str, float]]:
    """Compare selected dotted-path metrics using explicit numeric tolerances."""

    if not tolerances:
        raise ValueError("at least one numeric tolerance is required")
    report: dict[str, dict[str, float]] = {}
    failures: list[str] = []
    for path, tolerance in tolerances.items():
        observed = _metric_at_path(actual, path)
        expected = _metric_at_path(reference, path)
        error = abs(observed - expected)
        allowed = tolerance.absolute + tolerance.relative * abs(expected)
        report[path] = {
            "actual": observed,
            "reference": expected,
            "absolute_error": error,
            "allowed_error": allowed,
        }
        if error > allowed:
            failures.append(path)
    if failures:
        raise BenchmarkArtifactError(
            "numeric regression exceeded tolerance: " + ", ".join(failures)
        )
    return report


class BenchmarkArtifactWriter:
    """Write one benchmark directory and finalize a validated manifest."""

    def __init__(
        self,
        output: Path,
        benchmark: str,
        *,
        seed: int,
        solver_settings: Mapping[str, object] | object,
        repo_root: Path | None = None,
    ) -> None:
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.benchmark = benchmark
        self.provenance = benchmark_provenance(
            benchmark,
            seed=seed,
            solver_settings=solver_settings,
            repo_root=repo_root,
        )
        self._records: dict[str, ArtifactRecord] = {}

    def _path(self, relative_path: str) -> Path:
        relative = _relative_artifact_path(relative_path)
        return self.output / Path(*relative.parts)

    def register(
        self,
        relative_path: str,
        kind: ArtifactKind,
        *,
        schema: str | None = None,
    ) -> Path:
        path = self._path(relative_path)
        if not path.is_file():
            raise BenchmarkArtifactError(f"cannot register missing artifact: {relative_path}")
        normalized = PurePosixPath(relative_path).as_posix()
        record = ArtifactRecord(normalized, kind, schema)
        previous = self._records.get(normalized)
        if previous is not None and previous != record:
            raise BenchmarkArtifactError(
                f"artifact already registered with different metadata: {normalized}"
            )
        self._records[normalized] = record
        return path

    def write_json(
        self,
        relative_path: str,
        payload: Mapping[str, object] | Sequence[object],
        *,
        schema: str | None = None,
    ) -> Path:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_json_value(payload), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return self.register(relative_path, "json", schema=schema)

    def write_csv(
        self,
        relative_path: str,
        rows: Iterable[Mapping[str, object]],
        *,
        schema: str | None = None,
    ) -> Path:
        values = [dict(row) for row in rows]
        if not values:
            raise BenchmarkArtifactError("CSV output requires at least one row")
        fields = list(values[0])
        if not fields or any(list(row) != fields for row in values):
            raise BenchmarkArtifactError(
                "CSV rows must have the same nonempty field ordering"
            )
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in values:
                writer.writerow({key: _json_value(value) for key, value in row.items()})
        return self.register(relative_path, "csv", schema=schema)

    def write_tet4_vtu(
        self,
        relative_path: str,
        reference_nodes: object,
        elements: object,
        displacement: object,
        *,
        point_data: Mapping[str, object] | None = None,
        cell_data: Mapping[str, object] | None = None,
    ) -> Path:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_tet4_vtu(
            path,
            reference_nodes,
            elements,
            displacement,
            point_data=point_data,
            cell_data=cell_data,
        )
        return self.register(relative_path, "vtu")

    def write_surface_vtp(
        self,
        relative_path: str,
        reference_nodes: object,
        facets: Sequence[object],
        displacement: object | None = None,
        *,
        point_data: Mapping[str, object] | None = None,
        cell_data: Mapping[str, object] | None = None,
    ) -> Path:
        path = self._path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_surface_vtp(
            path,
            reference_nodes,
            facets,
            displacement,
            point_data=point_data,
            cell_data=cell_data,
        )
        return self.register(relative_path, "vtp")

    def finalize(self, *, required: Iterable[str] = ()) -> dict[str, object]:
        missing = sorted(
            PurePosixPath(path).as_posix()
            for path in required
            if PurePosixPath(path).as_posix() not in self._records
        )
        if missing:
            raise BenchmarkArtifactError(
                "benchmark is missing required artifacts: " + ", ".join(missing)
            )
        manifest: dict[str, object] = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "benchmark": self.benchmark,
            "provenance": self.provenance,
            "artifacts": [
                self._records[path].as_dict() for path in sorted(self._records)
            ],
        }
        validate_benchmark_manifest(manifest, root=self.output)
        path = self.output / "manifest.json"
        path.write_text(
            json.dumps(_json_value(manifest), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest
