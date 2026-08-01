"""Versioned checked-in numeric golden specifications for benchmarks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from .benchmark_artifacts import (
    BenchmarkArtifactError,
    NumericTolerance,
    compare_numeric_metrics,
)

GOLDEN_SCHEMA_VERSION = "contact3d-golden-metrics/v1"
_ALLOWED_PROFILES = frozenset({"full", "quick"})
_ROOT_FIELDS = frozenset({"schema_version", "benchmark", "source", "profiles", "metrics"})
_METRIC_FIELDS = frozenset({"reference", "absolute", "relative"})


@dataclass(frozen=True, slots=True)
class GoldenMetric:
    """One selected dotted-path metric and its reference tolerance."""

    path: str
    reference: float
    tolerance: NumericTolerance


@dataclass(frozen=True, slots=True)
class GoldenBenchmarkSpec:
    """Checked numeric references for one benchmark output artifact."""

    benchmark: str
    source: str
    profiles: tuple[str, ...]
    metrics: tuple[GoldenMetric, ...]


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.generic),
    ):
        raise BenchmarkArtifactError(f"{field} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise BenchmarkArtifactError(f"{field} must be finite")
    return result


def _relative_path(value: object) -> str:
    if not isinstance(value, str):
        raise BenchmarkArtifactError("golden source must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise BenchmarkArtifactError("golden source must stay under benchmark output")
    return path.as_posix()


def _reject_unknown_fields(
    payload: Mapping[str, object],
    allowed: frozenset[str],
    *,
    context: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise BenchmarkArtifactError(
            f"unsupported {context} fields: " + ", ".join(unknown)
        )


def validate_golden_spec(payload: Mapping[str, object]) -> GoldenBenchmarkSpec:
    """Validate and normalize one golden metric specification."""

    _reject_unknown_fields(payload, _ROOT_FIELDS, context="golden specification")
    if payload.get("schema_version") != GOLDEN_SCHEMA_VERSION:
        raise BenchmarkArtifactError("unsupported golden metric schema version")
    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark.strip():
        raise BenchmarkArtifactError("golden benchmark must be a nonempty string")
    source = _relative_path(payload.get("source"))
    raw_profiles = payload.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise BenchmarkArtifactError("golden profiles must be a nonempty list")
    if any(not isinstance(profile, str) for profile in raw_profiles):
        raise BenchmarkArtifactError("golden profiles must contain strings")
    profiles = tuple(dict.fromkeys(raw_profiles))
    unknown = sorted(set(profiles) - _ALLOWED_PROFILES)
    if unknown:
        raise BenchmarkArtifactError(
            "unsupported golden profiles: " + ", ".join(unknown)
        )

    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, Mapping) or not raw_metrics:
        raise BenchmarkArtifactError("golden metrics must be a nonempty object")
    metrics: list[GoldenMetric] = []
    for path, raw_metric in raw_metrics.items():
        if (
            not isinstance(path, str)
            or not path
            or any(not part for part in path.split("."))
        ):
            raise BenchmarkArtifactError(
                "golden metric paths must be dotted nonempty strings"
            )
        if not isinstance(raw_metric, Mapping):
            raise BenchmarkArtifactError(f"golden metric {path!r} must be an object")
        _reject_unknown_fields(
            raw_metric,
            _METRIC_FIELDS,
            context=f"golden metric {path!r}",
        )
        reference = _finite_number(
            raw_metric.get("reference"),
            field=f"golden metric {path!r} reference",
        )
        absolute = _finite_number(
            raw_metric.get("absolute", 0.0),
            field=f"golden metric {path!r} absolute tolerance",
        )
        relative = _finite_number(
            raw_metric.get("relative", 0.0),
            field=f"golden metric {path!r} relative tolerance",
        )
        try:
            tolerance = NumericTolerance(absolute=absolute, relative=relative)
        except ValueError as error:
            raise BenchmarkArtifactError(str(error)) from error
        metrics.append(GoldenMetric(path, reference, tolerance))
    return GoldenBenchmarkSpec(
        benchmark=benchmark,
        source=source,
        profiles=profiles,
        metrics=tuple(metrics),
    )


def load_golden_spec(path: Path) -> GoldenBenchmarkSpec:
    """Load one strict JSON golden metric specification."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkArtifactError(
            f"cannot load golden specification: {path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise BenchmarkArtifactError("golden specification root must be an object")
    return validate_golden_spec(payload)


def load_golden_directory(directory: Path) -> dict[str, GoldenBenchmarkSpec]:
    """Load all JSON golden specifications and reject duplicate benchmarks."""

    root = Path(directory)
    if not root.is_dir():
        raise BenchmarkArtifactError(
            f"golden specification directory does not exist: {root}"
        )
    specifications: dict[str, GoldenBenchmarkSpec] = {}
    for path in sorted(root.glob("*.json")):
        specification = load_golden_spec(path)
        if specification.benchmark in specifications:
            raise BenchmarkArtifactError(
                f"duplicate golden specification for {specification.benchmark}"
            )
        specifications[specification.benchmark] = specification
    if not specifications:
        raise BenchmarkArtifactError(
            "golden specification directory contains no JSON files"
        )
    return specifications


def _set_reference(root: dict[str, object], path: str, value: float) -> None:
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            child: dict[str, object] = {}
            current[part] = child
            current = child
        elif isinstance(existing, dict):
            current = existing
        else:
            raise BenchmarkArtifactError(f"conflicting golden metric path: {path}")
    if parts[-1] in current:
        raise BenchmarkArtifactError(f"duplicate golden metric path: {path}")
    current[parts[-1]] = value


def evaluate_golden_spec(
    specification: GoldenBenchmarkSpec,
    benchmark_output: Path,
    *,
    profile: str,
) -> dict[str, object]:
    """Evaluate one benchmark output using its selected numeric references."""

    if profile not in _ALLOWED_PROFILES:
        raise ValueError(f"unsupported benchmark profile: {profile}")
    if profile not in specification.profiles:
        return {
            "schema_version": GOLDEN_SCHEMA_VERSION,
            "benchmark": specification.benchmark,
            "source": specification.source,
            "profile": profile,
            "status": "skipped_profile",
            "metric_count": 0,
            "metrics": {},
        }
    source = Path(benchmark_output) / Path(*PurePosixPath(specification.source).parts)
    try:
        actual = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkArtifactError(
            f"cannot load golden source for {specification.benchmark}: {source}"
        ) from error
    if not isinstance(actual, Mapping):
        raise BenchmarkArtifactError("golden source root must be an object")

    reference: dict[str, object] = {}
    tolerances: dict[str, NumericTolerance] = {}
    for metric in specification.metrics:
        _set_reference(reference, metric.path, metric.reference)
        tolerances[metric.path] = metric.tolerance
    report = compare_numeric_metrics(actual, reference, tolerances)
    return {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "benchmark": specification.benchmark,
        "source": specification.source,
        "profile": profile,
        "status": "passed",
        "metric_count": len(specification.metrics),
        "metrics": report,
    }
