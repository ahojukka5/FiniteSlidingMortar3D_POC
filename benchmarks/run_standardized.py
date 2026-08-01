#!/usr/bin/env python3
"""Run benchmarks migrated to the versioned artifact contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from contact3d.benchmark_artifacts import (
    BENCHMARK_SCHEMA_VERSION,
    validate_benchmark_manifest,
)
from contact3d.benchmark_goldens import (
    GOLDEN_SCHEMA_VERSION,
    evaluate_golden_spec,
    load_golden_directory,
)

BENCHMARKS = {
    "tet4-patch": "tet4_patch.py",
    "nonlinear-equilibrium": "nonlinear_equilibrium.py",
    "coupled-mortar-patch": "coupled_mortar_patch.py",
    "adaptive-contact-policy": "adaptive_policy_regression.py",
    "adaptive-topology-events": "adaptive_event_regression.py",
    "mixed-load-path": "mixed_path_regression.py",
    "mixed-contact-onset": "mixed_contact_onset.py",
    "scale-aware-penalty": "scale_aware_penalty_regression.py",
    "warped-nonmatching-adapter": "warped_nonmatching_adapter.py",
    "warped-nonmatching-contact-onset": "warped_nonmatching_contact_onset.py",
    "topology-events": "topology_event_regression.py",
    "broad-phase-scaling": "broad_phase_scaling.py",
    "linear-solver-scaling": "linear_solver_scaling.py",
}

QUICK_ARGUMENTS = {
    "broad-phase-scaling": (
        "--subdivisions",
        "4",
        "8",
        "12",
    ),
    "linear-solver-scaling": (
        "--levels",
        "1",
        "2",
        "--backends",
        "dense",
        "sparse_lu",
        "--minimum-free-dofs",
        "0",
    ),
}

GOLDEN_SUITE_SCHEMA_VERSION = "contact3d-golden-suite/v1"


def _not_configured_report(name: str, profile: str) -> dict[str, object]:
    return {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "benchmark": name,
        "source": None,
        "profile": profile,
        "status": "not_configured",
        "metric_count": 0,
        "metrics": {},
    }


def run(
    output: Path,
    benchmarks: tuple[str, ...] | None = None,
    *,
    quick: bool = False,
    verify_goldens: bool = True,
    golden_directory: Path | None = None,
) -> dict[str, object]:
    """Execute selected benchmarks and validate manifests and golden metrics."""

    root = Path(__file__).resolve().parent
    selected = tuple(BENCHMARKS) if benchmarks is None else benchmarks
    if not selected:
        raise ValueError("at least one benchmark must be selected")
    unknown = sorted(set(selected) - set(BENCHMARKS))
    if unknown:
        raise ValueError("unknown benchmark names: " + ", ".join(unknown))
    profile = "quick" if quick else "full"
    specifications = {}
    if verify_goldens:
        directory = root / "goldens" if golden_directory is None else golden_directory
        specifications = load_golden_directory(directory)

    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    golden_reports: list[dict[str, object]] = []
    for name in selected:
        destination = output / name
        command = [
            sys.executable,
            str(root / BENCHMARKS[name]),
            "--output",
            str(destination),
        ]
        if quick:
            command.extend(QUICK_ARGUMENTS.get(name, ()))
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"standardized benchmark {name} failed with exit code "
                f"{completed.returncode}"
            )
        manifest_path = destination / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"standardized benchmark {name} did not write a manifest"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_benchmark_manifest(manifest, root=destination)

        if not verify_goldens:
            golden_report = _not_configured_report(name, profile)
            golden_report["status"] = "disabled"
        elif name in specifications:
            golden_report = evaluate_golden_spec(
                specifications[name],
                destination,
                profile=profile,
            )
        else:
            golden_report = _not_configured_report(name, profile)
        golden_reports.append(golden_report)
        rows.append(
            {
                "benchmark": name,
                "artifact_count": len(manifest["artifacts"]),
                "git_sha": manifest["provenance"]["git_sha"],
                "seed": manifest["provenance"]["seed"],
                "manifest": str(manifest_path.relative_to(output)),
                "golden_status": golden_report["status"],
                "golden_metric_count": golden_report["metric_count"],
            }
        )

    passed_reports = [
        report for report in golden_reports if report["status"] == "passed"
    ]
    golden_summary: dict[str, object] = {
        "schema_version": GOLDEN_SUITE_SCHEMA_VERSION,
        "profile": profile,
        "verification_enabled": verify_goldens,
        "configured_benchmarks": len(specifications),
        "evaluated_benchmarks": len(passed_reports),
        "evaluated_metrics": sum(
            int(report["metric_count"]) for report in passed_reports
        ),
        "reports": golden_reports,
    }
    golden_path = output / "golden-regressions.json"
    golden_path.write_text(
        json.dumps(golden_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, object] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite": "standardized-benchmarks",
        "profile": profile,
        "benchmark_count": len(rows),
        "golden_report": golden_path.name,
        "golden_evaluated_benchmarks": len(passed_reports),
        "golden_evaluated_metrics": golden_summary["evaluated_metrics"],
        "benchmarks": rows,
    }
    (output / "suite-summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/standardized-benchmarks"),
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=tuple(BENCHMARKS),
        dest="benchmarks",
        help="run only the selected benchmark; repeat to select several",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use bounded smoke settings for expensive scaling benchmarks",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        help="read checked metric specifications from this directory",
    )
    parser.add_argument(
        "--skip-goldens",
        action="store_true",
        help="run benchmarks without checked numeric golden verification",
    )
    arguments = parser.parse_args()
    selected = None if arguments.benchmarks is None else tuple(arguments.benchmarks)
    print(
        json.dumps(
            run(
                arguments.output,
                selected,
                quick=arguments.quick,
                verify_goldens=not arguments.skip_goldens,
                golden_directory=arguments.golden_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
