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

BENCHMARKS = {
    "tet4-patch": "tet4_patch.py",
    "nonlinear-equilibrium": "nonlinear_equilibrium.py",
    "coupled-mortar-patch": "coupled_mortar_patch.py",
    "adaptive-contact-policy": "adaptive_policy_regression.py",
    "mixed-load-path": "mixed_path_regression.py",
    "mixed-contact-onset": "mixed_contact_onset.py",
    "scale-aware-penalty": "scale_aware_penalty_regression.py",
    "warped-nonmatching-adapter": "warped_nonmatching_adapter.py",
}


def run(
    output: Path,
    benchmarks: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Execute selected standardized benchmarks and validate their manifests."""

    root = Path(__file__).resolve().parent
    selected = tuple(BENCHMARKS) if benchmarks is None else benchmarks
    if not selected:
        raise ValueError("at least one benchmark must be selected")
    unknown = sorted(set(selected) - set(BENCHMARKS))
    if unknown:
        raise ValueError("unknown benchmark names: " + ", ".join(unknown))
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name in selected:
        destination = output / name
        command = [
            sys.executable,
            str(root / BENCHMARKS[name]),
            "--output",
            str(destination),
        ]
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
        rows.append(
            {
                "benchmark": name,
                "artifact_count": len(manifest["artifacts"]),
                "git_sha": manifest["provenance"]["git_sha"],
                "seed": manifest["provenance"]["seed"],
                "manifest": str(manifest_path.relative_to(output)),
            }
        )
    summary: dict[str, object] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "suite": "standardized-benchmarks",
        "benchmark_count": len(rows),
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
    arguments = parser.parse_args()
    selected = None if arguments.benchmarks is None else tuple(arguments.benchmarks)
    print(json.dumps(run(arguments.output, selected), indent=2))


if __name__ == "__main__":
    main()
