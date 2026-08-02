#!/usr/bin/env python3
"""Run the rotating-blocks deterministic repetition check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rotating_blocks_model import build_rotating_blocks_model, rotating_blocks_profile

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.repetition import RepetitionTolerances, compare_kinematic_topology_scans
from contact3d.topology_scan import scan_kinematic_contact_path

SCHEMA = "contact3d-rotating-blocks-repetition/v1"
DEFAULT_SAMPLES = {"quick": 65, "full": 129}


def run(
    output: Path,
    *,
    profile: str = "quick",
    sample_count: int | None = None,
    absolute_tolerance: float = 1.0e-12,
    relative_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    selected = rotating_blocks_profile(profile)
    count = DEFAULT_SAMPLES[selected.name] if sample_count is None else int(sample_count)
    if count < 9:
        raise ValueError("rotating-blocks repetition check requires at least nine samples")
    parameters = np.linspace(0.0, 1.0, count)

    first_model = build_rotating_blocks_model(selected)
    second_model = build_rotating_blocks_model(selected)
    first = scan_kinematic_contact_path(first_model.problem, first_model.path, parameters)
    second = scan_kinematic_contact_path(second_model.problem, second_model.path, parameters)
    comparison = compare_kinematic_topology_scans(
        first,
        second,
        tolerances=RepetitionTolerances(
            absolute=absolute_tolerance,
            relative=relative_tolerance,
        ),
    )
    summary = {
        "schema_version": SCHEMA,
        "profile": selected.name,
        "sample_count": count,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "comparison": comparison.summary(),
    }

    writer = BenchmarkArtifactWriter(
        Path(output),
        "rotating-blocks-repetition",
        seed=0,
        solver_settings={
            "profile": selected.name,
            "sample_count": count,
            "nonlinear_solver": None,
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    writer.write_json("summary.json", summary, schema=SCHEMA)
    writer.finalize(required=("summary.json",))
    if not comparison.passed:
        raise RuntimeError(json.dumps(comparison.summary(), sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(DEFAULT_SAMPLES), default="quick")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rotating-blocks-repetition"),
    )
    arguments = parser.parse_args()
    summary = run(
        arguments.output,
        profile=arguments.profile,
        sample_count=arguments.samples,
        absolute_tolerance=arguments.absolute_tolerance,
        relative_tolerance=arguments.relative_tolerance,
    )
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    main()
