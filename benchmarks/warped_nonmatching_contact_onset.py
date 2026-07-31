#!/usr/bin/env python3
"""Solve a warped nonmatching finite-sliding contact-onset benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contact3d import solve_adaptive_contact_path

try:
    from .warped_onset_analysis import collect_histories
    from .warped_onset_model import BenchmarkModel, model, options
    from .warped_onset_results import write_results
except ImportError:  # Direct script execution from the repository root.
    from warped_onset_analysis import collect_histories
    from warped_onset_model import BenchmarkModel, model, options
    from warped_onset_results import write_results


__all__ = ["BenchmarkModel", "model", "options", "run"]


def run(output: Path) -> dict[str, object]:
    """Solve the path and write the complete deterministic result bundle."""

    output.mkdir(parents=True, exist_ok=True)
    benchmark = model()
    result = solve_adaptive_contact_path(
        benchmark.problem,
        1.0,
        path=benchmark.path,
        options=options(),
    )
    if not result.converged:
        raise RuntimeError(f"warped contact-onset path failed: {result.termination_reason}")
    histories = collect_histories(result)
    return write_results(output, benchmark, result, histories)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/warped-nonmatching-contact-onset"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output)["metrics"], indent=2))


if __name__ == "__main__":
    main()
