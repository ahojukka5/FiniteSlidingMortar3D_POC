"""Command-line entry point for the rotating-blocks example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the coarse frictionless rotating-blocks example."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="directory for summary, VTK, and SVG results",
    )
    arguments = parser.parse_args()
    summary = run(arguments.output)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
