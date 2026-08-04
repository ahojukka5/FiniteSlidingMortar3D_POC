"""Command-line entry point for the nonmatching contact-patch example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .run import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the small nonmatching frictionless contact patch."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="directory for summary.json, final.vtu, and pressure.svg",
    )
    arguments = parser.parse_args()
    summary = run(arguments.output)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
