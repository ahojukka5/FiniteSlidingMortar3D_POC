#!/usr/bin/env python3
"""Generate deterministic contact-topology event localization artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.topology_events import (
    ContactTopologyStateMachine,
    TopologyObservation,
)


@dataclass(frozen=True, slots=True)
class Signature:
    facet_pairs: tuple[tuple[int, int], ...]
    active_rows: tuple[bool, ...]
    supported_rows: tuple[bool, ...]
    geometry_tokens: tuple[tuple[int, int, int, int, int], ...] = ()


_EVENT_LOCATIONS = (0.2, 0.4, 0.515, 0.6, 0.85)


def _signature(parameter: float) -> tuple[Signature, ...]:
    pairs = () if parameter < 0.2 or parameter >= 0.85 else ((0, 0),)
    supported = (0.4 <= parameter < 0.85,)
    active = (0.6 <= parameter < 0.85,)
    if not pairs:
        geometry = ()
    elif parameter < 0.515:
        geometry = ((0, 0, 4, 4, 1),)
    else:
        geometry = ((0, 0, 5, 5, 1),)
    return (Signature(pairs, active, supported, geometry),)


def _observe(parameter: float) -> TopologyObservation:
    return TopologyObservation.valid(parameter, _signature(parameter), parameter)


def _segments(subdivisions: int) -> list[tuple[float, float]]:
    points = np.linspace(0.0, 1.0, subdivisions + 1)
    return list(zip(points[:-1], points[1:], strict=True))


def _crossings(subdivisions: int) -> list[dict[str, object]]:
    machine = ContactTopologyStateMachine()
    rows: list[dict[str, object]] = []
    for start, stop in _segments(subdivisions):
        left = _observe(float(start))
        right = _observe(float(stop))
        while left.signatures != right.signatures:
            batch = machine.localize(left, right, _observe).restarted()
            for event in batch.events:
                rows.append(
                    {
                        "subdivisions": subdivisions,
                        "segment_start": start,
                        "segment_stop": stop,
                        "kind": event.kind,
                        "interface": event.interface,
                        "entity": ":".join(str(value) for value in event.entity),
                        "left_fraction": batch.left_fraction,
                        "event_fraction": batch.event_fraction,
                        "right_fraction": batch.right_fraction,
                        "selected_fraction": batch.selected_fraction,
                        "selected_branch": batch.selected_branch,
                    }
                )
            if batch.selected_fraction <= left.fraction:
                raise RuntimeError("event localization did not advance the segment")
            left = batch.selected
    return rows


def _write_timeline(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 820, 360
    left, right, top, bottom = 72, 32, 48, 54
    kinds = sorted({str(row["kind"]) for row in rows})
    y_lookup = {kind: top + index * 42 for index, kind in enumerate(kinds)}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2}" y="26" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Localized topology events</text>'
        ),
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
    ]
    for kind, y in y_lookup.items():
        lines.append(
            f'<text x="{left-8}" y="{y+4}" text-anchor="end" '
            f'font-family="monospace" font-size="10">{kind}</text>'
        )
    for row in rows:
        x = left + float(row["event_fraction"]) * (width - left - right)
        y = y_lookup[str(row["kind"])]
        radius = 5 if int(row["subdivisions"]) == 5 else 3
        lines.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius}" '
            'fill="none" stroke="black"/>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_error(path: Path, errors: list[dict[str, object]]) -> None:
    width, height = 760, 420
    left, right, top, bottom = 76, 30, 44, 62
    maximum = max((float(row["maximum_error"]) for row in errors), default=1.0e-16)
    maximum = max(maximum, 1.0e-16)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2}" y="26" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Subdivision invariance</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
    ]
    for index, row in enumerate(errors):
        subdivisions = int(row["subdivisions"])
        error = float(row["maximum_error"])
        x = left + (index + 1) * (width - left - right) / (len(errors) + 1)
        y = height - bottom - error / maximum * (height - top - bottom)
        lines.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" fill="black"/>')
        lines.append(
            f'<text x="{x:.3f}" y="{height-bottom+20}" text-anchor="middle" '
            f'font-family="monospace" font-size="10">{subdivisions}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    subdivision_counts = (5, 10, 20, 40)
    artifacts = BenchmarkArtifactWriter(
        output,
        "topology-events",
        seed=0,
        solver_settings={
            "subdivision_counts": subdivision_counts,
            "expected_locations": _EVENT_LOCATIONS,
            "selected_branch": "right",
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    rows = [row for count in subdivision_counts for row in _crossings(count)]

    reference = {
        str(row["kind"]): float(row["event_fraction"])
        for row in rows
        if int(row["subdivisions"]) == subdivision_counts[-1]
    }
    error_rows: list[dict[str, object]] = []
    for count in subdivision_counts:
        values = {
            str(row["kind"]): float(row["event_fraction"])
            for row in rows
            if int(row["subdivisions"]) == count
        }
        shared = sorted(set(reference) & set(values))
        error = max(
            (abs(values[kind] - reference[kind]) for kind in shared),
            default=0.0,
        )
        error_rows.append(
            {
                "subdivisions": count,
                "shared_event_kinds": len(shared),
                "maximum_error": error,
            }
        )

    metrics = {
        "subdivision_counts": list(subdivision_counts),
        "event_rows": len(rows),
        "maximum_subdivision_error": max(
            float(row["maximum_error"]) for row in error_rows
        ),
        "branch_selection": "right",
        "localized_event_kinds": sorted({str(row["kind"]) for row in rows}),
    }
    summary = {
        "schema_version": "contact3d-topology-events/v1",
        "metrics": metrics,
        "expected_locations": list(_EVENT_LOCATIONS),
    }
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-topology-events/v1",
    )
    artifacts.write_csv(
        "event-history.csv",
        rows,
        schema="contact3d-topology-event-history/v1",
    )
    artifacts.write_csv(
        "subdivision-errors.csv",
        error_rows,
        schema="contact3d-topology-subdivision-errors/v1",
    )
    _write_timeline(output / "event-timeline.svg", rows)
    _write_error(output / "subdivision-error.svg", error_rows)
    for path in output.glob("*.svg"):
        ElementTree.parse(path)
        artifacts.register(path.name, "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "event-history.csv",
            "subdivision-errors.csv",
            "event-timeline.svg",
            "subdivision-error.svg",
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/topology-events"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output)["metrics"], indent=2))


if __name__ == "__main__":
    main()
