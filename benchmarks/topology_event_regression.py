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
from contact3d.benchmark_plots import write_category_timeline, write_line_chart
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
    write_category_timeline(
        output / "event-timeline.svg",
        title="Localized topology events",
        x_label="path fraction",
        categories=tuple(str(row["kind"]) for row in rows),
        x_values=np.asarray([float(row["event_fraction"]) for row in rows]),
        groups=tuple(int(row["subdivisions"]) for row in rows),
        emphasized_group=subdivision_counts[0],
    )
    write_line_chart(
        output / "subdivision-error.svg",
        title="Subdivision invariance",
        x_label="path subdivisions",
        y_label="maximum event-location error",
        x_values=np.asarray([float(row["subdivisions"]) for row in error_rows]),
        series=(
            (
                np.asarray([float(row["maximum_error"]) for row in error_rows]),
                "maximum error",
            ),
        ),
        show_markers=True,
    )
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
