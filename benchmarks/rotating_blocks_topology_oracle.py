#!/usr/bin/env python3
"""Generate the solver-independent rotating-blocks topology oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.benchmark_plots import write_category_timeline, write_line_chart
from contact3d.topology_scan import KinematicTopologyScan, scan_kinematic_contact_path
from rotating_blocks_model import build_rotating_blocks_model, rotating_blocks_profile

SUMMARY_SCHEMA = "contact3d-rotating-blocks-topology-oracle/v1"
SAMPLE_SCHEMA = "contact3d-kinematic-topology-samples/v1"
TRANSITION_SCHEMA = "contact3d-kinematic-topology-transitions/v1"
EXPECTED_SCHEMA = "contact3d-expected-topology-transitions/v1"
DEFAULT_SAMPLES = {"quick": 65, "full": 129}


def _encoded(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _sample_rows(scan: KinematicTopologyScan) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in scan.frames:
        for sample in frame.contacts:
            rows.append(
                {
                    "parameter": frame.parameter,
                    "phase": frame.phase,
                    "phase_parameter": frame.phase_parameter,
                    "interface": sample.interface,
                    "facet_pair_count": sample.facet_pair_count,
                    "overlap_area": sample.overlap_area,
                    "supported_row_count": sample.supported_row_count,
                    "active_row_count": sample.active_row_count,
                    "maximum_pressure": sample.maximum_pressure,
                    "facet_pairs": _encoded(sample.signature.facet_pairs),
                    "supported_rows": _encoded(sample.supported_rows),
                    "active_rows": _encoded(sample.active_rows),
                    "geometry_tokens": _encoded(sample.signature.geometry_tokens),
                }
            )
    return rows


def _transition_rows(scan: KinematicTopologyScan) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, transition in enumerate(scan.transitions, start=1):
        kinds = tuple(change.kind for change in transition.changes)
        rows.append(
            {
                "transition": index,
                "left_parameter": transition.left_parameter,
                "right_parameter": transition.right_parameter,
                "midpoint": transition.midpoint,
                "width": transition.width,
                "atomic_change_count": len(transition.changes),
                "pair_entries": transition.count("pair_entry"),
                "pair_exits": transition.count("pair_exit"),
                "support_activations": transition.count("support_activation"),
                "support_releases": transition.count("support_release"),
                "pressure_activations": transition.count("pressure_activation"),
                "pressure_releases": transition.count("pressure_release"),
                "clipping_transitions": transition.count("clipping_vertex_edge"),
                "pallet_transitions": transition.count("pallet_transition"),
                "event_kinds": _encoded(kinds),
                "event_interfaces": _encoded(
                    tuple(change.interface for change in transition.changes)
                ),
                "event_entities": _encoded(
                    tuple(change.entity for change in transition.changes)
                ),
            }
        )
    return rows


def _expected_transitions(
    scan: KinematicTopologyScan,
    *,
    profile: str,
) -> dict[str, object]:
    return {
        "schema_version": EXPECTED_SCHEMA,
        "profile": profile,
        "geometry_tolerance": scan.geometry_tolerance,
        "sample_count": len(scan.frames),
        "signature_digest": scan.signature_digest,
        "transitions": [
            {
                "left_parameter": transition.left_parameter,
                "right_parameter": transition.right_parameter,
                "changes": [
                    {
                        "kind": change.kind,
                        "interface": change.interface,
                        "entity": change.entity,
                        "detail": change.detail,
                    }
                    for change in transition.changes
                ],
            }
            for transition in scan.transitions
        ],
    }


def _metrics(scan: KinematicTopologyScan) -> dict[str, object]:
    samples = [sample for frame in scan.frames for sample in frame.contacts]
    pair_counts = np.asarray([sample.facet_pair_count for sample in samples])
    overlap_areas = np.asarray([sample.overlap_area for sample in samples])
    return {
        "sample_count": len(scan.frames),
        "interface_count": len(scan.frames[0].contacts),
        "transition_intervals": len(scan.transitions),
        "atomic_changes": sum(len(item.changes) for item in scan.transitions),
        "pair_entries": scan.event_count("pair_entry"),
        "pair_exits": scan.event_count("pair_exit"),
        "support_activations": scan.event_count("support_activation"),
        "support_releases": scan.event_count("support_release"),
        "pressure_activations": scan.event_count("pressure_activation"),
        "pressure_releases": scan.event_count("pressure_release"),
        "clipping_transitions": scan.event_count("clipping_vertex_edge"),
        "pallet_transitions": scan.event_count("pallet_transition"),
        "minimum_facet_pairs": int(np.min(pair_counts)),
        "maximum_facet_pairs": int(np.max(pair_counts)),
        "minimum_overlap_area": float(np.min(overlap_areas)),
        "maximum_overlap_area": float(np.max(overlap_areas)),
        "signature_digest": scan.signature_digest,
    }


def _write_plots(output: Path, scan: KinematicTopologyScan) -> tuple[str, ...]:
    if len(scan.frames[0].contacts) != 1:
        raise ValueError("rotating-blocks topology plots require exactly one interface")
    parameters = np.asarray([frame.parameter for frame in scan.frames])
    samples = [frame.contacts[0] for frame in scan.frames]
    write_line_chart(
        output / "overlap-area.svg",
        title="Rotating-blocks projected overlap",
        x_label="absolute continuation parameter",
        y_label="overlap area",
        x_values=parameters,
        series=((np.asarray([sample.overlap_area for sample in samples]), "area"),),
        show_markers=True,
    )
    write_line_chart(
        output / "topology-counts.svg",
        title="Rotating-blocks contact topology counts",
        x_label="absolute continuation parameter",
        y_label="count",
        x_values=parameters,
        series=(
            (np.asarray([sample.facet_pair_count for sample in samples]), "facet pairs"),
            (np.asarray([sample.supported_row_count for sample in samples]), "support rows"),
            (np.asarray([sample.active_row_count for sample in samples]), "active rows"),
        ),
        show_markers=True,
    )
    categories = [
        change.kind
        for transition in scan.transitions
        for change in transition.changes
    ]
    locations = [
        transition.midpoint
        for transition in scan.transitions
        for _ in transition.changes
    ]
    write_category_timeline(
        output / "transition-timeline.svg",
        title="Rotating-blocks topology transition brackets",
        x_label="absolute continuation parameter",
        categories=categories,
        x_values=np.asarray(locations),
    )
    for path in output.glob("*.svg"):
        ElementTree.parse(path)
    return ("overlap-area.svg", "topology-counts.svg", "transition-timeline.svg")


def run(
    output: Path,
    *,
    profile: str = "quick",
    sample_count: int | None = None,
    geometry_tolerance: float = 1.0e-12,
) -> dict[str, object]:
    selected = rotating_blocks_profile(profile)
    count = DEFAULT_SAMPLES[selected.name] if sample_count is None else int(sample_count)
    if count < 9:
        raise ValueError("rotating-blocks topology scan requires at least nine samples")
    model = build_rotating_blocks_model(selected)
    parameters = np.linspace(0.0, model.path.end_parameter, count)
    scan = scan_kinematic_contact_path(
        model.problem,
        model.path,
        parameters,
        geometry_tolerance=geometry_tolerance,
    )
    metrics = _metrics(scan)
    if metrics["pair_entries"] < 2 or metrics["pair_exits"] < 2:
        raise RuntimeError(
            "rotating-blocks geometry did not produce repeated pair entry and exit events"
        )

    output = Path(output)
    artifacts = BenchmarkArtifactWriter(
        output,
        "rotating-blocks-topology-oracle",
        seed=0,
        solver_settings={
            "profile": selected,
            "sample_count": count,
            "geometry_tolerance": geometry_tolerance,
            "nonlinear_solver": None,
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "profile": selected.name,
        "geometry": model.geometry,
        "metrics": metrics,
    }
    artifacts.write_json("summary.json", summary, schema=SUMMARY_SCHEMA)
    artifacts.write_json(
        "expected-transitions.json",
        _expected_transitions(scan, profile=selected.name),
        schema=EXPECTED_SCHEMA,
    )
    artifacts.write_csv(
        "sample-history.csv",
        _sample_rows(scan),
        schema=SAMPLE_SCHEMA,
    )
    artifacts.write_csv(
        "transition-history.csv",
        _transition_rows(scan),
        schema=TRANSITION_SCHEMA,
    )
    plot_names = _write_plots(output, scan)
    for name in plot_names:
        artifacts.register(name, "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "expected-transitions.json",
            "sample-history.csv",
            "transition-history.csv",
            *plot_names,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(DEFAULT_SAMPLES), default="quick")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--geometry-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rotating-blocks-topology-oracle"),
    )
    arguments = parser.parse_args()
    summary = run(
        arguments.output,
        profile=arguments.profile,
        sample_count=arguments.samples,
        geometry_tolerance=arguments.geometry_tolerance,
    )
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
