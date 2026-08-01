#!/usr/bin/env python3
"""Generate production moving-overlap diagnostics on a warped nonmatching interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from contact3d import (
    AugmentedLagrangeState,
    ContactPair,
    ContactSurface,
    MortarContactInterface,
    build_facet_overlap,
)
from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.benchmark_plots import write_bar_chart, write_polygon_overlay


def interface() -> MortarContactInterface:
    slave_nodes = np.array(
        [
            [0.14, -0.08, -0.012],
            [1.14, 0.18, -0.006],
            [0.84, 1.18, -0.014],
            [-0.16, 0.86, -0.008],
        ]
    )
    master_nodes = np.array(
        [
            [0.00, 0.00, 0.000],
            [1.00, 0.00, 0.004],
            [1.00, 1.00, -0.002],
            [0.00, 1.00, 0.003],
        ]
    )
    slave = ContactSurface(
        slave_nodes,
        (np.array([0, 1, 2, 3], dtype=np.int64),),
        normal_sign=-1.0,
    )
    master = ContactSurface(
        master_nodes,
        (
            np.array([0, 1, 2], dtype=np.int64),
            np.array([0, 2, 3], dtype=np.int64),
        ),
    )
    pair = ContactPair(slave, master, 2400.0, 0.2, quadrature_points=7)
    return MortarContactInterface(
        pair,
        np.arange(4, dtype=np.int64),
        np.arange(4, 8, dtype=np.int64),
    )


def displacement() -> np.ndarray:
    values = np.zeros((8, 3), dtype=float)
    values[:4] = np.array(
        [
            [0.006, -0.003, -0.005],
            [0.010, 0.002, -0.004],
            [0.004, 0.007, -0.006],
            [-0.005, 0.003, -0.003],
        ]
    )
    values[4:] = np.array(
        [
            [-0.002, 0.001, 0.000],
            [0.001, -0.002, 0.001],
            [0.003, 0.002, -0.001],
            [-0.001, 0.003, 0.000],
        ]
    )
    return values


def _projected_area(polygon: np.ndarray) -> float:
    shifted = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * shifted[:, 1] - shifted[:, 0] * polygon[:, 1]
    return 0.5 * abs(float(np.sum(cross)))


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    mapped = interface()
    values = displacement()
    artifacts = BenchmarkArtifactWriter(
        output,
        "warped-nonmatching-adapter",
        seed=90317,
        solver_settings={
            "normal_penalty": 2400.0,
            "search_distance": 0.2,
            "quadrature_points": 7,
            "directional_difference_step": 2.0e-7,
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    state = AugmentedLagrangeState.zeros(4)
    base = mapped.evaluate(values.ravel(), state, tolerance=1.0e-12)
    tangent = mapped.tangent(values.ravel(), state, base, tolerance=1.0e-12)

    rng = np.random.default_rng(90317)
    direction = rng.normal(size=24)
    direction /= np.linalg.norm(direction)
    step = 2.0e-7
    plus = mapped.evaluate(values.ravel() + step * direction, state, tolerance=1.0e-12)
    minus = mapped.evaluate(values.ravel() - step * direction, state, tolerance=1.0e-12)
    numerical = (plus.residual - minus.residual) / (2.0 * step)
    analytical = tangent @ direction
    tangent_error = float(np.linalg.norm(analytical - numerical) / np.linalg.norm(numerical))

    current_slave = mapped.pair.slave.current_nodes(values[:4])
    current_master = mapped.pair.master.current_nodes(values[4:])
    polygons: list[tuple[np.ndarray, str]] = []
    vtk_polygons: list[tuple[np.ndarray, int, int]] = []
    for pair_index, (_, master_index) in enumerate(base.signature.facet_pairs):
        master_facet = mapped.pair.master.facets[master_index]
        overlap = build_facet_overlap(current_slave, current_master[master_facet])
        if pair_index == 0:
            polygons.append((overlap.slave_polygon, "slave QUAD4"))
            vtk_polygons.append((overlap.slave_polygon, 0, pair_index))
        polygons.append((overlap.master_polygon, f"master TRI3 {master_index}"))
        polygons.append((overlap.intersection_polygon, f"intersection {master_index}"))
        vtk_polygons.append((overlap.master_polygon, 1, pair_index))
        vtk_polygons.append((overlap.intersection_polygon, 2, pair_index))

    rows = [
        {
            "node": node,
            "normal_gap": float(base.normal_gaps[node]),
            "pressure": float(base.pressure[node]),
            "supported": bool(base.signature.supported_rows[node]),
            "active": bool(base.signature.active_rows[node]),
        }
        for node in range(4)
    ]
    summary = {
        "schema_version": "contact3d-warped-adapter/v1",
        "geometry": {
            "slave_kind": "warped QUAD4",
            "master_kind": "two warped TRI3 facets",
            "facet_pairs": [list(pair) for pair in base.signature.facet_pairs],
            "overlap_areas": base.raw.contact.weights.overlap_areas.tolist(),
            "total_overlap_area": base.raw.contact.weights.total_area,
        },
        "metrics": {
            "supported_rows": int(np.count_nonzero(base.signature.supported_rows)),
            "active_rows": int(np.count_nonzero(base.signature.active_rows)),
            "maximum_penetration": base.diagnostics.maximum_penetration,
            "maximum_pressure": float(np.max(base.pressure, initial=0.0)),
            "contact_force_balance_norm": float(
                np.linalg.norm(base.residual.reshape((-1, 3)).sum(axis=0))
            ),
            "directional_tangent_error": tangent_error,
        },
        "interface_state": rows,
    }
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-warped-adapter/v1",
    )
    artifacts.write_csv(
        "interface-state.csv",
        rows,
        schema="contact3d-contact-nodes/v1",
    )
    artifacts.write_surface_vtp(
        "slave-contact.vtp",
        mapped.pair.slave.reference_nodes,
        mapped.pair.slave.facets,
        values[:4],
        point_data={
            "normal_gap": base.normal_gaps,
            "pressure": base.pressure,
            "supported": np.asarray(base.signature.supported_rows, dtype=np.int64),
            "active": np.asarray(base.signature.active_rows, dtype=np.int64),
        },
    )
    master_overlap = np.zeros(len(mapped.pair.master.facets), dtype=float)
    for (_, master_index), area in zip(
        base.signature.facet_pairs,
        base.raw.contact.weights.overlap_areas,
        strict=True,
    ):
        master_overlap[master_index] += area
    artifacts.write_surface_vtp(
        "master-contact.vtp",
        mapped.pair.master.reference_nodes,
        mapped.pair.master.facets,
        values[4:],
        cell_data={"overlap_area": master_overlap},
    )
    projected_points: list[np.ndarray] = []
    projected_facets: list[np.ndarray] = []
    region_kind: list[int] = []
    pair_indices: list[int] = []
    projected_areas: list[float] = []
    for polygon, kind, pair_index in vtk_polygons:
        start = len(projected_points)
        projected_points.extend(np.column_stack([polygon, np.zeros(len(polygon))]))
        projected_facets.append(np.arange(start, start + len(polygon), dtype=np.int64))
        region_kind.append(kind)
        pair_indices.append(pair_index)
        projected_areas.append(_projected_area(polygon))
    artifacts.write_surface_vtp(
        "projected-overlap.vtp",
        np.asarray(projected_points),
        tuple(projected_facets),
        cell_data={
            "region_kind": np.asarray(region_kind, dtype=np.int64),
            "pair_index": np.asarray(pair_indices, dtype=np.int64),
            "projected_area": np.asarray(projected_areas),
        },
    )
    write_bar_chart(
        output / "interface-pressure.svg",
        title="Warped-interface pressure",
        y_label="pressure",
        labels=tuple(f"A{node}" for node in range(len(base.pressure))),
        values=base.pressure,
        annotations=tuple(
            f"p={pressure:.4g}, g={gap:.3g}"
            for gap, pressure in zip(base.normal_gaps, base.pressure, strict=True)
        ),
    )
    write_polygon_overlay(
        output / "projected-overlap.svg",
        title="Projected nonmatching overlaps",
        polygons=polygons,
        emphasized=tuple("intersection" in label for _, label in polygons),
        dashed=tuple(index > 0 for index in range(len(polygons))),
    )
    for path in output.glob("*.svg"):
        ElementTree.parse(path)
        artifacts.register(path.name, "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "interface-state.csv",
            "slave-contact.vtp",
            "master-contact.vtp",
            "projected-overlap.vtp",
            "interface-pressure.svg",
            "projected-overlap.svg",
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/warped-nonmatching-adapter"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output)["metrics"], indent=2))


if __name__ == "__main__":
    main()
