#!/usr/bin/env python3
"""Generate deterministic BVH broad-phase equivalence and scaling artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from statistics import median
from xml.etree import ElementTree

import numpy as np

from contact3d.broad_phase import FacetAABBTree, facet_aabbs
from contact3d.surface import (
    ContactSurface,
    discover_facet_pairs_brute_force,
    discover_facet_pairs_with_diagnostics,
)


def _quad_grid(
    subdivisions: int,
    *,
    z: float,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    nodes = np.array(
        [
            [i / subdivisions, j / subdivisions, z]
            for j in range(subdivisions + 1)
            for i in range(subdivisions + 1)
        ],
        dtype=float,
    )
    facets = []
    for j in range(subdivisions):
        for i in range(subdivisions):
            first = j * (subdivisions + 1) + i
            facets.append(
                np.array(
                    [
                        first,
                        first + 1,
                        first + subdivisions + 2,
                        first + subdivisions + 1,
                    ],
                    dtype=np.int64,
                )
            )
    return nodes, tuple(facets)


def _current_coordinates(nodes: np.ndarray, *, phase: float) -> np.ndarray:
    current = nodes.copy()
    x = current[:, 0]
    y = current[:, 1]
    current[:, 2] += 0.005 * np.sin(2.0 * np.pi * x + phase) * np.sin(
        2.0 * np.pi * y
    )
    current[:, 0] += 0.003 * np.sin(np.pi * y + phase)
    return current


def _timed(callable_, *, repetitions: int) -> float:
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - start)
    return median(samples)


def _row(subdivisions: int) -> dict[str, object]:
    slave_reference, slave_facets = _quad_grid(subdivisions, z=0.0)
    master_reference, master_facets = _quad_grid(subdivisions, z=-0.02)
    slave = ContactSurface(slave_reference, slave_facets)
    master = ContactSurface(master_reference, master_facets)
    slave_current = _current_coordinates(slave_reference, phase=0.0)
    master_current = _current_coordinates(master_reference, phase=0.37)
    search_distance = 0.04

    result = discover_facet_pairs_with_diagnostics(
        slave,
        master,
        slave_current,
        master_current,
        search_distance=search_distance,
    )
    oracle = discover_facet_pairs_brute_force(
        slave,
        master,
        slave_current,
        master_current,
        search_distance=search_distance,
    )
    slave_minimums, slave_maximums = facet_aabbs(slave_current, slave.facets)

    build_seconds = _timed(
        lambda: FacetAABBTree.build(master_reference, master_facets),
        repetitions=5,
    )
    refit_seconds = _timed(
        lambda: master.broad_phase_tree.refit(master_current),
        repetitions=9,
    )
    fitted = master.broad_phase_tree.refit(master_current)
    query_seconds = _timed(
        lambda: fitted.query(
            slave_minimums,
            slave_maximums,
            search_distance=search_distance,
        ),
        repetitions=5,
    )

    diagnostics = result.diagnostics
    return {
        "subdivisions": subdivisions,
        "slave_facets": diagnostics.slave_facet_count,
        "master_facets": diagnostics.master_facet_count,
        "accepted_pairs": diagnostics.accepted_pairs,
        "tree_nodes": diagnostics.tree_node_count,
        "node_visits": diagnostics.node_visits,
        "facet_tests": diagnostics.facet_tests,
        "quadratic_tests": diagnostics.brute_force_tests,
        "tested_fraction": diagnostics.tested_fraction,
        "build_seconds": build_seconds,
        "refit_seconds": refit_seconds,
        "query_seconds": query_seconds,
        "refit_to_build_ratio": refit_seconds / build_seconds,
        "pair_sets_equal": result.pairs == oracle,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scale(value: float, lower: float, upper: float, start: float, stop: float) -> float:
    if upper == lower:
        return 0.5 * (start + stop)
    return start + (value - lower) / (upper - lower) * (stop - start)


def _write_operation_plot(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 820, 480
    left, right, top, bottom = 76, 32, 46, 62
    x_values = np.log10([float(row["slave_facets"]) for row in rows])
    bvh_values = np.log10([float(row["facet_tests"]) for row in rows])
    brute_values = np.log10([float(row["quadratic_tests"]) for row in rows])
    ymin = float(min(np.min(bvh_values), np.min(brute_values)))
    ymax = float(max(np.max(bvh_values), np.max(brute_values)))

    def point(x: float, y: float) -> tuple[float, float]:
        return (
            _scale(x, float(np.min(x_values)), float(np.max(x_values)), left, width - right),
            _scale(y, ymin, ymax, height - bottom, top),
        )

    bvh_points = [point(float(x), float(y)) for x, y in zip(x_values, bvh_values, strict=True)]
    brute_points = [
        point(float(x), float(y)) for x, y in zip(x_values, brute_values, strict=True)
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="26" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Broad-phase operation scaling</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
        (
            f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in bvh_points)}" '
            'fill="none" stroke="black" stroke-width="2"/>'
        ),
        (
            f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in brute_points)}" '
            'fill="none" stroke="black" stroke-width="2" stroke-dasharray="7 5"/>'
        ),
        '<text x="100" y="70" font-family="sans-serif" font-size="12">BVH facet tests</text>',
        (
            '<text x="100" y="88" font-family="sans-serif" font-size="12">'
            'quadratic oracle (dashed)</text>'
        ),
    ]
    for x, y in bvh_points:
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="black"/>')
    for x, y in brute_points:
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="white" stroke="black"/>')
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_refit_plot(path: Path, rows: list[dict[str, object]]) -> None:
    width, height = 820, 420
    left, right, top, bottom = 76, 32, 46, 62
    ratios = [float(row["refit_to_build_ratio"]) for row in rows]
    maximum = max(1.0, max(ratios))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="26" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Refit cost relative to topology build</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
    ]
    for index, (row, ratio) in enumerate(zip(rows, ratios, strict=True)):
        x = left + (index + 1) * (width - left - right) / (len(rows) + 1)
        y = height - bottom - ratio / maximum * (height - top - bottom)
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="black"/>')
        lines.append(
            f'<text x="{x:.2f}" y="{height-bottom+20}" text-anchor="middle" '
            f'font-family="monospace" font-size="10">{row["master_facets"]}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    rows = [_row(value) for value in (8, 16, 24, 32)]
    _write_csv(output / "scaling.csv", rows)

    facet_counts = np.asarray([float(row["slave_facets"]) for row in rows])
    test_counts = np.asarray([float(row["facet_tests"]) for row in rows])
    exponent = float(np.polyfit(np.log(facet_counts), np.log(test_counts), 1)[0])
    metrics = {
        "all_pair_sets_equal": all(bool(row["pair_sets_equal"]) for row in rows),
        "facet_counts": [int(row["slave_facets"]) for row in rows],
        "largest_tested_fraction": max(float(row["tested_fraction"]) for row in rows),
        "smallest_tested_fraction": min(float(row["tested_fraction"]) for row in rows),
        "facet_test_growth_exponent": exponent,
        "largest_refit_to_build_ratio": max(
            float(row["refit_to_build_ratio"]) for row in rows
        ),
    }
    summary = {"metrics": metrics, "rows": rows}
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_operation_plot(output / "operation-scaling.svg", rows)
    _write_refit_plot(output / "refit-cost.svg", rows)
    for path in output.glob("*.svg"):
        ElementTree.parse(path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/broad-phase-scaling"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output)["metrics"], indent=2))


if __name__ == "__main__":
    main()
