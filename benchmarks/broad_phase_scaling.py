#!/usr/bin/env python3
"""Generate deterministic BVH broad-phase equivalence and scaling artifacts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median
from xml.etree import ElementTree

import numpy as np

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.benchmark_plots import write_line_chart
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


def run(
    output: Path,
    *,
    subdivisions: tuple[int, ...] = (8, 16, 24, 32),
) -> dict[str, object]:
    if len(subdivisions) < 2 or any(value <= 0 for value in subdivisions):
        raise ValueError("subdivisions must contain at least two positive values")
    if len(set(subdivisions)) != len(subdivisions):
        raise ValueError("subdivision values must be unique")

    output.mkdir(parents=True, exist_ok=True)
    levels = tuple(sorted(subdivisions))
    artifacts = BenchmarkArtifactWriter(
        output,
        "broad-phase-scaling",
        seed=0,
        solver_settings={
            "subdivisions": levels,
            "search_distance": 0.04,
            "build_repetitions": 5,
            "refit_repetitions": 9,
            "query_repetitions": 5,
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    rows = [_row(value) for value in levels]

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
    summary = {
        "schema_version": "contact3d-broad-phase-scaling/v1",
        "metrics": metrics,
        "rows": rows,
    }
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-broad-phase-scaling/v1",
    )
    artifacts.write_csv(
        "scaling.csv",
        rows,
        schema="contact3d-broad-phase-levels/v1",
    )
    write_line_chart(
        output / "operation-scaling.svg",
        title="Broad-phase operation scaling",
        x_label="slave facets",
        y_label="tested facet pairs",
        x_values=facet_counts,
        series=(
            (test_counts, "BVH facet tests"),
            (
                np.asarray([float(row["quadratic_tests"]) for row in rows]),
                "quadratic oracle",
            ),
        ),
        logarithmic_x=True,
        logarithmic_y=True,
        show_markers=True,
    )
    write_line_chart(
        output / "refit-cost.svg",
        title="Refit cost relative to topology build",
        x_label="master facets",
        y_label="refit / build time",
        x_values=np.asarray([float(row["master_facets"]) for row in rows]),
        series=(
            (
                np.asarray([float(row["refit_to_build_ratio"]) for row in rows]),
                "refit / build",
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
            "scaling.csv",
            "operation-scaling.svg",
            "refit-cost.svg",
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/broad-phase-scaling"),
    )
    parser.add_argument(
        "--subdivisions",
        type=int,
        nargs="+",
        default=[8, 16, 24, 32],
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                arguments.output,
                subdivisions=tuple(arguments.subdivisions),
            )["metrics"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
