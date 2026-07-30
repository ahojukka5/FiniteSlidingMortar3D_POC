#!/usr/bin/env python3
"""Generate production moving-overlap diagnostics on a warped nonmatching interface."""

from __future__ import annotations

import argparse
import csv
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


def write_pressure(path: Path, gaps: np.ndarray, pressure: np.ndarray) -> None:
    width, height = 760, 420
    left, right, top, bottom = 72, 28, 40, 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(float(np.max(pressure, initial=0.0)), 1.0)
    bar_width = 0.5 * plot_width / len(pressure)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2}" y="24" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Warped-interface pressure</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
    ]
    for node, (gap, value) in enumerate(zip(gaps, pressure, strict=True)):
        center = left + (node + 0.5) * plot_width / len(pressure)
        bar_height = value / maximum * plot_height
        lines.extend(
            [
                (
                    f'<rect x="{center-bar_width/2:.2f}" '
                    f'y="{height-bottom-bar_height:.2f}" width="{bar_width:.2f}" '
                    f'height="{bar_height:.2f}" fill="none" stroke="black"/>'
                ),
                (
                    f'<text x="{center:.2f}" y="{height-bottom+22}" text-anchor="middle" '
                    f'font-family="monospace" font-size="12">A{node}</text>'
                ),
                (
                    f'<text x="{center:.2f}" y="{max(top+14, height-bottom-bar_height-6):.2f}" '
                    f'text-anchor="middle" font-family="monospace" font-size="10">'
                    f'p={value:.4g}, g={gap:.3g}</text>'
                ),
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overlap(path: Path, polygons: list[tuple[np.ndarray, str]]) -> None:
    width, height = 620, 620
    margin = 56
    points = np.vstack([polygon for polygon, _ in polygons])
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    span = np.maximum(upper - lower, 1.0e-12)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])

    def mapped(polygon: np.ndarray) -> str:
        transformed = np.column_stack(
            [
                margin + scale * (polygon[:, 0] - lower[0]),
                height - margin - scale * (polygon[:, 1] - lower[1]),
            ]
        )
        closed = np.vstack([transformed, transformed[0]])
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in closed)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2}" y="24" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Projected nonmatching overlaps</text>'
        ),
    ]
    for index, (polygon, label) in enumerate(polygons):
        dash = "" if index == 0 else ' stroke-dasharray="6,4"'
        lines.append(
            f'<polyline points="{mapped(polygon)}" fill="none" stroke="black" '
            f'stroke-width="{2.5 if "intersection" in label else 1.2}"{dash}/>'
        )
        lines.append(
            f'<text x="{width-24}" y="{48+18*index}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    mapped = interface()
    values = displacement()
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
    for pair_index, (_, master_index) in enumerate(base.signature.facet_pairs):
        master_facet = mapped.pair.master.facets[master_index]
        overlap = build_facet_overlap(current_slave, current_master[master_facet])
        if pair_index == 0:
            polygons.append((overlap.slave_polygon, "slave QUAD4"))
        polygons.append((overlap.master_polygon, f"master TRI3 {master_index}"))
        polygons.append((overlap.intersection, f"intersection {master_index}"))

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
    with (output / "interface-state.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
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
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_pressure(output / "interface-pressure.svg", base.normal_gaps, base.pressure)
    write_overlap(output / "projected-overlap.svg", polygons)
    for path in output.glob("*.svg"):
        ElementTree.parse(path)
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
