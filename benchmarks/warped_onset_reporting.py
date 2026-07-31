"""CSV and SVG writers for the warped nonmatching onset benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mesh_edges(elements: np.ndarray) -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for element in elements:
        for first in range(4):
            for second in range(first + 1, 4):
                edges.add(tuple(sorted((int(element[first]), int(element[second])))))
    return tuple(sorted(edges))


def _write_deformation(path: Path, reference: np.ndarray, current: np.ndarray, elements) -> None:
    width, height = 760, 560
    margin = 48
    points = np.vstack([reference[:, [0, 2]], current[:, [0, 2]]])
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    span = np.maximum(upper - lower, 1.0e-12)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])

    def mapped(point: np.ndarray) -> tuple[float, float]:
        return (
            margin + scale * (point[0] - lower[0]),
            height - margin - scale * (point[2] - lower[1]),
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2}" y="24" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Reference and final x-z mesh</text>'
        ),
    ]
    for coordinates, dash in ((reference, ' stroke-dasharray="5,4"'), (current, "")):
        for first, second in _mesh_edges(elements):
            x1, y1 = mapped(coordinates[first])
            x2, y2 = mapped(coordinates[second])
            lines.append(
                f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" '
                f'y2="{y2:.3f}" stroke="black" stroke-width="1"{dash}/>'
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pressure(path: Path, pressure_rows: list[dict[str, object]]) -> None:
    width, height = 760, 440
    left, right, top, bottom = 72, 28, 44, 64
    values = np.asarray([float(row["pressure"]) for row in pressure_rows])
    maximum = max(float(np.max(values, initial=0.0)), 1.0)
    count = len(values)
    step = (width - left - right) / count
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2}" y="26" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Final slave-row pressure</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
    ]
    for index, (row, value) in enumerate(zip(pressure_rows, values, strict=True)):
        center = left + (index + 0.5) * step
        bar_height = value / maximum * (height - top - bottom)
        lines.extend(
            [
                (
                    f'<rect x="{center-0.25*step:.3f}" '
                    f'y="{height-bottom-bar_height:.3f}" width="{0.5*step:.3f}" '
                    f'height="{bar_height:.3f}" fill="none" stroke="black"/>'
                ),
                (
                    f'<text x="{center:.3f}" y="{height-bottom+20}" '
                    f'text-anchor="middle" font-family="monospace" font-size="11">'
                    f'S{row["slave_row"]}</text>'
                ),
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_overlap(path: Path, polygons: list[tuple[np.ndarray, str]]) -> None:
    width, height = 620, 620
    margin = 56
    all_points = np.vstack([polygon for polygon, _ in polygons])
    lower = np.min(all_points, axis=0)
    upper = np.max(all_points, axis=0)
    span = np.maximum(upper - lower, 1.0e-12)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])

    def mapped(polygon: np.ndarray) -> str:
        result = np.column_stack(
            [
                margin + scale * (polygon[:, 0] - lower[0]),
                height - margin - scale * (polygon[:, 1] - lower[1]),
            ]
        )
        closed = np.vstack([result, result[0]])
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in closed)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2}" y="24" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Final projected overlaps</text>'
        ),
    ]
    for index, (polygon, label) in enumerate(polygons):
        dash = "" if "intersection" in label else ' stroke-dasharray="6,4"'
        width_value = 2.5 if "intersection" in label else 1.2
        lines.append(
            f'<polyline points="{mapped(polygon)}" fill="none" stroke="black" '
            f'stroke-width="{width_value}"{dash}/>'
        )
        lines.append(
            f'<text x="{width-20}" y="{48+17*index}" text-anchor="end" '
            f'font-family="sans-serif" font-size="10">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


