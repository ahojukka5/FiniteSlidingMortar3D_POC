"""Dependency-free deterministic SVG helpers for benchmark artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Protocol

import numpy as np


class _SparsityPattern(Protocol):
    shape: tuple[int, int]
    indptr: np.ndarray
    indices: np.ndarray


class _ProblemWithSparsity(Protocol):
    sparsity: _SparsityPattern


def _vector(name: str, values: object, *, positive: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    return array


def _coordinate(value: float, lower: float, upper: float, start: float, stop: float) -> float:
    if np.isclose(lower, upper):
        return 0.5 * (start + stop)
    return start + (value - lower) / (upper - lower) * (stop - start)


def _write(path: Path, lines: Sequence[str]) -> None:
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_line_chart(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: object,
    series: Sequence[tuple[object, str]],
    logarithmic_x: bool = False,
    logarithmic_y: bool = False,
    show_markers: bool = False,
) -> None:
    """Write a deterministic multi-series SVG line chart."""

    x = _vector("x_values", x_values, positive=logarithmic_x)
    if not series:
        raise ValueError("series must contain at least one data series")
    normalized: list[tuple[np.ndarray, str]] = []
    for index, (values, label) in enumerate(series):
        array = _vector(f"series[{index}]", values, positive=logarithmic_y)
        if len(array) != len(x):
            raise ValueError("every series must have the same length as x_values")
        normalized.append((array, str(label)))

    width, height = 760, 460
    left, right, top, bottom = 82, 28, 38, 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    tx = np.log10(x) if logarithmic_x else x
    transformed = [
        (np.log10(values) if logarithmic_y else values, label)
        for values, label in normalized
    ]
    x_min, x_max = float(np.min(tx)), float(np.max(tx))
    y_min = float(min(np.min(values) for values, _ in transformed))
    y_max = float(max(np.max(values) for values, _ in transformed))
    padding = 0.08 * max(1.0e-12, y_max - y_min)
    y_min -= padding
    y_max += padding

    def sx(value: float) -> float:
        return _coordinate(value, x_min, x_max, left, width - right)

    def sy(value: float) -> float:
        return _coordinate(value, y_min, y_max, height - bottom, top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" '
            f'font-family="sans-serif" font-size="16">{escape(title)}</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
        (
            f'<text x="{left + plot_width / 2:.1f}" y="{height-18}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="13">'
            f'{escape(x_label)}</text>'
        ),
        (
            f'<text x="18" y="{top + plot_height / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 18 {top + plot_height / 2:.1f})" '
            f'font-family="sans-serif" font-size="13">{escape(y_label)}</text>'
        ),
    ]
    for tick in np.linspace(x_min, x_max, 6):
        position = sx(float(tick))
        label = f"1e{tick:.0f}" if logarithmic_x else f"{tick:.3g}"
        lines.append(
            f'<text x="{position:.2f}" y="{height-bottom+22}" text-anchor="middle" '
            f'font-family="monospace" font-size="11">{label}</text>'
        )
    for tick in np.linspace(y_min, y_max, 7):
        position = sy(float(tick))
        label = f"1e{tick:.0f}" if logarithmic_y else f"{tick:.3g}"
        lines.extend(
            [
                (
                    f'<line x1="{left}" y1="{position:.2f}" x2="{width-right}" '
                    f'y2="{position:.2f}" stroke="#dddddd"/>'
                ),
                (
                    f'<text x="{left-10}" y="{position+4:.2f}" text-anchor="end" '
                    f'font-family="monospace" font-size="11">{label}</text>'
                ),
            ]
        )
    dash_patterns = ("", ' stroke-dasharray="6,4"', ' stroke-dasharray="2,4"')
    for index, (values, label) in enumerate(transformed):
        points = [
            (sx(float(x_value)), sy(float(y_value)))
            for x_value, y_value in zip(tx, values, strict=True)
        ]
        encoded = " ".join(f"{px:.2f},{py:.2f}" for px, py in points)
        dash = dash_patterns[index % len(dash_patterns)]
        lines.append(
            f'<polyline points="{encoded}" fill="none" stroke="black" '
            f'stroke-width="2"{dash}/>'
        )
        if show_markers:
            for px, py in points:
                lines.append(
                    f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3" '
                    'fill="white" stroke="black"/>'
                )
        lines.append(
            f'<text x="{width-right-8}" y="{top+18+18*index}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{escape(label)}</text>'
        )
    lines.append("</svg>")
    _write(Path(path), lines)


def write_bar_chart(
    path: Path,
    *,
    title: str,
    y_label: str,
    labels: Sequence[str],
    values: object,
    annotations: Sequence[str] | None = None,
) -> None:
    """Write a deterministic nonnegative scalar bar chart."""

    array = _vector("values", values)
    if np.any(array < 0.0):
        raise ValueError("bar-chart values must be nonnegative")
    if len(labels) != len(array):
        raise ValueError("labels and values must have the same length")
    if annotations is not None and len(annotations) != len(array):
        raise ValueError("annotations and values must have the same length")

    width, height = 760, 440
    left, right, top, bottom = 82, 28, 44, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(float(np.max(array, initial=0.0)), np.finfo(float).eps)
    step = plot_width / len(array)
    bar_width = 0.55 * step
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2:.1f}" y="26" text-anchor="middle" '
            f'font-family="sans-serif" font-size="16">{escape(title)}</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
        (
            f'<text x="18" y="{top + plot_height / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 18 {top + plot_height / 2:.1f})" '
            f'font-family="sans-serif" font-size="13">{escape(y_label)}</text>'
        ),
    ]
    for index, value in enumerate(array):
        center = left + (index + 0.5) * step
        bar_height = float(value) / maximum * plot_height
        y = height - bottom - bar_height
        lines.extend(
            [
                (
                    f'<rect x="{center-bar_width/2:.2f}" y="{y:.2f}" '
                    f'width="{bar_width:.2f}" height="{bar_height:.2f}" '
                    'fill="none" stroke="black"/>'
                ),
                (
                    f'<text x="{center:.2f}" y="{height-bottom+22}" '
                    f'text-anchor="middle" font-family="monospace" font-size="10">'
                    f'{escape(str(labels[index]))}</text>'
                ),
            ]
        )
        if annotations is not None:
            label_y = max(top + 13, y - 6)
            lines.append(
                f'<text x="{center:.2f}" y="{label_y:.2f}" text-anchor="middle" '
                f'font-family="monospace" font-size="9">'
                f'{escape(str(annotations[index]))}</text>'
            )
    lines.append("</svg>")
    _write(Path(path), lines)


def write_category_timeline(
    path: Path,
    *,
    title: str,
    x_label: str,
    categories: Sequence[str],
    x_values: object,
    groups: Sequence[object] | None = None,
    emphasized_group: object | None = None,
) -> None:
    """Write categorical event locations on a common numeric axis."""

    x = _vector("x_values", x_values)
    if len(categories) != len(x):
        raise ValueError("categories and x_values must have the same length")
    if groups is not None and len(groups) != len(x):
        raise ValueError("groups and x_values must have the same length")
    ordered = sorted({str(category) for category in categories})
    if not ordered:
        raise ValueError("categories must not be empty")

    width = 820
    height = max(300, 110 + 38 * len(ordered))
    left, right, top, bottom = 150, 32, 48, 58
    x_min, x_max = float(np.min(x)), float(np.max(x))
    padding = 0.04 * max(1.0e-12, x_max - x_min)
    x_min -= padding
    x_max += padding
    y_lookup = {
        category: top + index * (height - top - bottom) / max(1, len(ordered) - 1)
        for index, category in enumerate(ordered)
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2:.1f}" y="26" text-anchor="middle" '
            f'font-family="sans-serif" font-size="16">{escape(title)}</text>'
        ),
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
        (
            f'<text x="{left+(width-left-right)/2:.1f}" y="{height-16}" '
            f'text-anchor="middle" font-family="sans-serif" font-size="13">'
            f'{escape(x_label)}</text>'
        ),
    ]
    for category, y in y_lookup.items():
        lines.extend(
            [
                (
                    f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
                    f'y2="{y:.2f}" stroke="#dddddd"/>'
                ),
                (
                    f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" '
                    f'font-family="monospace" font-size="10">'
                    f'{escape(category)}</text>'
                ),
            ]
        )
    for index, (category, value) in enumerate(zip(categories, x, strict=True)):
        px = _coordinate(float(value), x_min, x_max, left, width - right)
        py = y_lookup[str(category)]
        emphasized = groups is not None and groups[index] == emphasized_group
        radius = 5 if emphasized else 3
        lines.append(
            f'<circle cx="{px:.3f}" cy="{py:.3f}" r="{radius}" '
            'fill="white" stroke="black"/>'
        )
    lines.append("</svg>")
    _write(Path(path), lines)


def write_polygon_overlay(
    path: Path,
    *,
    title: str,
    polygons: Sequence[tuple[object, str]],
    emphasized: Sequence[bool] | None = None,
    dashed: Sequence[bool] | None = None,
) -> None:
    """Write projected two-dimensional polygons with deterministic styling."""

    if not polygons:
        raise ValueError("polygons must contain at least one polygon")
    normalized: list[tuple[np.ndarray, str]] = []
    for index, (points, label) in enumerate(polygons):
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
            raise ValueError(f"polygon[{index}] must have shape (point_count, 2)")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"polygon[{index}] must contain only finite values")
        normalized.append((array, str(label)))
    if emphasized is not None and len(emphasized) != len(normalized):
        raise ValueError("emphasized and polygons must have the same length")
    if dashed is not None and len(dashed) != len(normalized):
        raise ValueError("dashed and polygons must have the same length")

    width, height = 620, 620
    margin = 56
    all_points = np.vstack([points for points, _ in normalized])
    lower = np.min(all_points, axis=0)
    upper = np.max(all_points, axis=0)
    span = np.maximum(upper - lower, 1.0e-12)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])

    def mapped(points: np.ndarray) -> str:
        transformed = np.column_stack(
            [
                margin + scale * (points[:, 0] - lower[0]),
                height - margin - scale * (points[:, 1] - lower[1]),
            ]
        )
        closed = np.vstack([transformed, transformed[0]])
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in closed)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2:.1f}" y="24" text-anchor="middle" '
            f'font-family="sans-serif" font-size="16">{escape(title)}</text>'
        ),
    ]
    for index, (points, label) in enumerate(normalized):
        strong = emphasized[index] if emphasized is not None else False
        use_dash = dashed[index] if dashed is not None else index > 0
        dash = ' stroke-dasharray="6,4"' if use_dash else ""
        stroke_width = 2.5 if strong else 1.2
        lines.extend(
            [
                (
                    f'<polyline points="{mapped(points)}" fill="none" stroke="black" '
                    f'stroke-width="{stroke_width}"{dash}/>'
                ),
                (
                    f'<text x="{width-20}" y="{48+17*index}" text-anchor="end" '
                    f'font-family="sans-serif" font-size="10">{escape(label)}</text>'
                ),
            ]
        )
    lines.append("</svg>")
    _write(Path(path), lines)


def write_mesh_projection_overlay(
    path: Path,
    *,
    title: str,
    reference_nodes: object,
    current_nodes: object,
    elements: object,
    axes: tuple[int, int] = (0, 2),
) -> None:
    """Write reference and current TET4 mesh edges in a selected projection."""

    reference = np.asarray(reference_nodes, dtype=float)
    current = np.asarray(current_nodes, dtype=float)
    cells = np.asarray(elements, dtype=np.int64)
    if reference.ndim != 2 or reference.shape[1] != 3 or current.shape != reference.shape:
        raise ValueError("reference_nodes and current_nodes must have shape (node_count, 3)")
    if cells.ndim != 2 or cells.shape[1] != 4:
        raise ValueError("elements must have shape (element_count, 4)")
    if any(axis not in (0, 1, 2) for axis in axes) or axes[0] == axes[1]:
        raise ValueError("axes must select two distinct coordinate components")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(current)):
        raise ValueError("mesh coordinates must be finite")
    if np.any(cells < 0) or np.any(cells >= len(reference)):
        raise ValueError("elements contain an invalid node index")

    edges: set[tuple[int, int]] = set()
    for cell in cells:
        for first in range(4):
            for second in range(first + 1, 4):
                edges.add(tuple(sorted((int(cell[first]), int(cell[second])))))
    width, height = 760, 560
    margin = 48
    projected = np.vstack([reference[:, axes], current[:, axes]])
    lower = np.min(projected, axis=0)
    upper = np.max(projected, axis=0)
    span = np.maximum(upper - lower, 1.0e-12)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])

    def mapped(point: np.ndarray) -> tuple[float, float]:
        return (
            margin + scale * (point[axes[0]] - lower[0]),
            height - margin - scale * (point[axes[1]] - lower[1]),
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2:.1f}" y="24" text-anchor="middle" '
            f'font-family="sans-serif" font-size="16">{escape(title)}</text>'
        ),
    ]
    for coordinates, dash in ((reference, ' stroke-dasharray="5,4"'), (current, "")):
        for first, second in sorted(edges):
            x1, y1 = mapped(coordinates[first])
            x2, y2 = mapped(coordinates[second])
            lines.append(
                f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" '
                f'y2="{y2:.3f}" stroke="black" stroke-width="1"{dash}/>'
            )
    lines.append("</svg>")
    _write(Path(path), lines)


def write_sparsity(problem: _ProblemWithSparsity, path: Path) -> None:
    """Write a compact deterministic CSR sparsity plot."""

    size, margin = 540, 48
    matrix_size = problem.sparsity.shape[0]
    if matrix_size <= 0 or problem.sparsity.shape[1] != matrix_size:
        raise ValueError("sparsity pattern must be a nonempty square matrix")
    plot = size - 2 * margin
    cell = plot / matrix_size
    commands: list[str] = []
    for row in range(matrix_size):
        start = int(problem.sparsity.indptr[row])
        stop = int(problem.sparsity.indptr[row + 1])
        columns = problem.sparsity.indices[start:stop]
        if len(columns) == 0:
            continue
        run_start = run_stop = int(columns[0])
        for value in columns[1:]:
            column = int(value)
            if column == run_stop + 1:
                run_stop = column
                continue
            x = margin + run_start * cell
            y = margin + (row + 0.5) * cell
            commands.append(f"M{x:.2f},{y:.2f}h{(run_stop-run_start+1)*cell:.2f}")
            run_start = run_stop = column
        x = margin + run_start * cell
        y = margin + (row + 0.5) * cell
        commands.append(f"M{x:.2f},{y:.2f}h{(run_stop-run_start+1)*cell:.2f}")
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="270" y="24" text-anchor="middle" font-family="sans-serif" '
        'font-size="16">TET4 tangent sparsity pattern</text>',
        (
            f'<rect x="{margin}" y="{margin}" width="{plot}" height="{plot}" '
            'fill="none" stroke="black"/>'
        ),
        (
            f'<path d="{" ".join(commands)}" fill="none" stroke="black" '
            f'stroke-width="{0.72*cell:.2f}" stroke-linecap="butt"/>'
        ),
        "</svg>",
    ]
    _write(Path(path), lines)
