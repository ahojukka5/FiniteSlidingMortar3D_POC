"""Dependency-free SVG helpers for nonlinear equilibrium benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from contact3d import EquilibriumProblem


def write_line_chart(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: np.ndarray,
    series: tuple[tuple[np.ndarray, str], ...],
    logarithmic_y: bool = False,
) -> None:
    width, height = 760, 460
    left, right, top, bottom = 82, 28, 38, 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    transformed = tuple(np.log10(values) if logarithmic_y else values for values, _ in series)
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min = float(min(np.min(values) for values in transformed))
    y_max = float(max(np.max(values) for values in transformed))
    padding = 0.08 * max(1.0e-12, y_max - y_min)
    y_min -= padding
    y_max += padding

    def sx(value: float) -> float:
        return left + (value - x_min) / max(1.0e-12, x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / max(1.0e-12, y_max - y_min) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width/2:.1f}" y="24" text-anchor="middle" '
            f'font-family="sans-serif" font-size="16">{title}</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
        (
            f'<text x="{left+plot_width/2:.1f}" y="{height-18}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13">{x_label}</text>'
        ),
        (
            f'<text x="18" y="{top+plot_height/2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 18 {top+plot_height/2:.1f})" '
            f'font-family="sans-serif" font-size="13">{y_label}</text>'
        ),
    ]
    for tick in np.linspace(x_min, x_max, 6):
        x = sx(float(tick))
        lines.append(
            f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" '
            f'font-family="monospace" font-size="11">{tick:.2g}</text>'
        )
    for tick in np.linspace(y_min, y_max, 7):
        y = sy(float(tick))
        label = f"1e{tick:.0f}" if logarithmic_y else f"{tick:.3f}"
        lines.extend(
            [
                (
                    f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
                    f'y2="{y:.2f}" stroke="#dddddd"/>'
                ),
                (
                    f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
                    f'font-family="monospace" font-size="11">{label}</text>'
                ),
            ]
        )
    for index, ((_, label), values) in enumerate(zip(series, transformed, strict=True)):
        points = " ".join(
            f"{sx(float(x)):.2f},{sy(float(y)):.2f}"
            for x, y in zip(x_values, values, strict=True)
        )
        dash = "" if index == 0 else ' stroke-dasharray="6,4"'
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="black" '
            f'stroke-width="2"{dash}/>'
        )
        lines.append(
            f'<text x="{width-right-8}" y="{top+18+18*index}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sparsity(problem: EquilibriumProblem, path: Path) -> None:
    size, margin = 540, 48
    matrix_size = problem.sparsity.shape[0]
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

