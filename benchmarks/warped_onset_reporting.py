"""SVG reporting adapters for the warped nonmatching onset benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from contact3d.benchmark_plots import (
    write_bar_chart,
    write_mesh_projection_overlay,
    write_polygon_overlay,
)


def _write_deformation(
    path: Path,
    reference: np.ndarray,
    current: np.ndarray,
    elements: object,
) -> None:
    write_mesh_projection_overlay(
        path,
        title="Reference and final x-z mesh",
        reference_nodes=reference,
        current_nodes=current,
        elements=elements,
        axes=(0, 2),
    )


def _write_pressure(path: Path, pressure_rows: list[dict[str, object]]) -> None:
    write_bar_chart(
        path,
        title="Final slave-row pressure",
        y_label="pressure",
        labels=tuple(f'S{row["slave_row"]}' for row in pressure_rows),
        values=np.asarray([float(row["pressure"]) for row in pressure_rows]),
    )


def _write_overlap(path: Path, polygons: list[tuple[np.ndarray, str]]) -> None:
    write_polygon_overlay(
        path,
        title="Final projected overlaps",
        polygons=polygons,
        emphasized=tuple("intersection" in label for _, label in polygons),
        dashed=tuple("intersection" not in label for _, label in polygons),
    )
