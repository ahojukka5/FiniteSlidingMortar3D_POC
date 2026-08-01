from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from contact3d.benchmark_plots import (
    write_bar_chart,
    write_category_timeline,
    write_line_chart,
    write_mesh_projection_overlay,
    write_polygon_overlay,
)


def _assert_svg(path: Path) -> None:
    root = ElementTree.parse(path).getroot()
    assert root.tag.endswith("svg")


def test_shared_plot_helpers_write_parseable_svg(tmp_path: Path) -> None:
    write_line_chart(
        tmp_path / "line.svg",
        title="Line <chart>",
        x_label="x",
        y_label="y",
        x_values=np.array([1.0, 2.0, 4.0]),
        series=((np.array([1.0, 4.0, 16.0]), "quadratic & finite"),),
        logarithmic_x=True,
        logarithmic_y=True,
        show_markers=True,
    )
    write_bar_chart(
        tmp_path / "bars.svg",
        title="Bars",
        y_label="pressure",
        labels=("A0", "A1"),
        values=np.array([2.0, 3.0]),
        annotations=("g=-0.1", "g=-0.2"),
    )
    write_category_timeline(
        tmp_path / "events.svg",
        title="Events",
        x_label="path fraction",
        categories=("pair", "pressure", "pair"),
        x_values=np.array([0.2, 0.6, 0.8]),
        groups=(5, 10, 5),
        emphasized_group=5,
    )
    write_polygon_overlay(
        tmp_path / "polygons.svg",
        title="Overlap",
        polygons=(
            (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), "slave"),
            (np.array([[0.1, 0.1], [0.8, 0.1], [0.1, 0.8]]), "intersection"),
        ),
        emphasized=(False, True),
        dashed=(True, False),
    )
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    write_mesh_projection_overlay(
        tmp_path / "mesh.svg",
        title="Reference and current",
        reference_nodes=nodes,
        current_nodes=1.1 * nodes,
        elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
    )

    for name in ("line.svg", "bars.svg", "events.svg", "polygons.svg", "mesh.svg"):
        _assert_svg(tmp_path / name)
    assert "&lt;chart&gt;" in (tmp_path / "line.svg").read_text(encoding="utf-8")
    assert "quadratic &amp; finite" in (tmp_path / "line.svg").read_text(
        encoding="utf-8"
    )


def test_shared_plot_helpers_reject_invalid_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same length"):
        write_line_chart(
            tmp_path / "bad-line.svg",
            title="bad",
            x_label="x",
            y_label="y",
            x_values=np.array([1.0, 2.0]),
            series=((np.array([1.0]), "short"),),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        write_bar_chart(
            tmp_path / "bad-bars.svg",
            title="bad",
            y_label="y",
            labels=("negative",),
            values=np.array([-1.0]),
        )
    with pytest.raises(ValueError, match="finite"):
        write_category_timeline(
            tmp_path / "bad-events.svg",
            title="bad",
            x_label="x",
            categories=("event",),
            x_values=np.array([np.nan]),
        )
    with pytest.raises(ValueError, match="shape"):
        write_polygon_overlay(
            tmp_path / "bad-polygon.svg",
            title="bad",
            polygons=((np.array([[0.0, 0.0], [1.0, 0.0]]), "line"),),
        )
