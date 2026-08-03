"""VTK and SVG writers for the rotating-blocks result bundle."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from rotating_blocks_bundle_data import (
    Checkpoint,
    bulk_fields,
    checkpoint_rows,
    contact,
    global_contact_force,
    multiplier,
    projected_regions,
)
from rotating_blocks_model import RotatingBlocksModel
from rotating_blocks_solver import RotatingBlocksSolverRun

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.benchmark_plots import (
    write_category_timeline,
    write_line_chart,
    write_polygon_overlay,
)


def write_checkpoint(
    writer: BenchmarkArtifactWriter,
    model: RotatingBlocksModel,
    checkpoint: Checkpoint,
    ordinal: int,
) -> tuple[
    tuple[str, ...],
    tuple[dict[str, object], ...],
    tuple[tuple[np.ndarray, str], ...],
]:
    """Write one volume, two contact surfaces, and projected overlap."""

    prefix = f"checkpoints/{ordinal:02d}-{checkpoint.name}"
    interface = checkpoint.path_state.problem.interfaces[0]
    evaluation = contact(checkpoint)
    displacement = checkpoint.displacement.reshape((-1, 3))
    local_force = np.asarray(evaluation.residual, dtype=float).reshape((-1, 3))
    effective = np.asarray(
        getattr(
            checkpoint.path_state,
            "effective_force",
            np.zeros_like(checkpoint.reaction),
        ),
        dtype=float,
    ).reshape((-1, 3))
    paths = (
        f"{prefix}/volume.vtu",
        f"{prefix}/slave-contact.vtp",
        f"{prefix}/master-contact.vtp",
        f"{prefix}/projected-overlap.vtp",
    )
    writer.write_tet4_vtu(
        paths[0],
        checkpoint.path_state.problem.mesh.reference_nodes,
        checkpoint.path_state.problem.mesh.elements,
        checkpoint.displacement,
        point_data={
            "reaction": checkpoint.reaction.reshape((-1, 3)),
            "external_load": effective,
            "contact_force": global_contact_force(checkpoint).reshape((-1, 3)),
        },
        cell_data=bulk_fields(checkpoint),
    )
    writer.write_surface_vtp(
        paths[1],
        interface.pair.slave.reference_nodes,
        interface.pair.slave.facets,
        displacement[model.slave_nodes],
        point_data={
            "normal_gap": evaluation.normal_gaps,
            "pressure": evaluation.pressure,
            "multiplier": multiplier(checkpoint),
            "supported": np.asarray(
                evaluation.signature.supported_rows,
                dtype=np.int64,
            ),
            "active": np.asarray(
                evaluation.signature.active_rows,
                dtype=np.int64,
            ),
            "contact_force": local_force[: len(evaluation.pressure)],
        },
    )
    overlap = np.zeros(len(interface.pair.master.facets), dtype=float)
    for (_, master_index), area in zip(
        evaluation.signature.facet_pairs,
        evaluation.raw.contact.weights.overlap_areas,
        strict=True,
    ):
        overlap[master_index] += float(area)
    writer.write_surface_vtp(
        paths[2],
        interface.pair.master.reference_nodes,
        interface.pair.master.facets,
        displacement[model.master_nodes],
        point_data={"contact_force": local_force[len(evaluation.pressure) :]},
        cell_data={"overlap_area": overlap},
    )
    points, facets, rows, overlays = projected_regions(model, checkpoint)
    writer.write_surface_vtp(
        paths[3],
        points,
        facets,
        cell_data={
            "region_kind": np.asarray([row["region_kind"] for row in rows]),
            "pair_index": np.asarray([row["pair"] for row in rows]),
            "slave_facet": np.asarray([row["slave_facet"] for row in rows]),
            "master_facet": np.asarray([row["master_facet"] for row in rows]),
            "projected_area": np.asarray([row["projected_area"] for row in rows]),
        },
    )
    return paths, rows, overlays


def write_plots(
    writer: BenchmarkArtifactWriter,
    output: Path,
    completed: RotatingBlocksSolverRun,
    checkpoints: Sequence[Checkpoint],
    final_overlays: Sequence[tuple[np.ndarray, str]],
) -> tuple[str, ...]:
    """Write deterministic response, redistribution, event, and overlap plots."""

    (output / "plots").mkdir(parents=True, exist_ok=True)
    accepted = completed.accepted_rows
    parameters = np.asarray([row["parameter"] for row in accepted])
    paths = (
        "plots/overlap-area.svg",
        "plots/maximum-pressure.svg",
        "plots/controlled-reactions.svg",
        "plots/deformation.svg",
        "plots/pressure-redistribution.svg",
        "plots/event-locations.svg",
        "plots/final-projected-overlap.svg",
    )
    write_line_chart(
        output / paths[0],
        title="Rotating-blocks overlap area",
        x_label="continuation parameter",
        y_label="area",
        x_values=parameters,
        series=((np.asarray([row["overlap_area"] for row in accepted]), "overlap"),),
        show_markers=True,
    )
    write_line_chart(
        output / paths[1],
        title="Rotating-blocks maximum pressure",
        x_label="continuation parameter",
        y_label="pressure",
        x_values=parameters,
        series=(
            (np.asarray([row["maximum_pressure"] for row in accepted]), "pressure"),
        ),
        show_markers=True,
    )
    write_line_chart(
        output / paths[2],
        title="Rotating-blocks controlled reactions",
        x_label="continuation parameter",
        y_label="reaction",
        x_values=parameters,
        series=tuple(
            (np.asarray([row[field] for row in accepted]), field)
            for field in ("reaction_x", "reaction_y", "reaction_z")
        ),
        show_markers=True,
    )
    summaries = checkpoint_rows(checkpoints)
    write_line_chart(
        output / paths[3],
        title="Rotating-blocks deformation checkpoints",
        x_label="continuation parameter",
        y_label="maximum displacement",
        x_values=np.asarray([row["selected_parameter"] for row in summaries]),
        series=(
            (
                np.asarray([row["maximum_displacement"] for row in summaries]),
                "deformation",
            ),
        ),
        show_markers=True,
    )
    write_line_chart(
        output / paths[4],
        title="Rotating-blocks pressure redistribution",
        x_label="slave node",
        y_label="pressure",
        x_values=np.arange(len(contact(checkpoints[0]).pressure), dtype=float),
        series=tuple(
            (np.asarray(contact(checkpoint).pressure), checkpoint.name)
            for checkpoint in checkpoints
        ),
        show_markers=True,
    )
    write_category_timeline(
        output / paths[5],
        title="Rotating-blocks topology events",
        x_label="continuation parameter",
        categories=tuple(str(row["kind"]) for row in completed.event_rows),
        x_values=np.asarray(
            [row["continuation_parameter"] for row in completed.event_rows]
        ),
        groups=tuple(row.get("attempt") for row in completed.event_rows),
    )
    write_polygon_overlay(
        output / paths[6],
        title="Final projected overlap regions",
        polygons=tuple(final_overlays),
        emphasized=tuple("intersection" in label for _, label in final_overlays),
        dashed=tuple("master" in label for _, label in final_overlays),
    )
    for relative in paths:
        ElementTree.parse(output / relative)
        writer.register(relative, "svg")
    return paths
