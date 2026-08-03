#!/usr/bin/env python3
"""Audit one-sided mortar branches at exact rotating-blocks special states."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rotating_blocks_model import QUICK_PROFILE, RotatingBlocksGeometry

from contact3d import (
    ClippingTopologyError,
    InverseMapTopologyError,
    PalletTopologyError,
    integrate_facet_pair,
    integrate_facet_pair_linearized,
)
from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.topology_events import (
    ContactTopologySignature,
    ContactTopologyStateMachine,
    TopologyEventLocalizationOptions,
    TopologyObservation,
)

SUMMARY_SCHEMA = "contact3d-rotating-blocks-special-states/v1"
BRANCH_SCHEMA = "contact3d-rotating-blocks-special-state-branches/v1"
_RECOVERABLE_ERRORS = (
    ClippingTopologyError,
    PalletTopologyError,
    InverseMapTopologyError,
)


@dataclass(frozen=True, slots=True)
class RotatingBlocksSpecialState:
    """One exact projected facet coincidence and its crossing direction."""

    name: str
    slave: np.ndarray
    master: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True, slots=True)
class BranchEvaluation:
    """One smooth side of an exact special state."""

    side: str
    offset: float
    area: float
    operator_norm: float
    tangent_norm: float
    tangent_error: float
    signature: ContactTopologySignature


def _quad(
    x_minimum: float,
    x_maximum: float,
    y_minimum: float,
    y_maximum: float,
    *,
    z: float,
) -> np.ndarray:
    return np.asarray(
        [
            (x_minimum, y_minimum, z),
            (x_maximum, y_minimum, z),
            (x_maximum, y_maximum, z),
            (x_minimum, y_maximum, z),
        ],
        dtype=float,
    )


def rotating_blocks_special_states() -> tuple[RotatingBlocksSpecialState, ...]:
    """Return exact local states derived from the quick rotating-blocks facets."""

    geometry = RotatingBlocksGeometry()
    master_width = (
        geometry.lower_maximum[0] - geometry.lower_minimum[0]
    ) / QUICK_PROFILE.lower_cells[0]
    master_height = (
        geometry.lower_maximum[1] - geometry.lower_minimum[1]
    ) / QUICK_PROFILE.lower_cells[1]
    slave_width = (
        geometry.upper_maximum[0] - geometry.upper_minimum[0]
    ) / QUICK_PROFILE.upper_cells[0]
    slave_height = (
        geometry.upper_maximum[1] - geometry.upper_minimum[1]
    ) / QUICK_PROFILE.upper_cells[1]
    z = geometry.lower_maximum[2]
    master = _quad(
        -master_width,
        0.0,
        -master_height,
        0.0,
        z=z,
    )
    edge_center = -0.5 * master_height
    edge = _quad(
        0.0,
        slave_width,
        edge_center - 0.5 * slave_height,
        edge_center + 0.5 * slave_height,
        z=z,
    )
    vertex = _quad(
        0.0,
        slave_width,
        0.0,
        slave_height,
        z=z,
    )
    return (
        RotatingBlocksSpecialState(
            "edge-on-edge",
            edge,
            master,
            np.asarray((1.0, 0.0, 0.0)),
        ),
        RotatingBlocksSpecialState(
            "on-vertex",
            vertex,
            master,
            np.asarray((1.0, 1.0, 0.0)),
        ),
    )


def _shifted_slave(case: RotatingBlocksSpecialState, offset: float) -> np.ndarray:
    return case.slave + float(offset) * case.direction[None, :]


def _local_direction(case: RotatingBlocksSpecialState) -> np.ndarray:
    direction = np.zeros(3 * (len(case.slave) + len(case.master)), dtype=float)
    for node in range(len(case.slave)):
        direction[3 * node : 3 * node + 3] = case.direction
    return direction


def _operator_vector(value: object) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(value.d, dtype=float).reshape(-1),
            np.asarray(value.m, dtype=float).reshape(-1),
        ]
    )


def _operator_directional_tangent(value: object, direction: np.ndarray) -> np.ndarray:
    derivative_d = np.tensordot(value.d_jacobian, direction, axes=(2, 0))
    derivative_m = np.tensordot(value.m_jacobian, direction, axes=(2, 0))
    return np.concatenate([derivative_d.reshape(-1), derivative_m.reshape(-1)])


def _unlinearized_operator(case: RotatingBlocksSpecialState, offset: float) -> np.ndarray:
    value = integrate_facet_pair(_shifted_slave(case, offset), case.master)
    return _operator_vector(value)


def _signature(value: object) -> ContactTopologySignature:
    geometry = value.quadrature.geometry
    area = float(geometry.fan.total_area)
    intersection = geometry.intersection.intersection_polygon
    supported = tuple(
        bool(item > 1.0e-14) for item in np.sum(value.d, axis=1)
    )
    occupied = area > 1.0e-14
    return ContactTopologySignature(
        ((0, 0),) if occupied else (),
        tuple(False for _ in supported),
        supported,
        (
            (
                0,
                0,
                len(intersection),
                len(geometry.fan.pallets),
                int(np.sign(area)),
            ),
        )
        if occupied
        else (),
    )


def _branch(
    case: RotatingBlocksSpecialState,
    side: str,
    offset: float,
    *,
    difference_step: float,
) -> BranchEvaluation:
    value = integrate_facet_pair_linearized(
        _shifted_slave(case, offset),
        case.master,
    )
    direction = _local_direction(case)
    tangent = _operator_directional_tangent(value, direction)
    numerical = (
        _unlinearized_operator(case, offset + difference_step)
        - _unlinearized_operator(case, offset - difference_step)
    ) / (2.0 * difference_step)
    scale = max(float(np.linalg.norm(numerical)), np.finfo(float).tiny)
    error = float(np.linalg.norm(tangent - numerical) / scale)
    return BranchEvaluation(
        side,
        float(offset),
        float(value.quadrature.geometry.fan.total_area),
        float(np.linalg.norm(_operator_vector(value))),
        float(np.linalg.norm(tangent)),
        error,
        _signature(value),
    )


def _observation(
    case: RotatingBlocksSpecialState,
    fraction: float,
    *,
    half_width: float,
) -> TopologyObservation:
    offset = (2.0 * float(fraction) - 1.0) * half_width
    try:
        value = integrate_facet_pair_linearized(
            _shifted_slave(case, offset),
            case.master,
        )
    except _RECOVERABLE_ERRORS as error:
        kind = (
            "clipping_vertex_edge"
            if isinstance(error, ClippingTopologyError)
            else "pallet_transition"
            if isinstance(error, PalletTopologyError)
            else "inverse_map_boundary"
        )
        return TopologyObservation.recoverable(fraction, kind, str(error))
    return TopologyObservation.valid(fraction, (_signature(value),), value)


def _selected_branch(
    case: RotatingBlocksSpecialState,
    branch: str,
    *,
    half_width: float,
) -> object:
    def observe(fraction: float) -> TopologyObservation:
        return _observation(
            case,
            fraction,
            half_width=half_width,
        )

    machine = ContactTopologyStateMachine(
        TopologyEventLocalizationOptions(branch_selection=branch)
    )
    return machine.localize(observe(0.0), observe(1.0), observe)


def _case_summary(
    case: RotatingBlocksSpecialState,
    *,
    half_width: float,
    difference_step: float,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    try:
        integrate_facet_pair_linearized(case.slave, case.master)
    except _RECOVERABLE_ERRORS as error:
        diagnostic = {
            "type": type(error).__name__,
            "detail": str(error),
        }
    else:
        diagnostic = {"type": None, "detail": ""}

    left = _branch(
        case,
        "left",
        -half_width,
        difference_step=difference_step,
    )
    right = _branch(
        case,
        "right",
        half_width,
        difference_step=difference_step,
    )
    left_selection = _selected_branch(case, "left", half_width=half_width)
    right_selection = _selected_branch(case, "right", half_width=half_width)
    event_kinds = sorted(
        {
            event.kind
            for batch in (left_selection, right_selection)
            for event in batch.events
        }
    )
    finite = all(
        np.isfinite(value)
        for branch in (left, right)
        for value in (
            branch.area,
            branch.operator_norm,
            branch.tangent_norm,
            branch.tangent_error,
        )
    )
    selected_branches_distinct = (
        left_selection.selected.signatures != right_selection.selected.signatures
    )
    criteria = {
        "typed_exact_diagnostic": diagnostic["type"] in {
            "ClippingTopologyError",
            "PalletTopologyError",
            "InverseMapTopologyError",
        },
        "clipping_event_recorded": "clipping_vertex_edge" in event_kinds,
        "left_branch_selected": (
            left_selection.selected_branch == "left"
            and left_selection.selected.is_valid
        ),
        "right_branch_selected": (
            right_selection.selected_branch == "right"
            and right_selection.selected.is_valid
        ),
        "selected_branches_distinct": selected_branches_distinct,
        "finite_branch_data": finite,
        "one_sided_tangents_verified": max(
            left.tangent_error,
            right.tangent_error,
        )
        <= 5.0e-5,
        "branches_are_distinct": left.signature != right.signature,
    }
    summary = {
        "name": case.name,
        "passed": all(criteria.values()),
        "criteria": criteria,
        "exact_diagnostic": diagnostic,
        "event_kinds": event_kinds,
        "left_selected_branch": str(left_selection.selected_branch),
        "right_selected_branch": str(right_selection.selected_branch),
        "left_selected_fraction": float(left_selection.selected_fraction),
        "right_selected_fraction": float(right_selection.selected_fraction),
        "maximum_tangent_error": max(left.tangent_error, right.tangent_error),
    }
    rows = tuple(
        {
            "case": case.name,
            "side": branch.side,
            "offset": branch.offset,
            "overlap_area": branch.area,
            "operator_norm": branch.operator_norm,
            "directional_tangent_norm": branch.tangent_norm,
            "directional_tangent_error": branch.tangent_error,
            "facet_pair_count": len(branch.signature.facet_pairs),
            "supported_row_count": sum(branch.signature.supported_rows),
            "intersection_vertex_count": (
                branch.signature.geometry_tokens[0][2]
                if branch.signature.geometry_tokens
                else 0
            ),
            "pallet_count": (
                branch.signature.geometry_tokens[0][3]
                if branch.signature.geometry_tokens
                else 0
            ),
        }
        for branch in (left, right)
    )
    return summary, rows


def run(
    output: Path | None = None,
    *,
    half_width: float = 1.0e-3,
    difference_step: float = 1.0e-6,
) -> dict[str, object]:
    """Evaluate exact special states and both smooth one-sided branches."""

    if not 0.0 < difference_step < half_width:
        raise ValueError("difference_step must lie between zero and half_width")
    summaries: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for case in rotating_blocks_special_states():
        summary, branch_rows = _case_summary(
            case,
            half_width=half_width,
            difference_step=difference_step,
        )
        summaries.append(summary)
        rows.extend(branch_rows)
    result = {
        "schema_version": SUMMARY_SCHEMA,
        "passed": all(bool(item["passed"]) for item in summaries),
        "half_width": half_width,
        "difference_step": difference_step,
        "cases": summaries,
    }
    if output is not None:
        writer = BenchmarkArtifactWriter(
            Path(output),
            "rotating-blocks-special-states",
            seed=0,
            solver_settings={
                "half_width": half_width,
                "difference_step": difference_step,
                "branch_policy": "explicit-left-and-right",
            },
            repo_root=Path(__file__).resolve().parents[1],
        )
        writer.write_json("summary.json", result, schema=SUMMARY_SCHEMA)
        writer.write_csv("branches.csv", rows, schema=BRANCH_SCHEMA)
        writer.finalize(required=("summary.json", "branches.csv"))
    if not result["passed"]:
        failed = [
            f"{item['name']}: "
            + ", ".join(
                name for name, passed in item["criteria"].items() if not passed
            )
            for item in summaries
            if not item["passed"]
        ]
        raise RuntimeError("special-state criteria failed: " + "; ".join(failed))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rotating-blocks-special-states"),
    )
    parser.add_argument("--half-width", type=float, default=1.0e-3)
    parser.add_argument("--difference-step", type=float, default=1.0e-6)
    arguments = parser.parse_args()
    summary = run(
        arguments.output,
        half_width=arguments.half_width,
        difference_step=arguments.difference_step,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
