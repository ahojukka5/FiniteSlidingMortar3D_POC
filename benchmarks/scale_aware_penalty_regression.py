#!/usr/bin/env python3
"""Generate deterministic unit-invariance and local-penalty regression artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.bulk_material import NeoHookeanMaterial
from contact3d.enforcement_state import KKTDiagnostics
from contact3d.scaling import (
    coupled_problem_scales,
    propose_interface_penalties,
)


@dataclass(frozen=True, slots=True)
class Mesh:
    reference_nodes: np.ndarray


@dataclass(frozen=True, slots=True)
class Interface:
    penalty: float
    areas: np.ndarray

    @property
    def normal_penalty(self) -> float:
        return self.penalty

    def with_normal_penalty(self, normal_penalty: float) -> Interface:
        return replace(self, penalty=normal_penalty)

    def reference_tributary_areas(self) -> np.ndarray:
        return self.areas.copy()


@dataclass(frozen=True, slots=True)
class Problem:
    mesh: Mesh
    material: NeoHookeanMaterial
    interfaces: tuple[Interface, ...]


@dataclass(frozen=True, slots=True)
class Signature:
    active_rows: tuple[bool, ...]
    supported_rows: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class Contact:
    diagnostics: KKTDiagnostics
    pressure: np.ndarray
    signature: Signature


def material(young: float) -> NeoHookeanMaterial:
    return NeoHookeanMaterial.from_young_poisson(young, 0.3)


def case(
    length_factor: float,
    pressure_factor: float,
) -> tuple[Problem, tuple[Contact, ...]]:
    nodes = length_factor * np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 1.0]])
    areas = length_factor**2
    penalty_factor = pressure_factor / length_factor
    problem = Problem(
        Mesh(nodes),
        material(pressure_factor * 210.0),
        (
            Interface(penalty_factor * 100.0, areas * np.array([0.25, 0.25])),
            Interface(penalty_factor * 100.0, areas * np.array([0.01, 0.01])),
        ),
    )
    contacts = (
        Contact(
            KKTDiagnostics(
                np.array([length_factor * 1.0e-10]),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
            ),
            np.array([pressure_factor * 0.2]),
            Signature((False,), (True,)),
        ),
        Contact(
            KKTDiagnostics(
                np.array([length_factor * 2.0e-4]),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
                np.zeros(1),
            ),
            np.array([pressure_factor * 1.0]),
            Signature((True,), (True,)),
        ),
    )
    return problem, contacts


def evaluate(name: str, length_factor: float, pressure_factor: float):
    problem, contacts = case(length_factor, pressure_factor)
    scales = coupled_problem_scales(problem)
    plan = propose_interface_penalties(
        problem,
        contacts,
        increase_factor=4.0,
        absolute_maximum=pressure_factor / length_factor * 1.0e12,
        minimum_scale_factor=0.25,
        maximum_scale_factor=100.0,
        dimensional_target=None,
        normalized_target=1.0e-6,
        use_normalized_target=True,
        interface_local=True,
    )
    rows = []
    for index, (interface, contact, scale, new_penalty) in enumerate(
        zip(
            problem.interfaces,
            contacts,
            scales.interfaces,
            plan.penalties,
            strict=True,
        )
    ):
        rows.append(
            {
                "unit_system": name,
                "interface": index,
                "length_scale": scale.length,
                "pressure_scale": scale.pressure,
                "area_scale": scale.area,
                "penetration": contact.diagnostics.maximum_penetration,
                "normalized_penetration": (
                    contact.diagnostics.maximum_penetration / scale.length
                ),
                "penalty_before": interface.normal_penalty,
                "penalty_after": new_penalty,
                "penalty_ratio_before": scale.penalty_ratio(interface.normal_penalty),
                "penalty_ratio_after": scale.penalty_ratio(new_penalty),
                "updated": new_penalty > interface.normal_penalty,
                "reason": next(
                    (
                        decision.reason
                        for decision in plan.decisions
                        if decision.interface == index
                    ),
                    "resolved",
                ),
            }
        )
    return rows, plan


def write_chart(
    path: Path,
    title: str,
    rows: list[dict[str, object]],
    field: str,
) -> None:
    width, height = 760, 420
    left, right, top, bottom = 72, 28, 44, 64
    values = np.array([float(row[field]) for row in rows], dtype=float)
    maximum = max(float(np.max(values, initial=0.0)), 1.0e-15)
    x_step = (width - left - right) / max(len(rows), 1)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="26" text-anchor="middle" '
            f'font-family="sans-serif" font-size="16">{title}</text>'
        ),
        (
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
    ]
    for index, (row, value) in enumerate(zip(rows, values, strict=True)):
        center = left + (index + 0.5) * x_step
        bar_height = value / maximum * (height - top - bottom)
        y = height - bottom - bar_height
        lines.extend(
            [
                (
                    f'<rect x="{center - 0.25 * x_step:.2f}" y="{y:.2f}" '
                    f'width="{0.5 * x_step:.2f}" height="{bar_height:.2f}" '
                    'fill="none" stroke="black"/>'
                ),
                (
                    f'<text x="{center:.2f}" y="{height-bottom+20}" '
                    f'text-anchor="middle" font-family="monospace" font-size="10">'
                    f'{row["unit_system"]}:I{row["interface"]}</text>'
                ),
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        "increase_factor": 4.0,
        "minimum_scale_factor": 0.25,
        "maximum_scale_factor": 100.0,
        "normalized_target": 1.0e-6,
        "use_normalized_target": True,
        "interface_local": True,
    }
    artifacts = BenchmarkArtifactWriter(
        output,
        "scale-aware-penalty",
        seed=0,
        solver_settings=settings,
        repo_root=Path(__file__).resolve().parents[1],
    )
    base_rows, base_plan = evaluate("m-Pa", 1.0, 1.0)
    converted_rows, converted_plan = evaluate("mm-MPa", 1000.0, 1.0e-6)
    rows = base_rows + converted_rows

    invariant = all(
        np.isclose(
            float(base[field]),
            float(converted[field]),
            rtol=1.0e-12,
            atol=1.0e-15,
        )
        for base, converted in zip(base_rows, converted_rows, strict=True)
        for field in (
            "normalized_penetration",
            "penalty_ratio_before",
            "penalty_ratio_after",
        )
    )
    summary = {
        "schema_version": "contact3d-scale-aware-penalty/v1",
        "unit_systems": (
            {"name": "m-Pa", "length_factor": 1.0, "pressure_factor": 1.0},
            {
                "name": "mm-MPa",
                "length_factor": 1000.0,
                "pressure_factor": 1.0e-6,
            },
        ),
        "metrics": {
            "unit_invariant": bool(invariant),
            "updated_interfaces": [
                decision.interface for decision in base_plan.decisions
            ],
            "converted_updated_interfaces": [
                decision.interface for decision in converted_plan.decisions
            ],
            "normalized_target": 1.0e-6,
            "interface_count": 2,
            "updated_interface_count": len(base_plan.decisions),
        },
        "interfaces": rows,
        "reasons": list(base_plan.reasons),
    }
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-scale-aware-penalty/v1",
    )
    artifacts.write_csv(
        "interface-penalty-history.csv",
        rows,
        schema="contact3d-interface-penalties/v1",
    )
    write_chart(
        output / "normalized-penetration.svg",
        "Normalized interface penetration",
        rows,
        "normalized_penetration",
    )
    write_chart(
        output / "penalty-ratio.svg",
        "Dimensionless normal-penalty ratio after update",
        rows,
        "penalty_ratio_after",
    )
    for path in output.glob("*.svg"):
        ElementTree.parse(path)
        artifacts.register(path.name, "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "interface-penalty-history.csv",
            "normalized-penetration.svg",
            "penalty-ratio.svg",
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/scale-aware-penalty"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output)["metrics"], indent=2))


if __name__ == "__main__":
    main()
