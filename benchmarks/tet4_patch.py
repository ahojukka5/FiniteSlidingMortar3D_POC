#!/usr/bin/env python3
"""Generate deterministic TET4 affine-patch and tangent-convergence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.bulk import NeoHookeanMaterial, Tet4Mesh, evaluate_tet4_mesh


def cube_star_mesh() -> Tet4Mesh:
    nodes = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 0.5, 0.5],
        ]
    )
    surface_triangles = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (3, 7, 6),
        (3, 6, 2),
        (0, 4, 7),
        (0, 7, 3),
        (1, 2, 6),
        (1, 6, 5),
    ]
    elements = np.array([(8, *triangle) for triangle in surface_triangles], dtype=np.int64)
    return Tet4Mesh(nodes, elements)


def affine_displacement(
    nodes: np.ndarray,
    deformation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return nodes @ deformation.T + translation - nodes


def convergence_rows(
    mesh: Tet4Mesh,
    displacement: np.ndarray,
    material: NeoHookeanMaterial,
) -> list[dict[str, float | None]]:
    base = evaluate_tet4_mesh(mesh, displacement, material)
    rng = np.random.default_rng(714205)
    direction = rng.normal(size=displacement.shape)
    direction /= np.linalg.norm(direction)
    exact = base.tangent @ direction.ravel()
    rows: list[dict[str, float | None]] = []
    for step in (
        1.0e-2,
        3.0e-3,
        1.0e-3,
        3.0e-4,
        1.0e-4,
        3.0e-5,
        1.0e-5,
        3.0e-6,
        1.0e-6,
    ):
        plus = evaluate_tet4_mesh(
            mesh,
            displacement + step * direction,
            material,
        ).residual.ravel()
        minus = evaluate_tet4_mesh(
            mesh,
            displacement - step * direction,
            material,
        ).residual.ravel()
        numerical = (plus - minus) / (2.0 * step)
        error = np.linalg.norm(numerical - exact) / np.linalg.norm(exact)
        observed_order = None
        if rows:
            previous = rows[-1]
            previous_error = float(previous["relative_error"])
            previous_step = float(previous["step"])
            observed_order = float(
                np.log(previous_error / error) / np.log(previous_step / step)
            )
        rows.append(
            {
                "step": step,
                "relative_error": float(error),
                "observed_order": observed_order,
            }
        )
    return rows


def write_svg(rows: list[dict[str, float | None]], path: Path) -> None:
    width = 760
    height = 460
    left = 82
    right = 28
    top = 36
    bottom = 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = np.log10([float(row["step"]) for row in rows])
    y_values = np.log10([float(row["relative_error"]) for row in rows])
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
    y_padding = 0.08 * max(1.0, y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    points = " ".join(
        f"{sx(x):.2f},{sy(y):.2f}"
        for x, y in zip(x_values, y_values, strict=True)
    )
    x_ticks = range(int(np.ceil(x_min)), int(np.floor(x_max)) + 1)
    y_ticks = range(int(np.ceil(y_min)), int(np.floor(y_max)) + 1)
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="380" y="23" text-anchor="middle" font-family="sans-serif" '
            'font-size="16">TET4 analytical tangent convergence</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
    ]
    for tick in x_ticks:
        x = sx(float(tick))
        lines.extend(
            [
                (
                    f'<line x1="{x:.2f}" y1="{height-bottom}" x2="{x:.2f}" '
                    f'y2="{height-bottom+6}" stroke="black"/>'
                ),
                (
                    f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" '
                    f'font-family="monospace" font-size="12">1e{tick}</text>'
                ),
            ]
        )
    for tick in y_ticks:
        y = sy(float(tick))
        lines.extend(
            [
                (
                    f'<line x1="{left-6}" y1="{y:.2f}" x2="{left}" '
                    f'y2="{y:.2f}" stroke="black"/>'
                ),
                (
                    f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" '
                    f'y2="{y:.2f}" stroke="#dddddd"/>'
                ),
                (
                    f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" '
                    f'font-family="monospace" font-size="12">1e{tick}</text>'
                ),
            ]
        )
    lines.extend(
        [
            f'<polyline points="{points}" fill="none" stroke="black" stroke-width="2"/>',
            *[
                f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.5" fill="black"/>'
                for x, y in zip(x_values, y_values, strict=True)
            ],
            (
                f'<text x="{left + plot_width/2:.2f}" y="{height-18}" '
                'text-anchor="middle" font-family="sans-serif" font-size="13">'
                "centered-difference step</text>"
            ),
            (
                f'<text x="18" y="{top + plot_height/2:.2f}" text-anchor="middle" '
                f'transform="rotate(-90 18 {top + plot_height/2:.2f})" '
                'font-family="sans-serif" font-size="13">relative directional '
                "tangent error</text>"
            ),
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    mesh = cube_star_mesh()
    material = NeoHookeanMaterial.from_young_poisson(210.0, 0.3)
    deformation = np.array(
        [[1.08, 0.12, -0.04], [0.03, 0.94, 0.08], [0.02, -0.05, 1.11]]
    )
    translation = np.array([0.2, -0.1, 0.05])
    displacement = affine_displacement(
        mesh.reference_nodes,
        deformation,
        translation,
    )
    artifacts = BenchmarkArtifactWriter(
        output,
        "tet4-patch",
        seed=714205,
        solver_settings={
            "deformation_gradient": deformation,
            "translation": translation,
            "difference_steps": (
                1.0e-2,
                3.0e-3,
                1.0e-3,
                3.0e-4,
                1.0e-4,
                3.0e-5,
                1.0e-5,
                3.0e-6,
                1.0e-6,
            ),
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    evaluation = evaluate_tet4_mesh(mesh, displacement, material)
    rows = convergence_rows(mesh, displacement, material)
    result = {
        "schema_version": "contact3d-tet4-patch/v1",
        "mesh": {
            "node_count": mesh.node_count,
            "element_count": mesh.element_count,
            "reference_volume": mesh.reference_volume,
        },
        "material": {
            "shear_modulus": material.shear_modulus,
            "bulk_modulus": material.bulk_modulus,
            "lame_lambda": material.lame_lambda,
        },
        "deformation_gradient": deformation.tolist(),
        "translation": translation.tolist(),
        "metrics": {
            "minimum_jacobian": evaluation.minimum_jacobian,
            "interior_residual_norm": float(np.linalg.norm(evaluation.residual[8])),
            "force_balance_norm": float(np.linalg.norm(evaluation.force_balance)),
            "moment_balance_norm": float(np.linalg.norm(evaluation.moment_balance)),
            "maximum_deformation_gradient_error": max(
                float(np.linalg.norm(element.deformation_gradient - deformation))
                for element in evaluation.element_evaluations
            ),
            "total_energy": evaluation.energy,
            "minimum_tangent_error": min(
                float(row["relative_error"]) for row in rows
            ),
        },
        "tangent_convergence": rows,
    }
    artifacts.write_json(
        "summary.json",
        result,
        schema="contact3d-tet4-patch/v1",
    )
    artifacts.write_csv(
        "tangent-convergence.csv",
        rows,
        schema="contact3d-tangent-convergence/v1",
    )
    artifacts.write_tet4_vtu(
        "affine-patch.vtu",
        mesh.reference_nodes,
        mesh.elements,
        displacement,
        point_data={
            "internal_force": evaluation.residual,
        },
        cell_data={
            "jacobian": np.asarray(
                [item.jacobian for item in evaluation.element_evaluations]
            ),
            "energy_density": np.asarray(
                [item.energy_density for item in evaluation.element_evaluations]
            ),
            "deformation_gradient_error": np.asarray(
                [
                    np.linalg.norm(item.deformation_gradient - deformation)
                    for item in evaluation.element_evaluations
                ]
            ),
        },
    )
    write_svg(rows, output / "tangent-convergence.svg")
    ElementTree.parse(output / "tangent-convergence.svg")
    artifacts.register("tangent-convergence.svg", "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "tangent-convergence.csv",
            "affine-patch.vtu",
            "tangent-convergence.svg",
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/tet4-patch"))
    arguments = parser.parse_args()
    result = run(arguments.output)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
