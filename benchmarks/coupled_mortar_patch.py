#!/usr/bin/env python3
"""Generate coupled bulk/contact Newton and augmented-Lagrange artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from contact3d import (
    AugmentedContactOptions,
    AugmentedContactResult,
    CoupledEquilibriumProblem,
    DeadLoad,
    DirichletConstraints,
    NeoHookeanMaterial,
    NewtonOptions,
    Tet4Mesh,
    solve_augmented_contact,
)
from contact3d.coupled_oracle import FrozenMatchingMortarInterface
from svg_plots import write_line_chart


def block_nodes(z_origin: float) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, z_origin],
            [1.0, 0.0, z_origin],
            [1.0, 1.0, z_origin],
            [0.0, 1.0, z_origin],
            [0.0, 0.0, z_origin + 1.0],
            [1.0, 0.0, z_origin + 1.0],
            [1.0, 1.0, z_origin + 1.0],
            [0.0, 1.0, z_origin + 1.0],
            [0.5, 0.5, z_origin + 0.5],
        ]
    )


def block_elements(offset: int) -> np.ndarray:
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
    return np.asarray(
        [(offset + 8, offset + a, offset + b, offset + c) for a, b, c in surface_triangles],
        dtype=np.int64,
    )


def problem(*, penalty: float = 6400.0) -> CoupledEquilibriumProblem:
    nodes = np.vstack([block_nodes(0.0), block_nodes(1.0)])
    elements = np.vstack([block_elements(0), block_elements(9)])
    mesh = Tet4Mesh(nodes, elements)
    interface = FrozenMatchingMortarInterface(
        np.array([9, 12, 11, 10], dtype=np.int64),
        np.array([4, 7, 6, 5], dtype=np.int64),
        np.array([0.0, 0.0, -1.0]),
        penalty,
    )

    constrained_dofs: list[int] = []
    prescribed_values: list[float] = []
    for node in (0, 1, 2, 3):
        for component in range(3):
            constrained_dofs.append(3 * node + component)
            prescribed_values.append(0.0)
    for node in (13, 14, 15, 16):
        for component in range(3):
            constrained_dofs.append(3 * node + component)
            prescribed_values.append(-0.12 if component == 2 else 0.0)

    constraints = DirichletConstraints(
        np.asarray(constrained_dofs, dtype=np.int64),
        np.asarray(prescribed_values),
    )
    load = DeadLoad(np.zeros(3 * mesh.node_count))
    material = NeoHookeanMaterial.from_young_poisson(210.0, 0.3)
    return CoupledEquilibriumProblem(mesh, material, constraints, load, (interface,))


def write_interface_plot(
    path: Path,
    gaps: np.ndarray,
    pressures: np.ndarray,
) -> None:
    width, height = 760, 420
    left, right, top, bottom = 70, 30, 40, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = max(float(np.max(pressures, initial=0.0)), 1.0)
    bar_width = plot_width / (2.0 * len(pressures))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<text x="{width / 2}" y="24" text-anchor="middle" '
            'font-family="sans-serif" font-size="16">Final interface pressure</text>'
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        (
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="black"/>'
        ),
    ]
    for node, (gap, pressure) in enumerate(zip(gaps, pressures, strict=True)):
        center = left + (node + 0.5) * plot_width / len(pressures)
        bar_height = pressure / maximum * plot_height
        x = center - bar_width / 2.0
        y = height - bottom - bar_height
        lines.extend(
            [
                (
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                    f'height="{bar_height:.2f}" fill="none" stroke="black"/>'
                ),
                (
                    f'<text x="{center:.2f}" y="{height-bottom+22}" text-anchor="middle" '
                    f'font-family="monospace" font-size="12">A{node}</text>'
                ),
                (
                    f'<text x="{center:.2f}" y="{max(top+14, y-6):.2f}" '
                    f'text-anchor="middle" font-family="monospace" font-size="10">'
                    f'p={pressure:.4g}, g={gap:.3g}</text>'
                ),
            ]
        )
    lines.append('</svg>')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')



def solve_directional_residual(
    coupled_problem: CoupledEquilibriumProblem,
    result: AugmentedContactResult,
    direction: np.ndarray,
    step: float,
) -> np.ndarray:
    from contact3d import evaluate_coupled_equilibrium

    return evaluate_coupled_equilibrium(
        coupled_problem,
        result.displacement + step * direction,
        result.states,
        assemble_tangent=False,
    ).residual


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    coupled_problem = problem()
    options = AugmentedContactOptions(
        maximum_augmentations=16,
        gap_tolerance=1.0e-8,
        complementarity_tolerance=1.0e-7,
        projection_tolerance=1.0e-5,
        multiplier_tolerance=1.0e-8,
        event_policy='restart',
        newton=NewtonOptions(
            maximum_iterations=40,
            absolute_tolerance=1.0e-10,
            relative_tolerance=1.0e-10,
        ),
    )
    result = solve_augmented_contact(coupled_problem, options=options)

    augmentation_rows = [
        {name: getattr(row, name) for name in row.__dataclass_fields__}
        for row in result.history
    ]
    newton_rows: list[dict[str, object]] = []
    global_iteration = 0
    for augmentation, equilibrium in enumerate(result.equilibria):
        for row in equilibrium.history:
            global_iteration += 1
            values = {name: getattr(row, name) for name in row.__dataclass_fields__}
            values['augmentation'] = augmentation
            values['global_iteration'] = global_iteration
            newton_rows.append(values)

    contact = result.equilibrium.evaluation.contacts[0]
    final_evaluation = result.equilibrium.evaluation
    rng = np.random.default_rng(8841)
    direction = np.zeros_like(result.displacement)
    direction[final_evaluation.free_dofs] = rng.normal(
        size=len(final_evaluation.free_dofs)
    )
    direction /= np.linalg.norm(direction)
    assert final_evaluation.tangent is not None
    exact_directional_tangent = final_evaluation.tangent.to_dense() @ direction
    tangent_step = 2.0e-7
    plus = solve_directional_residual(
        coupled_problem, result, direction, tangent_step
    )
    minus = solve_directional_residual(
        coupled_problem, result, direction, -tangent_step
    )
    numerical_directional_tangent = (plus - minus) / (2.0 * tangent_step)
    free = final_evaluation.free_dofs
    tangent_error = float(
        np.linalg.norm(
            exact_directional_tangent[free] - numerical_directional_tangent[free]
        )
        / np.linalg.norm(numerical_directional_tangent[free])
    )
    interface_rows = [
        {
            'node': node,
            'normal_gap': float(contact.normal_gaps[node]),
            'pressure': float(contact.pressure[node]),
            'multiplier': float(result.states[0].multipliers[node]),
            'active': bool(contact.signature.active_rows[node]),
        }
        for node in range(len(contact.normal_gaps))
    ]
    metrics = {
        'converged': result.converged,
        'termination_reason': result.termination_reason,
        'augmentations': len(result.history),
        'total_newton_iterations': sum(row.newton_iterations for row in result.history),
        'final_residual': result.equilibrium.evaluation.free_residual_norm,
        'maximum_penetration': result.equilibrium.evaluation.maximum_penetration,
        'contact_event_restarts': sum(row.contact_event_restarts for row in result.history),
        'minimum_jacobian': result.equilibrium.evaluation.bulk.minimum_jacobian,
        'maximum_pressure': float(np.max(contact.pressure, initial=0.0)),
        'interface_mean_gap': float(np.mean(contact.normal_gaps)),
        'directional_tangent_error': tangent_error,
        'global_force_balance_norm': float(
            np.linalg.norm(final_evaluation.residual.reshape((-1, 3)).sum(axis=0))
        ),
        'contact_force_balance_norm': float(
            np.linalg.norm(contact.residual.reshape((-1, 3)).sum(axis=0))
        ),
    }
    summary = {
        'mesh': {
            'node_count': coupled_problem.mesh.node_count,
            'element_count': coupled_problem.mesh.element_count,
            'total_dofs': 3 * coupled_problem.mesh.node_count,
            'sparse_nnz': coupled_problem.sparsity.nnz,
        },
        'contact': {
            'penalty': coupled_problem.interfaces[0].penalty,
            'prescribed_compression': 0.12,
            'operator': 'frozen matching Q1 standard-mortar D=M',
        },
        'metrics': metrics,
        'augmentation_history': augmentation_rows,
        'interface_state': interface_rows,
    }
    (output / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n',
        encoding='utf-8',
    )

    with (output / 'augmentation-history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(augmentation_rows[0]))
        writer.writeheader()
        writer.writerows(augmentation_rows)
    with (output / 'newton-history.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(newton_rows[0]))
        writer.writeheader()
        writer.writerows(newton_rows)
    with (output / 'interface-state.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(interface_rows[0]))
        writer.writeheader()
        writer.writerows(interface_rows)

    augmentation_index = np.arange(len(augmentation_rows), dtype=float)
    write_line_chart(
        output / 'kkt-convergence.svg',
        title='Augmented-Lagrange KKT convergence',
        x_label='augmentation',
        y_label='residual magnitude',
        x_values=augmentation_index,
        series=(
            (
                np.asarray([row['maximum_penetration'] for row in augmentation_rows]),
                'penetration',
            ),
            (
                np.asarray(
                    [row['maximum_projection_residual'] for row in augmentation_rows]
                ),
                'projection',
            ),
        ),
        logarithmic_y=True,
    )
    write_line_chart(
        output / 'newton-residual.svg',
        title='Coupled Newton convergence across augmentations',
        x_label='accepted Newton iteration',
        y_label='free residual norm',
        x_values=np.arange(1, len(newton_rows) + 1, dtype=float),
        series=((np.asarray([row['residual_norm'] for row in newton_rows]), 'residual'),),
        logarithmic_y=True,
    )
    write_interface_plot(
        output / 'interface-pressure.svg',
        contact.normal_gaps,
        contact.pressure,
    )
    for path in output.glob('*.svg'):
        ElementTree.parse(path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('results/coupled-mortar-patch'),
    )
    arguments = parser.parse_args()
    summary = run(arguments.output)
    print(json.dumps(summary['metrics'], indent=2))


if __name__ == '__main__':
    main()
