#!/usr/bin/env python3
"""Generate deterministic mixed-boundary continuation regression artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np

from contact3d.adaptive import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    LinearBoundaryPath,
    LinearPathValue,
    contact_penalties,
    solve_adaptive_contact_path,
)
from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.coupled import AugmentedContactOptions
from contact3d.enforcement_state import AugmentedLagrangeState
from contact3d.equilibrium import DeadLoad, DirichletConstraints


@dataclass(frozen=True, slots=True)
class Mesh:
    node_count: int


@dataclass(frozen=True, slots=True)
class Pair:
    normal_penalty: float


@dataclass(frozen=True, slots=True)
class Interface:
    pair: Pair

    @property
    def normal_penalty(self) -> float:
        return self.pair.normal_penalty

    def with_normal_penalty(self, normal_penalty: float) -> Interface:
        return replace(self, pair=replace(self.pair, normal_penalty=normal_penalty))

    def reference_tributary_areas(self) -> np.ndarray:
        return np.ones(1)

    def initial_state(self) -> AugmentedLagrangeState:
        return AugmentedLagrangeState.zeros(1)


@dataclass(frozen=True, slots=True)
class Problem:
    mesh: Mesh
    constraints: DirichletConstraints
    load: DeadLoad
    interfaces: tuple[Interface, ...]
    sparsity: object

    def initial_states(self) -> tuple[AugmentedLagrangeState, ...]:
        return tuple(interface.initial_state() for interface in self.interfaces)


@dataclass(frozen=True, slots=True)
class Equilibrium:
    displacement: np.ndarray
    iteration_count: int
    contact_event_restarts: int
    evaluation: object


@dataclass(frozen=True, slots=True)
class Result:
    displacement: np.ndarray
    states: tuple[AugmentedLagrangeState, ...]
    converged: bool
    termination_reason: str
    equilibrium: Equilibrium
    equilibria: tuple[Equilibrium, ...]
    history: tuple[object, ...]


def model() -> tuple[Problem, LinearBoundaryPath]:
    constraints = DirichletConstraints(
        np.array([2, 4], dtype=np.int64),
        np.array([-0.12, 0.02]),
    )
    load = DeadLoad(np.array([3.0, 0.0, 0.0, 0.0, -2.0, 0.0]))
    problem = Problem(
        Mesh(2),
        constraints,
        load,
        (Interface(Pair(100.0)),),
        object(),
    )
    path = LinearBoundaryPath.proportional_mixed(
        problem,
        values=(
            LinearPathValue("tool_z", 0.0, -0.12),
            LinearPathValue("tool_x", 0.0, 0.02),
        ),
    )
    return problem, path


def _result(
    parameter: float,
    state_value: float,
    *,
    converged: bool,
) -> Result:
    displacement = np.full(6, parameter)
    residual = np.array(
        [0.0, 0.0, 120.0 * parameter, 0.0, -24.0 * parameter, 0.0]
    )
    evaluation = SimpleNamespace(
        maximum_penetration=0.0 if converged else 2.0e-3,
        free_residual_norm=1.0e-12 if converged else 3.0e-4,
        residual=residual,
        free_dofs=np.array([0, 1, 3, 5], dtype=np.int64),
    )
    equilibrium = Equilibrium(
        displacement,
        6 if converged else 12,
        int(parameter > 0.7),
        evaluation,
    )
    return Result(
        displacement,
        (AugmentedLagrangeState(np.array([state_value])),),
        converged,
        "converged" if converged else "inner_equilibrium_failed",
        equilibrium,
        (equilibrium,),
        (object(), object()),
    )


def _solver(problem, displacement, states, *, load_factor, options, tolerance):
    del displacement, options, tolerance
    parameter = abs(problem.constraints.values[0]) / 0.12
    first_attempt = states[0].multipliers[0] == 0.0
    if np.isclose(parameter, 0.75) and first_attempt:
        return _result(parameter, 99.0, converged=False)
    return _result(parameter, 10.0 + parameter, converged=True)


def _write_chart(
    path: Path,
    *,
    title: str,
    x_values: np.ndarray,
    series: tuple[tuple[np.ndarray, str], ...],
) -> None:
    width, height = 760, 430
    left, right, top, bottom = 72, 28, 42, 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min = min(float(np.min(values)) for values, _ in series)
    y_max = max(float(np.max(values)) for values, _ in series)
    padding = 0.08 * max(1.0, y_max - y_min)
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
            f'<text x="{width/2}" y="25" text-anchor="middle" '
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
    dash_patterns = ("", ' stroke-dasharray="7,4"', ' stroke-dasharray="2,4"')
    for index, (values, label) in enumerate(series):
        points = " ".join(
            f"{sx(float(x)):.2f},{sy(float(y)):.2f}"
            for x, y in zip(x_values, values, strict=True)
        )
        dash = dash_patterns[index % len(dash_patterns)]
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="black" '
            f'stroke-width="2"{dash}/>'
        )
        lines.append(
            f'<text x="{width-right}" y="{52+18*index}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{label}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    problem, path = model()
    options = AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.75,
            minimum_step=0.05,
            maximum_step=0.75,
            cutback_factor=0.5,
            growth_factor=1.5,
            easy_newton_iterations=2,
        ),
        penalty=AdaptivePenaltyOptions(enabled=False),
        augmented=AugmentedContactOptions(maximum_augmentations=3),
    )
    artifacts = BenchmarkArtifactWriter(
        output,
        "mixed-load-path",
        seed=0,
        solver_settings={"path": path, "adaptive": options},
        repo_root=Path(__file__).resolve().parents[1],
    )
    result = solve_adaptive_contact_path(
        problem,
        1.0,
        path=path,
        options=options,
        _solver=_solver,
    )
    if not result.converged:
        raise RuntimeError(result.termination_reason)

    step_rows: list[dict[str, object]] = []
    for index, step in enumerate(result.accepted_steps, start=1):
        reaction = step.reaction.reshape((-1, 3)).sum(axis=0)
        step_rows.append(
            {
                "accepted_step": index,
                "parameter": step.parameter,
                "tool_z": step.path_state.value("tool_z"),
                "tool_x": step.path_state.value("tool_x"),
                "prescribed_norm": step.path_state.prescribed_norm,
                "effective_load_norm": step.path_state.effective_load_norm,
                "reaction_x": float(reaction[0]),
                "reaction_y": float(reaction[1]),
                "reaction_z": float(reaction[2]),
                "reaction_norm": step.reaction_norm,
                "state_multiplier": float(step.result.states[0].multipliers[0]),
            }
        )
    attempt_rows = [
        {
            "attempt": item.attempt,
            "start_parameter": item.start_load_factor,
            "target_parameter": item.target_load_factor,
            "step_size": item.step_size,
            "action": item.action,
            "inner_termination_reason": item.inner_termination_reason,
            "augmentations": item.augmentations,
            "newton_iterations": item.newton_iterations,
            "contact_event_restarts": item.contact_event_restarts,
            "equilibrium_residual": item.equilibrium_residual,
            "maximum_penetration": item.maximum_penetration,
            "effective_load_norm": item.effective_load_norm,
            "reaction_norm": item.reaction_norm,
            "penalties_before": item.penalties_before,
            "penalties_after": item.penalties_after,
            "prescribed_values": item.prescribed_values,
            "penalty_update_reasons": item.penalty_update_reasons,
        }
        for item in result.attempts
    ]

    metrics = {
        "converged": result.converged,
        "termination_reason": result.termination_reason,
        "accepted_steps": result.accepted_step_count,
        "cutbacks": result.cutback_count,
        "attempts": len(result.attempts),
        "final_parameter": result.load_factor,
        "final_tool_z": step_rows[-1]["tool_z"],
        "final_tool_x": step_rows[-1]["tool_x"],
        "final_effective_load_norm": step_rows[-1]["effective_load_norm"],
        "final_reaction_norm": result.accepted_steps[-1].reaction_norm,
        "sparsity_reused": all(
            step.path_state.problem.sparsity is problem.sparsity
            for step in result.accepted_steps
        ),
    }
    summary = {
        "schema_version": "contact3d-mixed-load-path/v1",
        "path": {
            "type": "linear proportional mixed boundary/load path",
            "values": [
                {"name": value.name, "start": value.start, "end": value.end}
                for value in path.values
            ],
        },
        "metrics": metrics,
        "accepted_steps": step_rows,
        "attempts": attempt_rows,
        "penalties": contact_penalties(result.problem),
    }
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-mixed-load-path/v1",
    )
    artifacts.write_csv(
        "accepted-steps.csv",
        step_rows,
        schema="contact3d-mixed-path-steps/v1",
    )
    artifacts.write_csv(
        "attempt-history.csv",
        attempt_rows,
        schema="contact3d-continuation-attempts/v1",
    )

    parameter = np.asarray([float(row["parameter"]) for row in step_rows])
    _write_chart(
        output / "boundary-path.svg",
        title="Accepted mixed boundary/load path",
        x_values=parameter,
        series=(
            (np.asarray([float(row["tool_z"]) for row in step_rows]), "tool z"),
            (np.asarray([float(row["tool_x"]) for row in step_rows]), "tool x"),
            (
                np.asarray([float(row["effective_load_norm"]) for row in step_rows]),
                "load norm",
            ),
        ),
    )
    _write_chart(
        output / "reaction-path.svg",
        title="Constrained reaction history",
        x_values=parameter,
        series=(
            (np.asarray([float(row["reaction_x"]) for row in step_rows]), "reaction x"),
            (np.asarray([float(row["reaction_z"]) for row in step_rows]), "reaction z"),
        ),
    )
    for svg in output.glob("*.svg"):
        ElementTree.parse(svg)
        artifacts.register(svg.name, "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "accepted-steps.csv",
            "attempt-history.csv",
            "boundary-path.svg",
            "reaction-path.svg",
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mixed-load-path"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output)["metrics"], indent=2))


if __name__ == "__main__":
    main()
