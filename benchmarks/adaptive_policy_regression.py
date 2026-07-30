#!/usr/bin/env python3
"""Generate deterministic adaptive-continuation controller regression artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np

from contact3d.adaptive import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    contact_penalties,
    solve_adaptive_contact_path,
)
from contact3d.coupled import AugmentedContactOptions, AugmentedContactResult
from contact3d.enforcement_state import AugmentedLagrangeState
from svg_plots import write_line_chart


@dataclass(frozen=True, slots=True)
class Mesh:
    node_count: int = 2


@dataclass(frozen=True, slots=True)
class Pair:
    normal_penalty: float


@dataclass(frozen=True, slots=True)
class Interface:
    pair: Pair

    def initial_state(self) -> AugmentedLagrangeState:
        return AugmentedLagrangeState.zeros(1)


@dataclass(frozen=True, slots=True)
class Problem:
    mesh: Mesh
    interfaces: tuple[Interface, ...]

    def initial_states(self) -> tuple[AugmentedLagrangeState, ...]:
        return tuple(interface.initial_state() for interface in self.interfaces)


@dataclass(frozen=True, slots=True)
class Equilibrium:
    displacement: np.ndarray
    iteration_count: int
    contact_event_restarts: int
    evaluation: object


def make_result(
    *,
    load_factor: float,
    penalty: float,
    converged: bool,
    reason: str,
    penetration: float,
    newton_iterations: int,
    state_value: float,
) -> AugmentedContactResult:
    displacement = np.full(6, load_factor + 1.0e-8 * penalty)
    state = AugmentedLagrangeState(np.array([state_value]))
    evaluation = SimpleNamespace(
        maximum_penetration=penetration,
        free_residual_norm=1.0e-12 if converged else 2.0e-4,
    )
    equilibrium = Equilibrium(
        displacement,
        newton_iterations,
        int(load_factor >= 0.8),
        evaluation,
    )
    return AugmentedContactResult(
        displacement,
        (state,),
        converged,
        reason,
        equilibrium,
        (equilibrium,),
        tuple(range(2 if converged else 3)),
    )


def scripted_solver(problem, displacement, states, *, load_factor, options, tolerance):
    penalty = contact_penalties(problem)[0]
    state_value = float(states[0].multipliers[0])
    if np.isclose(load_factor, 0.8) and np.isclose(penalty, 100.0):
        return make_result(
            load_factor=load_factor,
            penalty=penalty,
            converged=False,
            reason="inner_equilibrium_failed",
            penetration=3.0e-3,
            newton_iterations=12,
            state_value=state_value,
        )
    if np.isclose(load_factor, 0.4) and np.isclose(penalty, 100.0):
        return make_result(
            load_factor=load_factor,
            penalty=penalty,
            converged=False,
            reason="maximum_augmentations",
            penetration=2.0e-3,
            newton_iterations=7,
            state_value=7.0,
        )
    return make_result(
        load_factor=load_factor,
        penalty=penalty,
        converged=True,
        reason="converged",
        penetration=4.0e-9,
        newton_iterations=3,
        state_value=11.0 + load_factor,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    options = AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.8,
            minimum_step=0.05,
            maximum_step=0.8,
            growth_factor=2.0,
            easy_newton_iterations=5,
        ),
        penalty=AdaptivePenaltyOptions(
            increase_factor=4.0,
            maximum_penalty=1600.0,
            maximum_updates_per_step=2,
        ),
        augmented=AugmentedContactOptions(maximum_augmentations=3, gap_tolerance=1.0e-8),
    )
    result = solve_adaptive_contact_path(
        Problem(Mesh(), (Interface(Pair(100.0)),)),
        1.0,
        options=options,
        _solver=scripted_solver,
    )
    rows = [
        {name: getattr(attempt, name) for name in attempt.__dataclass_fields__}
        for attempt in result.attempts
    ]
    write_csv(output / "attempt-history.csv", rows)

    accepted_rows = [
        {
            "accepted_step": index + 1,
            "load_factor": attempt.target_load_factor,
            "step_size": attempt.step_size,
            "penalty": attempt.penalties_after[0],
            "newton_iterations": attempt.newton_iterations,
            "maximum_penetration": attempt.maximum_penetration,
        }
        for index, attempt in enumerate(
            item for item in result.attempts if item.action == "accepted"
        )
    ]
    write_csv(output / "accepted-steps.csv", accepted_rows)

    summary = {
        "metrics": {
            "converged": result.converged,
            "termination_reason": result.termination_reason,
            "accepted_steps": result.accepted_step_count,
            "cutbacks": result.cutback_count,
            "penalty_updates": result.penalty_update_count,
            "attempts": len(result.attempts),
            "final_load_factor": result.load_factor,
            "initial_penalty": 100.0,
            "final_penalty": contact_penalties(result.problem)[0],
        },
        "attempts": rows,
        "accepted_path": accepted_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    accepted = [item for item in result.attempts if item.action == "accepted"]
    x_values = np.arange(1, len(accepted) + 1, dtype=float)
    write_line_chart(
        output / "load-path.svg",
        title="Adaptive accepted load path",
        x_label="accepted step",
        y_label="load factor",
        x_values=x_values,
        series=((np.array([item.target_load_factor for item in accepted]), "load"),),
    )
    write_line_chart(
        output / "penalty-path.svg",
        title="Committed normal-penalty path",
        x_label="accepted step",
        y_label="normal penalty",
        x_values=x_values,
        series=((np.array([item.penalties_after[0] for item in accepted]), "penalty"),),
    )
    for path in output.glob("*.svg"):
        ElementTree.parse(path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/adaptive-contact-policy"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output)["metrics"], indent=2))


if __name__ == "__main__":
    main()
