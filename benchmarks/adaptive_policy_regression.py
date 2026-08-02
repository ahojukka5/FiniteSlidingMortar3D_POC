#!/usr/bin/env python3
"""Generate deterministic adaptive-continuation controller regression artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
from svg_plots import write_line_chart

from contact3d.adaptive import (
    AdaptiveContactOptions,
    AdaptiveLoadOptions,
    AdaptivePenaltyOptions,
    contact_penalties,
    solve_adaptive_contact_path,
)
from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.coupled import AugmentedContactOptions, AugmentedContactResult
from contact3d.enforcement_state import AugmentedLagrangeState


@dataclass(frozen=True, slots=True)
class Mesh:
    node_count: int = 2


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
        return replace(
            self,
            pair=replace(self.pair, normal_penalty=normal_penalty),
        )

    def reference_tributary_areas(self) -> np.ndarray:
        return np.ones(1)

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


def scripted_solver(
    problem,
    displacement,
    states,
    *,
    load_factor,
    options,
    tolerance,
):
    del displacement, options, tolerance
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


def _attempt_rows(result) -> list[dict[str, object]]:
    return [
        {
            "attempt": attempt.attempt,
            "start_load_factor": attempt.start_load_factor,
            "target_load_factor": attempt.target_load_factor,
            "step_size": attempt.step_size,
            "action": attempt.action,
            "inner_termination_reason": attempt.inner_termination_reason,
            "augmentations": attempt.augmentations,
            "newton_iterations": attempt.newton_iterations,
            "contact_event_restarts": attempt.contact_event_restarts,
            "equilibrium_residual": attempt.equilibrium_residual,
            "maximum_penetration": attempt.maximum_penetration,
            "penalties_before": attempt.penalties_before,
            "penalties_after": attempt.penalties_after,
            "path_values": attempt.path_values,
            "prescribed_dofs": attempt.prescribed_dofs,
            "prescribed_values": attempt.prescribed_values,
            "effective_load_norm": attempt.effective_load_norm,
            "reaction_norm": attempt.reaction_norm,
            "normalized_equilibrium_residual": (
                attempt.normalized_equilibrium_residual
            ),
            "normalized_maximum_penetration": (
                attempt.normalized_maximum_penetration
            ),
            "interface_penetrations": attempt.interface_penetrations,
            "normalized_interface_penetrations": (
                attempt.normalized_interface_penetrations
            ),
            "penalty_ratios_before": attempt.penalty_ratios_before,
            "penalty_ratios_after": attempt.penalty_ratios_after,
            "penalty_update_reasons": attempt.penalty_update_reasons,
        }
        for attempt in result.attempts
    ]


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
        augmented=AugmentedContactOptions(
            maximum_augmentations=3,
            gap_tolerance=1.0e-8,
        ),
    )
    artifacts = BenchmarkArtifactWriter(
        output,
        "adaptive-contact-policy",
        seed=0,
        solver_settings=options,
        repo_root=Path(__file__).resolve().parents[1],
    )
    result = solve_adaptive_contact_path(
        Problem(Mesh(), (Interface(Pair(100.0)),)),
        1.0,
        options=options,
        _solver=scripted_solver,
    )
    if not result.converged:
        raise RuntimeError(
            f"adaptive controller regression failed: {result.termination_reason}"
        )

    rows = _attempt_rows(result)
    accepted_attempts = [
        item for item in result.attempts if item.action == "accepted"
    ]
    accepted_rows = [
        {
            "accepted_step": index + 1,
            "start_load_factor": attempt.start_load_factor,
            "load_factor": attempt.target_load_factor,
            "step_size": attempt.step_size,
            "penalty": attempt.penalties_after[0],
            "newton_iterations": attempt.newton_iterations,
            "contact_event_restarts": attempt.contact_event_restarts,
            "equilibrium_residual": attempt.equilibrium_residual,
            "maximum_penetration": attempt.maximum_penetration,
            "effective_load_norm": attempt.effective_load_norm,
            "reaction_norm": attempt.reaction_norm,
            "normalized_equilibrium_residual": (
                attempt.normalized_equilibrium_residual
            ),
            "normalized_maximum_penetration": (
                attempt.normalized_maximum_penetration
            ),
        }
        for index, attempt in enumerate(accepted_attempts)
    ]
    summary = {
        "schema_version": "contact3d-adaptive-policy/v1",
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
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-adaptive-policy/v1",
    )
    artifacts.write_csv(
        "attempt-history.csv",
        rows,
        schema="contact3d-adaptive-attempts/v1",
    )
    artifacts.write_csv(
        "accepted-steps.csv",
        accepted_rows,
        schema="contact3d-adaptive-accepted-steps/v1",
    )

    x_values = np.arange(1, len(accepted_attempts) + 1, dtype=float)
    write_line_chart(
        output / "load-path.svg",
        title="Adaptive accepted load path",
        x_label="accepted step",
        y_label="load factor",
        x_values=x_values,
        series=(
            (
                np.array(
                    [item.target_load_factor for item in accepted_attempts]
                ),
                "load",
            ),
        ),
    )
    write_line_chart(
        output / "penalty-path.svg",
        title="Committed normal-penalty path",
        x_label="accepted step",
        y_label="normal penalty",
        x_values=x_values,
        series=(
            (
                np.array([item.penalties_after[0] for item in accepted_attempts]),
                "penalty",
            ),
        ),
    )
    for path in output.glob("*.svg"):
        ElementTree.parse(path)
        artifacts.register(path.name, "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "attempt-history.csv",
            "accepted-steps.csv",
            "load-path.svg",
            "penalty-path.svg",
        )
    )
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
