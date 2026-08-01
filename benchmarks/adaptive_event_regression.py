#!/usr/bin/env python3
"""Generate adaptive mixed-path topology-event propagation artifacts."""

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
)
from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.benchmark_plots import write_bar_chart, write_category_timeline
from contact3d.enforcement_state import AugmentedLagrangeState
from contact3d.event_solver import solve_event_aware_adaptive_contact_path
from contact3d.load_path import CoupledPathState
from contact3d.topology_events import (
    ContactTopologyEvent,
    ContactTopologyEventBatch,
    TopologyObservation,
)


@dataclass(frozen=True, slots=True)
class Mesh:
    node_count: int = 2


@dataclass(frozen=True, slots=True)
class Pair:
    normal_penalty: float = 100.0


@dataclass(frozen=True, slots=True)
class Interface:
    pair: Pair = Pair()

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
    mesh: Mesh = Mesh()
    interfaces: tuple[Interface, ...] = (Interface(),)

    def initial_states(self) -> tuple[AugmentedLagrangeState, ...]:
        return tuple(interface.initial_state() for interface in self.interfaces)


@dataclass(frozen=True, slots=True)
class Signature:
    facet_pairs: tuple[tuple[int, int], ...] = ((0, 0),)
    active_rows: tuple[bool, ...] = (True,)
    supported_rows: tuple[bool, ...] = (True,)


@dataclass(frozen=True, slots=True)
class MixedPath:
    def evaluate(self, problem: Problem, parameter: float) -> CoupledPathState:
        return CoupledPathState(
            parameter,
            problem,
            1.0,
            np.empty(0, dtype=np.int64),
            np.empty(0),
            np.empty(0),
            (
                ("tool_x", 0.08 * parameter),
                ("tool_z", -0.12 * parameter),
            ),
        )


def _batch(fraction: float, kind: str) -> ContactTopologyEventBatch:
    selected_fraction = fraction + 1.0e-8
    selected = TopologyObservation.valid(selected_fraction, (Signature(),), selected_fraction)
    event = ContactTopologyEvent(
        kind,
        0,
        (0, 0),
        fraction,
        "right",
        "scripted adaptive topology transition",
    )
    return ContactTopologyEventBatch(
        "restarted",
        fraction - 1.0e-8,
        fraction,
        selected_fraction,
        selected_fraction,
        "right",
        (event,),
        selected,
    )


def _result(
    load_factor: float,
    *,
    converged: bool,
    batch: ContactTopologyEventBatch,
) -> object:
    displacement = np.full(6, load_factor)
    state = AugmentedLagrangeState(np.array([load_factor]))
    evaluation = SimpleNamespace(
        maximum_penetration=0.0 if converged else 1.0e-3,
        free_residual_norm=1.0e-12 if converged else 2.0e-4,
    )
    equilibrium = SimpleNamespace(
        displacement=displacement,
        load_factor=load_factor,
        iteration_count=4 if converged else 10,
        contact_event_restarts=1,
        evaluation=evaluation,
        events=(batch,),
    )
    return SimpleNamespace(
        displacement=displacement,
        states=(state,),
        converged=converged,
        termination_reason="converged" if converged else "inner_equilibrium_failed",
        equilibrium=equilibrium,
        equilibria=(equilibrium,),
        history=(object(),),
    )


def run(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    settings = AdaptiveContactOptions(
        load=AdaptiveLoadOptions(
            initial_step=0.75,
            minimum_step=0.1,
            maximum_step=0.75,
            cutback_factor=0.5,
            easy_newton_iterations=0,
        ),
        penalty=AdaptivePenaltyOptions(enabled=False),
    )
    artifacts = BenchmarkArtifactWriter(
        output,
        "adaptive-topology-events",
        seed=0,
        solver_settings={
            "path": "scripted mixed displacement path",
            "adaptive": settings,
            "event_branch": "right",
        },
        repo_root=Path(__file__).resolve().parents[1],
    )
    calls = 0
    kinds = (
        "pair_entry",
        "clipping_vertex_edge",
        "pallet_transition",
        "pressure_activation",
    )

    def solver(problem, displacement, states, *, load_factor, options, tolerance):
        nonlocal calls
        del problem, displacement, states, options, tolerance
        kind = kinds[min(calls, len(kinds) - 1)]
        calls += 1
        return _result(
            load_factor,
            converged=calls != 1,
            batch=_batch(0.2 + 0.1 * calls, kind),
        )

    result = solve_event_aware_adaptive_contact_path(
        Problem(),
        1.0,
        path=MixedPath(),
        options=settings,
        _solver=solver,
    )
    if not result.converged:
        raise RuntimeError(result.termination_reason)

    rows = list(result.event_rows())
    metrics = {
        "converged": result.converged,
        "attempts": len(result.attempts),
        "accepted_steps": result.accepted_step_count,
        "cutbacks": result.cutback_count,
        "event_batches": result.contact_event_restarts,
        "atomic_events": len(rows),
        "accepted_event_batches": sum(
            record.action == "accepted" for record in result.event_batches
        ),
        "rejected_event_batches": sum(
            record.action != "accepted" for record in result.event_batches
        ),
        "all_right_branch": all(
            record.batch.selected_branch == "right" for record in result.event_batches
        ),
        "continuation_parameters": [
            record.continuation_parameter for record in result.event_batches
        ],
        "solver_load_factors": [
            record.solver_load_factor for record in result.event_batches
        ],
    }
    summary = {
        "schema_version": "contact3d-adaptive-topology-events/v1",
        "metrics": metrics,
        "events": rows,
    }
    artifacts.write_json(
        "summary.json",
        summary,
        schema="contact3d-adaptive-topology-events/v1",
    )
    artifacts.write_csv(
        "event-history.csv",
        rows,
        schema="contact3d-adaptive-topology-event-history/v1",
    )

    write_category_timeline(
        output / "continuation-events.svg",
        title="Adaptive topology events on the absolute mixed path",
        x_label="continuation parameter",
        categories=tuple(str(row["kind"]) for row in rows),
        x_values=np.asarray([float(row["continuation_parameter"]) for row in rows]),
        groups=tuple(str(row["action"]) for row in rows),
        emphasized_group="accepted",
    )
    by_attempt = {
        attempt.attempt: sum(int(row["attempt"]) == attempt.attempt for row in rows)
        for attempt in result.attempts
    }
    write_bar_chart(
        output / "events-per-attempt.svg",
        title="Localized event count by continuation attempt",
        y_label="atomic events",
        labels=tuple(str(attempt) for attempt in by_attempt),
        values=np.asarray(tuple(by_attempt.values()), dtype=float),
        annotations=tuple(
            next(item.action for item in result.attempts if item.attempt == attempt)
            for attempt in by_attempt
        ),
    )
    for path in output.glob("*.svg"):
        ElementTree.parse(path)
        artifacts.register(path.name, "svg")
    artifacts.finalize(
        required=(
            "summary.json",
            "event-history.csv",
            "continuation-events.svg",
            "events-per-attempt.svg",
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/adaptive-topology-events"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output)["metrics"], indent=2))


if __name__ == "__main__":
    main()
