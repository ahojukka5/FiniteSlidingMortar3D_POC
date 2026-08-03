from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))


def _load_module(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, BENCHMARKS / filename)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


_load_module("rotating_blocks_model", "rotating_blocks_model.py")
_load_module("rotating_blocks_profiles", "rotating_blocks_profiles.py")
_load_module("rotating_blocks_diagnostics", "rotating_blocks_diagnostics.py")
SOLVER = _load_module("rotating_blocks_solver", "rotating_blocks_solver.py")


def _path_state(parameter: float) -> SimpleNamespace:
    values = {
        "phase_index": 1.0,
        "phase_parameter": parameter,
        "rotation_angle": 0.5 * np.pi * parameter,
    }
    return SimpleNamespace(value=values.__getitem__)


def _attempt(index: int, action: str, parameter: float) -> SimpleNamespace:
    return SimpleNamespace(
        attempt=index,
        start_parameter=max(0.0, parameter - 0.1),
        target_parameter=parameter,
        step_size=0.1,
        action=action,
        inner_termination_reason="converged",
        augmentations=2,
        newton_iterations=4,
        contact_event_restarts=1,
        normalized_equilibrium_residual=2.0e-10,
        normalized_maximum_penetration=3.0e-9,
    )


def _attempt_result(index: int) -> SimpleNamespace:
    linear = SimpleNamespace(
        requested_backend="sparse_lu",
        backend="sparse_lu",
        preconditioner="none",
        converged=True,
        iterations=1,
        matrix_nnz=100 + index,
        materialized_dense=False,
        setup_seconds=0.01 * index,
        solve_seconds=0.02 * index,
    )
    event = SimpleNamespace(events=(SimpleNamespace(),))
    equilibrium = SimpleNamespace(
        history=(
            SimpleNamespace(
                line_search_iterations=index,
                linear_solve=linear,
            ),
        ),
        linear_solve_failure=None,
        events=(event,),
    )
    return SimpleNamespace(equilibria=(equilibrium,))


def _result(*, converged: bool = True) -> SimpleNamespace:
    inner = SimpleNamespace(converged=True, termination_reason="converged")
    accepted_steps = (
        SimpleNamespace(
            parameter=0.5,
            path_state=_path_state(0.5),
            reaction_norm=3.0,
            result=inner,
        ),
        SimpleNamespace(
            parameter=1.0,
            path_state=_path_state(1.0),
            reaction_norm=4.0,
            result=inner,
        ),
    )
    attempts = (
        _attempt(1, "cutback", 0.5),
        _attempt(2, "accepted", 0.5),
        _attempt(3, "penalty_increase", 1.0),
        _attempt(4, "accepted", 1.0),
    )
    event_rows = (
        {
            "kind": "clipping_vertex_edge",
            "interface": 0,
            "entity": "0:1:2",
            "continuation_parameter": 0.5,
        },
        {
            "kind": "pallet_transition",
            "interface": 0,
            "entity": "0:1:2",
            "continuation_parameter": 0.75,
        },
    )
    return SimpleNamespace(
        accepted_steps=accepted_steps,
        attempts=attempts,
        event_batches=(),
        attempt_results=tuple(_attempt_result(index) for index in range(1, 5)),
        event_rows=lambda: event_rows,
        converged=converged,
        termination_reason="converged" if converged else "maximum_attempts",
        load_factor=1.0 if converged else 0.75,
        cutback_count=1,
        penalty_update_count=1,
    )


def test_profiles_map_to_scale_aware_production_options() -> None:
    quick = SOLVER.solver_options(
        SOLVER.rotating_blocks_execution_profile("quick")
    )
    full = SOLVER.solver_options(SOLVER.rotating_blocks_execution_profile("full"))

    assert quick.scaling.enabled
    assert full.scaling.enabled
    assert quick.augmented.event_policy == "restart"
    assert quick.load.initial_step == pytest.approx(1.0 / 16.0)
    assert full.load.initial_step == pytest.approx(1.0 / 64.0)
    assert quick.load.maximum_attempts == 128
    assert full.load.maximum_attempts == 1024
    assert quick.augmented.maximum_augmentations == 32
    assert full.augmented.maximum_augmentations == 48
    assert quick.load.easy_newton_iterations == 32
    assert full.load.easy_newton_iterations == 48
    assert quick.penalty.interface_local
    assert quick.penalty.normalized_penetration_target == pytest.approx(1.0e-7)
    assert quick.scaling.gap_tolerance == pytest.approx(1.0e-7)
    assert quick.scaling.complementarity_tolerance == pytest.approx(1.0e-7)
    assert quick.scaling.projection_tolerance == pytest.approx(1.0e-5)
    assert quick.scaling.projection_tolerance > quick.scaling.gap_tolerance
    assert quick.augmented.projection_tolerance == pytest.approx(1.0e-5)
    assert quick.augmented.newton.linear_solver.backend == "auto"
    assert full.augmented.newton.linear_solver.backend == "sparse_lu"


def test_topology_count_deduplicates_retry_records() -> None:
    duplicate = {
        "kind": "clipping_vertex_edge",
        "interface": 0,
        "entity": "0:1:2",
        "continuation_parameter": 0.5,
    }
    counts = SOLVER._event_counts((duplicate, dict(duplicate)))
    assert counts["event_count"] == 2
    assert counts["unique_transition_count"] == 1

    distinct = dict(duplicate, continuation_parameter=0.75)
    counts = SOLVER._event_counts((duplicate, distinct))
    assert counts["unique_transition_count"] == 2


@pytest.mark.parametrize("profile", ("quick", "full"))
def test_run_uses_model_path_and_returns_refinement_ready_rows(profile: str) -> None:
    calls: list[tuple[object, float, object, object]] = []

    def fake_solver(problem, target, *, path, options):
        calls.append((problem, target, path, options))
        return _result()

    completed = SOLVER.run(profile, _solver=fake_solver)

    assert completed.passed
    assert completed.summary["profile"] == profile
    assert completed.summary["final_parameter"] == pytest.approx(1.0)
    assert completed.summary["accepted_step_count"] == 2
    assert completed.summary["events"]["pair_entries"] == 0
    assert completed.summary["events"]["pair_exits"] == 0
    assert completed.summary["events"]["unique_transition_count"] == 2
    assert completed.summary["criteria"]["repeated_topology_transitions"]
    assert completed.accepted_rows[-1]["rotation_angle"] == pytest.approx(0.5 * np.pi)
    assert [row["action"] for row in completed.attempt_rows] == [
        "cutback",
        "accepted",
        "penalty_increase",
        "accepted",
    ]
    diagnostics = completed.summary["solver_diagnostics"]
    assert diagnostics["diagnostics_complete"]
    assert diagnostics["deterministic_counts"]["linear_solves"] == 4
    assert diagnostics["deterministic_counts"]["linear_iterations"] == 4
    assert diagnostics["deterministic_counts"]["event_localization_batches"] == 4
    assert diagnostics["worst_accepted_attempt"]["attempt"] == 4
    assert diagnostics["worst_failed_attempt"]["attempt"] == 3
    assert not diagnostics["timings_used_for_acceptance"]
    assert completed.attempt_rows[0]["maximum_matrix_nnz"] == 101
    assert len(calls) == 1
    assert calls[0][1] == pytest.approx(1.0)
    assert calls[0][2].end_parameter == pytest.approx(1.0)
    json.dumps(completed.summary, allow_nan=False)


def test_failed_production_criteria_report_all_relevant_fields() -> None:
    completed = SOLVER.run(
        "quick",
        raise_on_failure=False,
        _solver=lambda *args, **kwargs: _result(converged=False),
    )

    assert not completed.passed
    assert not completed.summary["criteria"]["solver_converged"]
    assert not completed.summary["criteria"]["final_motion_reached"]
    with pytest.raises(RuntimeError, match="solver_converged"):
        SOLVER.run(
            "quick",
            _solver=lambda *args, **kwargs: _result(converged=False),
        )
