from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from contact3d.benchmark_artifacts import BenchmarkArtifactWriter
from contact3d.coupled import evaluate_coupled_equilibrium
from contact3d.enforcement_state import AugmentedLagrangeState

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))


def _load_module(name: str, filename: str):
    specification = importlib.util.spec_from_file_location(name, BENCHMARKS / filename)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


MODEL = _load_module("rotating_blocks_model", "rotating_blocks_model.py")
_load_module("rotating_blocks_profiles", "rotating_blocks_profiles.py")
_load_module("rotating_blocks_diagnostics", "rotating_blocks_diagnostics.py")
_load_module("rotating_blocks_solver", "rotating_blocks_solver.py")
_load_module("svg_plots", "svg_plots.py")
_load_module("rotating_blocks_refinement", "rotating_blocks_refinement.py")
PRESSURE = _load_module("rotating_blocks_pressure", "rotating_blocks_pressure.py")


def _step(model, parameter: float, states=None):
    path_state = model.path.evaluate(model.problem, parameter)
    displacement = np.zeros(3 * path_state.problem.mesh.node_count)
    displacement[path_state.prescribed_dofs] = path_state.prescribed_values
    selected_states = (
        tuple(path_state.problem.initial_states()) if states is None else tuple(states)
    )
    evaluation = evaluate_coupled_equilibrium(
        path_state.problem,
        displacement,
        selected_states,
        load_factor=path_state.solver_load_factor,
        assemble_tangent=False,
    )
    result = SimpleNamespace(
        displacement=evaluation.displacement,
        states=selected_states,
        equilibrium=SimpleNamespace(evaluation=evaluation),
    )
    return SimpleNamespace(
        parameter=parameter,
        path_state=path_state,
        result=result,
    )


def _run(model, parameters=(0.25, 0.625, 1.0)):
    return SimpleNamespace(
        result=SimpleNamespace(
            accepted_steps=tuple(_step(model, parameter) for parameter in parameters)
        ),
        event_rows=(),
    )


def _refinement(run):
    levels = (
        SimpleNamespace(requested_steps=16, run=run),
        SimpleNamespace(requested_steps=32, run=run),
    )
    return SimpleNamespace(
        levels=levels,
        comparison_parameters=(0.25, 0.625, 1.0),
        event_rows=(),
    )


def test_pressure_history_tracks_every_accepted_nodal_state() -> None:
    model = MODEL.build_rotating_blocks_model("quick")
    completed = _run(model)

    history = PRESSURE.collect_pressure_history(model, completed)

    slave_count = len(model.slave_nodes)
    assert len(history.aggregate_rows) == 3
    assert len(history.nodal_rows) == 3 * slave_count
    assert history.passed
    assert history.summary["maximum_resultant_relative_error"] < 1.0e-10
    assert history.summary["maximum_normalized_unsupported_pressure"] == 0.0
    assert history.summary["maximum_normalized_unsupported_multiplier"] == 0.0
    assert all(
        {
            "pressure",
            "multiplier",
            "normal_gap",
            "row_area",
            "supported",
            "active",
        }
        <= set(row)
        for row in history.nodal_rows
    )


def test_pressure_history_rejects_stale_unsupported_state() -> None:
    model = MODEL.build_rotating_blocks_model("quick")
    step = _step(model, 0.25)
    contact = step.result.equilibrium.evaluation.contacts[0]
    supported = list(contact.signature.supported_rows)
    supported[0] = False
    altered_contact = SimpleNamespace(
        pressure=contact.pressure,
        normal_gaps=contact.normal_gaps,
        residual=contact.residual,
        raw=contact.raw,
        signature=SimpleNamespace(
            supported_rows=tuple(supported),
            active_rows=contact.signature.active_rows,
        ),
    )
    altered_evaluation = SimpleNamespace(contacts=(altered_contact,))
    stale = np.asarray(step.result.states[0].multipliers).copy()
    stale[0] = 1.0
    altered_result = SimpleNamespace(
        displacement=step.result.displacement,
        states=(AugmentedLagrangeState(stale),),
        equilibrium=SimpleNamespace(evaluation=altered_evaluation),
    )
    altered_step = SimpleNamespace(
        parameter=step.parameter,
        path_state=step.path_state,
        result=altered_result,
    )
    completed = SimpleNamespace(
        result=SimpleNamespace(accepted_steps=(altered_step,)),
        event_rows=(),
    )

    history = PRESSURE.collect_pressure_history(model, completed)

    assert not history.passed
    assert not history.summary["criteria"]["unsupported_pressure_zero"]
    assert not history.summary["criteria"]["unsupported_multiplier_zero"]


def test_pressure_refinement_compares_nodal_and_aggregate_histories() -> None:
    model = MODEL.build_rotating_blocks_model("quick")
    completed = _run(model)

    comparison = PRESSURE.compare_pressure_refinement(
        model,
        _refinement(completed),
    )

    assert comparison.passed
    assert len(comparison.aggregate_rows) == 3
    assert len(comparison.nodal_rows) == 3 * len(model.slave_nodes)
    assert comparison.summary["discrete_state_mismatch_count"] == 0
    assert max(
        comparison.summary["maximum_relative_aggregate_errors"].values()
    ) == 0.0
    assert max(comparison.summary["maximum_relative_nodal_errors"].values()) == 0.0


def test_pressure_artifacts_validate_as_one_manifest(tmp_path: Path) -> None:
    model = MODEL.build_rotating_blocks_model("quick")
    completed = _run(model)
    refinement = _refinement(completed)
    writer = BenchmarkArtifactWriter(
        tmp_path,
        "rotating-pressure-test",
        seed=0,
        solver_settings={},
        repo_root=Path(__file__).resolve().parents[1],
    )

    artifacts = PRESSURE.write_pressure_artifacts(
        writer,
        tmp_path,
        model,
        completed,
        refinement,
    )
    manifest = writer.finalize(required=artifacts.required)

    assert artifacts.summary["passed"]
    paths = {record["path"] for record in manifest["artifacts"]}
    assert "tables/pressure-nodes.csv" in paths
    assert "tables/refinement-pressure-aggregates.csv" in paths
    assert "plots/pressure-centroid-history.svg" in paths
