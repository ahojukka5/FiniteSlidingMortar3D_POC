from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np

from contact3d.benchmark_artifacts import validate_benchmark_manifest
from contact3d.coupled import evaluate_coupled_equilibrium

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
PROFILES = _load_module("rotating_blocks_profiles", "rotating_blocks_profiles.py")
_load_module("rotating_blocks_diagnostics", "rotating_blocks_diagnostics.py")
_load_module("rotating_blocks_solver", "rotating_blocks_solver.py")
_load_module("svg_plots", "svg_plots.py")
BALANCE = _load_module("rotating_blocks_balance", "rotating_blocks_balance.py")
_load_module("rotating_blocks_refinement", "rotating_blocks_refinement.py")
_load_module("rotating_blocks_pressure", "rotating_blocks_pressure.py")
BUNDLE = _load_module("rotating_blocks_bundle", "rotating_blocks_bundle.py")


def _reaction(evaluation) -> np.ndarray:
    residual = np.asarray(evaluation.residual, dtype=float)
    values = np.zeros_like(residual)
    constrained = np.ones(len(residual), dtype=bool)
    constrained[np.asarray(evaluation.free_dofs, dtype=np.int64)] = False
    values[constrained] = residual[constrained]
    return values


def _step(model, parameter: float, *, evaluation_parameter: float | None = None):
    path_state = model.path.evaluate(model.problem, parameter)
    evaluated_state = model.path.evaluate(
        model.problem,
        parameter if evaluation_parameter is None else evaluation_parameter,
    )
    displacement = np.zeros(3 * evaluated_state.problem.mesh.node_count, dtype=float)
    displacement[evaluated_state.prescribed_dofs] = evaluated_state.prescribed_values
    states = tuple(evaluated_state.problem.initial_states())
    evaluation = evaluate_coupled_equilibrium(
        evaluated_state.problem,
        displacement,
        states,
        load_factor=evaluated_state.solver_load_factor,
        assemble_tangent=False,
    )
    result = SimpleNamespace(
        displacement=evaluation.displacement,
        states=states,
        equilibrium=SimpleNamespace(evaluation=evaluation),
        converged=True,
        termination_reason="converged",
    )
    reaction = _reaction(evaluation)
    return SimpleNamespace(
        parameter=parameter,
        path_state=path_state,
        result=result,
        reaction=reaction,
        reaction_norm=float(np.linalg.norm(reaction)),
    )


def _accepted_row(model, index: int, step) -> dict[str, object]:
    contact = step.result.equilibrium.evaluation.contacts[0]
    reaction = step.reaction.reshape((-1, 3))[model.controlled_nodes].sum(axis=0)
    value = step.path_state.value
    return {
        "accepted_step": index,
        "parameter": step.parameter,
        "phase_index": int(round(value("phase_index"))),
        "phase_parameter": float(value("phase_parameter")),
        "rotation_angle": float(value("rotation_angle")),
        "reaction_norm": step.reaction_norm,
        "reaction_x": float(reaction[0]),
        "reaction_y": float(reaction[1]),
        "reaction_z": float(reaction[2]),
        "inner_converged": True,
        "inner_termination_reason": "converged",
        "maximum_pressure": float(np.max(contact.pressure, initial=0.0)),
        "overlap_area": float(contact.raw.contact.weights.total_area),
        "active_rows": int(np.count_nonzero(contact.signature.active_rows)),
        "supported_rows": int(np.count_nonzero(contact.signature.supported_rows)),
        "facet_pairs": len(contact.signature.facet_pairs),
    }


def _fixtures():
    profile = PROFILES.rotating_blocks_execution_profile("quick")
    model = BUNDLE.build_rotating_blocks_model(profile.model_profile)
    steps = (
        _step(model, 0.25),
        _step(model, 0.625),
        _step(model, 1.0, evaluation_parameter=0.625),
    )
    accepted = tuple(
        _accepted_row(model, index, step)
        for index, step in enumerate(steps, start=1)
    )
    attempts = tuple(
        {
            "attempt": index,
            "start_parameter": 0.0 if index == 1 else steps[index - 2].parameter,
            "target_parameter": step.parameter,
            "step_size": 0.25,
            "action": "accepted",
            "inner_termination_reason": "converged",
            "augmentations": 1,
            "newton_iterations": 1,
            "contact_event_restarts": 1,
            "normalized_equilibrium_residual": 0.0,
            "normalized_maximum_penetration": 0.0,
            "diagnostics_complete": True,
        }
        for index, step in enumerate(steps, start=1)
    )
    diagnostics = tuple(
        {
            "attempt": index,
            "action": "accepted",
            "newton_iterations": 1,
            "linear_iterations": 1,
            "event_localization_batches": 1,
            "dense_materializations": 0,
        }
        for index in range(1, 4)
    )
    event_rows = tuple(
        {
            "attempt": index,
            "action": "accepted",
            "start_parameter": max(0.0, parameter - 0.1),
            "target_parameter": parameter,
            "continuation_parameter": parameter,
            "solver_load_factor": 1.0,
            "augmentation": 0,
            "batch": index - 1,
            "event": 0,
            "kind": kind,
            "interface": 0,
            "entity": "0:0",
            "left_newton_fraction": 0.4,
            "event_newton_fraction": 0.5,
            "right_newton_fraction": 0.6,
            "selected_newton_fraction": 0.6,
            "selected_branch": "right",
            "path_values": "",
            "detail": "test",
        }
        for index, (parameter, kind) in enumerate(
            (
                (0.30, "pair_entry"),
                (0.45, "pair_exit"),
                (0.70, "pair_entry"),
                (0.85, "pair_exit"),
            ),
            start=1,
        )
    )
    completed = SimpleNamespace(
        profile=profile,
        result=SimpleNamespace(accepted_steps=steps),
        summary={"schema_version": "test-solver/v1", "passed": True},
        accepted_rows=accepted,
        attempt_rows=attempts,
        event_rows=event_rows,
        diagnostic_rows=diagnostics,
        passed=True,
    )
    comparison_rows = tuple(
        {
            "parameter": value,
            "medium_reaction_x": value,
            "fine_reaction_x": value,
            "relative_error_reaction_x": 0.0,
        }
        for value in (0.0, 0.5, 1.0)
    )
    refinement_events = (
        {
            "kind": "pair_entry",
            "entity": "0:0",
            "interface": 0,
            "occurrence": 0,
            "medium_parameter": 0.3,
            "fine_parameter": 0.3,
            "absolute_error": 0.0,
        },
    )
    levels = (
        SimpleNamespace(requested_steps=16, run=completed),
        SimpleNamespace(requested_steps=32, run=completed),
    )
    refinement = SimpleNamespace(
        levels=levels,
        comparison_parameters=(0.25, 0.625, 1.0),
        comparison_rows=comparison_rows,
        event_rows=refinement_events,
        summary={
            "schema_version": "test-refinement/v1",
            "passed": True,
            "requested_steps": [8, 16, 32],
        },
        passed=True,
    )
    return model, completed, refinement


def _balance(completed) -> object:
    rows = tuple(
        {
            "accepted_step": index,
            "parameter": float(row["parameter"]),
            "normalized_global_force_error": 0.0,
            "normalized_contact_force_error": 0.0,
            "normalized_global_moment_origin_error": 0.0,
            "normalized_global_moment_pivot_error": 0.0,
            "normalized_contact_moment_origin_error": 0.0,
            "normalized_contact_moment_pivot_error": 0.0,
        }
        for index, row in enumerate(completed.accepted_rows, start=1)
    )
    return BALANCE.RotatingBlocksBalance(rows, BALANCE.summarize_balance(rows))


def test_checkpoint_selection_covers_required_physical_regimes() -> None:
    model, completed, _ = _fixtures()

    checkpoints = BUNDLE.select_checkpoints(model, completed)

    assert [checkpoint.name for checkpoint in checkpoints] == [
        "pre-contact",
        "compressed",
        "mid-rotation",
        "final",
    ]
    assert [checkpoint.parameter for checkpoint in checkpoints] == [
        0.0,
        0.25,
        0.625,
        1.0,
    ]


def test_bundle_writes_valid_manifest_tables_vtk_and_plots(tmp_path: Path) -> None:
    model, completed, refinement = _fixtures()

    summary = BUNDLE.write_bundle(
        tmp_path,
        model,
        completed,
        refinement,
        balance=_balance(completed),
    )

    assert summary["passed"]
    assert summary["balance_summary"]["passed"]
    assert summary["pressure_summary"]["passed"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    validate_benchmark_manifest(manifest, root=tmp_path)
    paths = {record["path"] for record in manifest["artifacts"]}
    assert "tables/interface-rows.csv" in paths
    assert "tables/refinement-fields.csv" in paths
    assert "tables/force-moment-balance.csv" in paths
    assert "tables/pressure-nodes.csv" in paths
    assert "tables/pressure-aggregates.csv" in paths
    assert "tables/refinement-pressure-nodes.csv" in paths
    assert "tables/refinement-pressure-aggregates.csv" in paths
    assert "plots/pressure-redistribution.svg" in paths
    assert "plots/pressure-centroid-history.svg" in paths
    assert "plots/pressure-refinement-errors.svg" in paths
    assert "plots/force-balance.svg" in paths
    assert "plots/moment-balance.svg" in paths
    assert "plots/balance-worst-states.svg" in paths
    assert "checkpoints/03-final/volume.vtu" in paths
    assert "checkpoints/03-final/projected-overlap.vtp" in paths

    slave = ElementTree.parse(
        tmp_path / "checkpoints/03-final/slave-contact.vtp"
    ).getroot()
    names = {element.attrib.get("Name") for element in slave.iter("DataArray")}
    assert {
        "normal_gap",
        "pressure",
        "multiplier",
        "supported",
        "active",
        "contact_force",
    } <= names

    master = ElementTree.parse(
        tmp_path / "checkpoints/03-final/master-contact.vtp"
    ).getroot()
    master_names = {
        element.attrib.get("Name") for element in master.iter("DataArray")
    }
    assert {"contact_force", "overlap_area"} <= master_names

    projected = ElementTree.parse(
        tmp_path / "checkpoints/03-final/projected-overlap.vtp"
    ).getroot()
    projected_names = {
        element.attrib.get("Name") for element in projected.iter("DataArray")
    }
    assert {"region_kind", "pair_index", "projected_area"} <= projected_names
