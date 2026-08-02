from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
_load_module("rotating_blocks_solver", "rotating_blocks_solver.py")
REFINEMENT = _load_module(
    "rotating_blocks_refinement",
    "rotating_blocks_refinement.py",
)


def _accepted_rows(steps: int, perturbation: float) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "parameter": index / steps,
            "reaction_x": 2.0 * index / steps + perturbation,
            "reaction_y": 0.5 * index / steps,
            "reaction_z": -3.0 * index / steps,
            "maximum_pressure": 4.0 + index / steps + perturbation,
            "overlap_area": 0.75 + 0.1 * index / steps,
            "active_rows": 3,
            "supported_rows": 4,
            "facet_pairs": 5,
        }
        for index in range(steps + 1)
    )


def _run(profile, *, perturbation: float = 0.0, cutbacks: int = 0):
    events = (
        {
            "kind": "pair_entry",
            "entity": "1:2",
            "interface": 0,
            "continuation_parameter": 0.4 + perturbation,
        },
        {
            "kind": "pair_exit",
            "entity": "1:2",
            "interface": 0,
            "continuation_parameter": 0.7 + perturbation,
        },
    )
    return SimpleNamespace(
        profile=profile,
        passed=True,
        summary={
            "final_parameter": 1.0,
            "cutback_count": cutbacks,
        },
        accepted_rows=_accepted_rows(profile.requested_path_steps, perturbation),
        event_rows=events,
    )


def test_refinement_compares_three_requested_resolutions() -> None:
    requested: list[int] = []

    def runner(profile):
        requested.append(profile.requested_path_steps)
        perturbation = 1.0 / profile.requested_path_steps**2
        return _run(profile, perturbation=perturbation, cutbacks=profile.requested_path_steps // 16)

    result = REFINEMENT.run("quick", _runner=runner)

    assert result.passed
    assert requested == [8, 16, 32]
    assert result.summary["requested_steps"] == [8, 16, 32]
    assert result.summary["adaptive_cutbacks"] == [0, 1, 2]
    assert len(result.comparison_parameters) == 33
    assert result.comparison_rows[-1]["fine_reaction_z"] == pytest.approx(-3.0)
    assert result.summary["maximum_event_location_error"] == pytest.approx(
        1.0 / 16**2 - 1.0 / 32**2
    )
    json.dumps(result.summary, allow_nan=False)


def test_refinement_reports_event_count_and_field_failures() -> None:
    def runner(profile):
        completed = _run(profile)
        if profile.requested_path_steps == 16:
            completed.accepted_rows = _accepted_rows(16, 2.0)
            completed.event_rows = completed.event_rows[:1]
        return completed

    result = REFINEMENT.run("quick", raise_on_failure=False, _runner=runner)

    assert not result.passed
    assert not result.summary["criteria"]["medium_fine_fields_converged"]
    assert not result.summary["criteria"]["event_counts_match"]
    with pytest.raises(RuntimeError, match="medium_fine_fields_converged"):
        REFINEMENT.run("quick", _runner=runner)


def test_refinement_writes_machine_readable_tables_and_plot(tmp_path: Path) -> None:
    result = REFINEMENT.run("quick", _runner=lambda profile: _run(profile))

    REFINEMENT.write_results(tmp_path, result)

    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "field-comparison.csv").is_file()
    assert (tmp_path / "event-comparison.csv").is_file()
    assert (tmp_path / "refinement-error.svg").is_file()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["schema_version"] == REFINEMENT.SCHEMA


def test_refined_profile_keeps_model_and_uses_fixed_requested_increment() -> None:
    profile = REFINEMENT.rotating_blocks_execution_profile("full")
    refined = REFINEMENT._refined_profile(profile, 128)

    assert refined.name == "full"
    assert refined.model_profile == "full"
    assert refined.requested_path_steps == 128
    assert refined.initial_step == pytest.approx(1.0 / 128.0)
    assert refined.maximum_step == pytest.approx(1.0 / 128.0)
    assert refined.maximum_attempts >= 1024
