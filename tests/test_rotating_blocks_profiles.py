from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


PROFILES = _load_module("rotating_blocks_profiles", "rotating_blocks_profiles.py")
MODEL = _load_module("rotating_blocks_model_profiles", "rotating_blocks_model.py")


def test_named_profiles_are_deterministic_and_serializable() -> None:
    quick = PROFILES.rotating_blocks_execution_profile("quick")
    full = PROFILES.rotating_blocks_execution_profile("full")

    assert quick is PROFILES.QUICK_PROFILE
    assert full is PROFILES.FULL_PROFILE
    assert quick.as_dict() == quick.as_dict()
    assert full.as_dict() == full.as_dict()
    assert quick.name == quick.model_profile == "quick"
    assert full.name == full.model_profile == "full"


def test_profiles_preserve_geometry_motion_and_event_semantics() -> None:
    quick_profile = PROFILES.rotating_blocks_execution_profile("quick")
    full_profile = PROFILES.rotating_blocks_execution_profile("full")
    quick = MODEL.build_rotating_blocks_model(quick_profile.model_profile)
    full = MODEL.build_rotating_blocks_model(full_profile.model_profile)

    assert quick.geometry == full.geometry
    assert quick.path.end_parameter == full.path.end_parameter == 1.0
    quick_final = quick.path.evaluate(quick.problem, 1.0)
    full_final = full.path.evaluate(full.problem, 1.0)
    assert quick_final.values == full_final.values
    assert np.isclose(dict(quick_final.values)["rotation_angle"], 0.5 * np.pi)
    assert np.isclose(dict(full_final.values)["rotation_angle"], 0.5 * np.pi)


def test_quick_profile_is_bounded_but_retains_evidence_paths() -> None:
    quick = PROFILES.QUICK_PROFILE

    assert quick.requested_path_steps == 16
    assert quick.topology_samples == 65
    assert quick.maximum_attempts == 128
    assert quick.repetition_runs == 2
    assert quick.refinement_steps == (8, 16, 32)
    assert not quick.export_checkpoints
    assert not quick.run_refinement


def test_full_profile_enables_publication_evidence() -> None:
    full = PROFILES.FULL_PROFILE

    assert full.requested_path_steps == 64
    assert full.topology_samples == 129
    assert full.maximum_attempts == 1024
    assert full.repetition_runs == 2
    assert full.refinement_steps == (32, 64, 128)
    assert full.export_checkpoints
    assert full.run_refinement


def test_profile_validation_rejects_inconsistent_settings() -> None:
    with pytest.raises(ValueError, match="must match"):
        PROFILES.RotatingBlocksExecutionProfile(
            name="quick",
            model_profile="full",
            requested_path_steps=16,
            topology_samples=65,
            maximum_attempts=128,
            initial_step=1.0 / 16.0,
            minimum_step=1.0 / 1024.0,
            maximum_step=1.0 / 8.0,
            repetition_runs=2,
            refinement_steps=(8, 16, 32),
            export_checkpoints=False,
            run_refinement=False,
        )

    with pytest.raises(ValueError, match="quick.*full"):
        PROFILES.rotating_blocks_execution_profile("unknown")
