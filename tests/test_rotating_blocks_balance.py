from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

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
_load_module("rotating_blocks_solver", "rotating_blocks_solver.py")
BALANCE = _load_module("rotating_blocks_balance", "rotating_blocks_balance.py")


def _balanced_inputs() -> tuple[np.ndarray, ...]:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    reaction = np.zeros((4, 3))
    applied = np.zeros((4, 3))
    applied[0] = np.array([1.0, 0.0, 0.0])
    reaction[0] = -applied[0]
    contact = np.asarray(
        [
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -2.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ]
    )
    slave = np.asarray([0, 1], dtype=np.int64)
    master = np.asarray([2, 3], dtype=np.int64)
    pivot = np.asarray([0.5, 0.0, 0.5])
    return coordinates, reaction, applied, contact, slave, master, pivot


def _row(
    contact: np.ndarray,
    master: np.ndarray,
    *,
    accepted_step: int = 1,
    parameter: float = 0.5,
) -> dict[str, object]:
    coordinates, reaction, applied, _, slave, _, pivot = _balanced_inputs()
    return BALANCE.evaluate_balance(
        coordinates,
        reaction,
        applied,
        contact,
        slave,
        master,
        pivot,
        accepted_step=accepted_step,
        parameter=parameter,
    )


def test_balanced_resultants_and_moments_pass() -> None:
    _, _, _, contact, _, master, _ = _balanced_inputs()

    row = _row(contact, master)
    summary = BALANCE.summarize_balance((row,))

    assert summary["passed"]
    assert all(float(row[field]) < 1.0e-15 for field in BALANCE.FORCE_FIELDS)
    assert all(float(row[field]) < 1.0e-15 for field in BALANCE.MOMENT_FIELDS)


def test_master_force_sign_error_fails_contact_balance() -> None:
    _, _, _, contact, _, master, _ = _balanced_inputs()
    corrupted = contact.copy()
    corrupted[2:] *= -1.0

    row = _row(corrupted, master, accepted_step=2, parameter=0.75)
    summary = BALANCE.summarize_balance((row,))

    assert not summary["passed"]
    assert float(row["normalized_contact_force_error"]) > 0.5
    assert summary["worst_force_state"]["accepted_step"] == 2


def test_master_node_mapping_error_fails_moment_balance() -> None:
    _, _, _, contact, _, master, _ = _balanced_inputs()
    swapped = master[::-1].copy()

    row = _row(contact, swapped, accepted_step=3, parameter=1.0)
    summary = BALANCE.summarize_balance((row,))

    assert float(row["normalized_contact_force_error"]) < 1.0e-15
    assert float(row["normalized_contact_moment_origin_error"]) > 0.1
    assert not summary["passed"]
    assert summary["worst_moment_state"]["accepted_step"] == 3


def test_force_and_moment_tolerances_are_independent() -> None:
    _, _, _, contact, _, master, _ = _balanced_inputs()
    row = _row(contact, master)
    row["normalized_global_force_error"] = 2.0e-6
    row["normalized_global_moment_origin_error"] = 3.0e-5

    summary = BALANCE.summarize_balance(
        (row,),
        force_tolerance=1.0e-5,
        moment_tolerance=1.0e-4,
    )

    assert summary["passed"]
    assert summary["force_tolerance"] == 1.0e-5
    assert summary["moment_tolerance"] == 1.0e-4
