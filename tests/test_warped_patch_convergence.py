from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

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


CONVERGENCE = _load_module(
    "warped_patch_convergence",
    "warped_patch_convergence.py",
)


def _fake_case(
    level: str,
    *,
    surface_family: str,
    bias_side: str,
    publication: bool,
):
    del publication
    level_index = {"coarse": 0, "medium": 1, "fine": 2}[level]
    characteristic_size = (0.5, 1.0 / 3.0, 0.25)[level_index]
    error = (0.08, 0.04, 0.02)[level_index]
    bias_shift = 0.0 if bias_side == "lower" else 0.005
    case_id = f"{surface_family}-{bias_side}-{level}"

    nodes = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        dtype=float,
    )
    facets = (np.asarray((0, 1, 2, 3), dtype=np.int64),)
    pressure = np.full(4, 1.0 + bias_shift)
    gaps = np.zeros(4)
    contact = SimpleNamespace(
        pressure=pressure,
        normal_gaps=gaps,
        signature=SimpleNamespace(
            active_rows=(True,) * 4,
            supported_rows=(True,) * 4,
        ),
    )
    solved = SimpleNamespace(
        displacement=np.zeros(12),
        states=(SimpleNamespace(multipliers=pressure.copy()),),
        equilibrium=SimpleNamespace(
            evaluation=SimpleNamespace(contacts=(contact,)),
        ),
    )
    result = SimpleNamespace(
        accepted_steps=(SimpleNamespace(result=solved),),
    )
    interface = SimpleNamespace(
        slave_nodes=np.arange(4, dtype=np.int64),
        pair=SimpleNamespace(
            slave=SimpleNamespace(reference_nodes=nodes, facets=facets),
        ),
    )
    model = SimpleNamespace(
        interface=interface,
        profile=SimpleNamespace(name=level),
        surface_family=SimpleNamespace(name=surface_family),
        bias_side=bias_side,
    )
    interface_rows = tuple(
        {
            "case_id": case_id,
            "profile": level,
            "surface_family": surface_family,
            "bias_side": bias_side,
            "slave_row": row,
            "global_node": row,
            "x": float(nodes[row, 0]),
            "y": float(nodes[row, 1]),
            "z": float(nodes[row, 2]),
            "row_area": 0.25,
            "normal_gap": 0.0,
            "pressure": float(pressure[row]),
            "multiplier": float(pressure[row]),
            "active": True,
            "supported": True,
        }
        for row in range(4)
    )
    metrics = {
        "case_id": case_id,
        "profile": level,
        "surface_family": surface_family,
        "bias_side": bias_side,
        "characteristic_size": characteristic_size,
        "converged": True,
        "final_parameter": 1.0,
        "normalized_equilibrium_residual": 1.0e-10,
        "normalized_maximum_penetration": 1.0e-10,
        "contact_force_balance_relative": 1.0e-12,
        "partition_error": 1.0e-12,
        "displacement_relative_l2_error": error,
        "reaction_relative_error": 0.75 * error,
        "pressure_relative_l2_error": 1.5 * error,
        "gap_weighted_l2": characteristic_size * 1.0e-6,
        "gap_over_h": 1.0e-6,
        "overlap_area_error": 0.25 * error,
        "reference_reaction": 1.0,
        "reaction": 1.0 + bias_shift,
        "maximum_pressure": 1.0 + bias_shift,
        "overlap_area": 1.0,
        "topology_events": 1,
    }
    return SimpleNamespace(
        case_id=case_id,
        model=model,
        result=result,
        metrics=metrics,
        interface_rows=interface_rows,
        attempt_rows=(),
        event_rows=(),
    )


def test_observed_rate_handles_refinement_and_exact_error() -> None:
    np.testing.assert_allclose(
        CONVERGENCE.observed_rate(0.25, 0.0625, 0.5, 0.25),
        2.0,
    )
    assert CONVERGENCE.observed_rate(0.0, 0.0, 0.5, 0.25) is None


def test_quick_campaign_writes_complete_evidence(tmp_path: Path) -> None:
    summary = CONVERGENCE.run(
        tmp_path,
        profile="quick",
        _solve_case=_fake_case,
    )

    assert summary["schema_version"] == "contact3d-warped-patch-convergence/v1"
    assert summary["case_count"] == 6
    assert summary["passed"] is True
    gate = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert gate["passed"] is True
    assert not gate["failed_criteria"]

    with (tmp_path / "levels.csv").open(encoding="utf-8", newline="") as stream:
        levels = list(csv.DictReader(stream))
    with (tmp_path / "rates.csv").open(encoding="utf-8", newline="") as stream:
        rates = list(csv.DictReader(stream))
    with (tmp_path / "bias.csv").open(encoding="utf-8", newline="") as stream:
        bias = list(csv.DictReader(stream))
    assert len(levels) == 6
    assert len(rates) == 10
    assert len(bias) == 3
    assert {row["profile"] for row in levels} == {"coarse", "medium", "fine"}
    assert {row["bias_side"] for row in levels} == {"lower", "upper"}

    for row in levels:
        case = tmp_path / "cases" / row["case_id"]
        assert (case / "interface.csv").is_file()
        ElementTree.parse(case / "interface.vtp")
        ElementTree.parse(case / "pressure.svg")
        ElementTree.parse(case / "gap.svg")

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    registered = {item["path"] for item in manifest["artifacts"]}
    assert {
        "summary.json",
        "gate.json",
        "levels.csv",
        "rates.csv",
        "bias.csv",
    } <= registered
    assert len([path for path in registered if path.endswith("interface.vtp")]) == 6
    assert len([path for path in registered if path.startswith("convergence/")]) == 3
