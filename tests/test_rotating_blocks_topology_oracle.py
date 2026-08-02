from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from contact3d.topology_scan import scan_kinematic_contact_path

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
ORACLE = _load_module(
    "rotating_blocks_topology_oracle",
    "rotating_blocks_topology_oracle.py",
)


@pytest.fixture(scope="module")
def oracle_outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("rotating-topology-oracle")
    first = root / "first"
    second = root / "second"
    first_summary = ORACLE.run(first, profile="quick", sample_count=65)
    second_summary = ORACLE.run(second, profile="quick", sample_count=65)
    return first, second, first_summary, second_summary


def test_oracle_contains_repeated_deterministic_pair_transitions(oracle_outputs) -> None:
    first, second, first_summary, second_summary = oracle_outputs
    metrics = first_summary["metrics"]
    assert metrics["pair_entries"] >= 2
    assert metrics["pair_exits"] >= 2
    assert metrics["transition_intervals"] > 0
    assert metrics["minimum_overlap_area"] > 0.0
    assert metrics["maximum_overlap_area"] >= metrics["minimum_overlap_area"]
    assert metrics["signature_digest"] == second_summary["metrics"]["signature_digest"]

    first_expected = json.loads((first / "expected-transitions.json").read_text())
    second_expected = json.loads((second / "expected-transitions.json").read_text())
    assert first_expected == second_expected
    assert first_expected["signature_digest"] == metrics["signature_digest"]
    assert len(first_expected["transitions"]) == metrics["transition_intervals"]


def test_oracle_artifacts_are_complete_and_solver_independent(oracle_outputs) -> None:
    first, _, summary, _ = oracle_outputs
    expected = {
        "summary.json",
        "expected-transitions.json",
        "sample-history.csv",
        "transition-history.csv",
        "overlap-area.svg",
        "topology-counts.svg",
        "transition-timeline.svg",
        "manifest.json",
    }
    assert expected <= {path.name for path in first.iterdir()}

    manifest = json.loads((first / "manifest.json").read_text())
    settings = manifest["provenance"]["solver_settings"]
    assert settings["nonlinear_solver"] is None
    assert settings["sample_count"] == 65

    with (first / "sample-history.csv").open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == summary["metrics"]["sample_count"]
    assert {row["phase"] for row in samples} == {"compression", "rotation"}
    assert all(json.loads(row["facet_pairs"]) for row in samples)

    with (first / "transition-history.csv").open(newline="") as stream:
        transitions = list(csv.DictReader(stream))
    assert len(transitions) == summary["metrics"]["transition_intervals"]
    assert sum(int(row["pair_entries"]) for row in transitions) >= 2
    assert sum(int(row["pair_exits"]) for row in transitions) >= 2
    assert all(float(row["right_parameter"]) > float(row["left_parameter"]) for row in transitions)

    for name in ("overlap-area.svg", "topology-counts.svg", "transition-timeline.svg"):
        ElementTree.parse(first / name)


def test_scan_rejects_nonincreasing_parameters() -> None:
    model = MODEL.build_rotating_blocks_model("quick")
    with pytest.raises(ValueError, match="strictly increasing"):
        scan_kinematic_contact_path(
            model.problem,
            model.path,
            np.array([0.0, 0.5, 0.5]),
        )
