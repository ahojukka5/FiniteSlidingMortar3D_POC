from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from contact3d.repetition import (
    RepetitionTolerances,
    compare_kinematic_topology_scans,
)
from contact3d.topology_model import ContactTopologySignature
from contact3d.topology_scan import (
    KinematicContactTopologySample,
    KinematicTopologyChange,
    KinematicTopologyFrame,
    KinematicTopologyScan,
    KinematicTopologyTransition,
)


def _signature(*, vertices: int = 4) -> ContactTopologySignature:
    return ContactTopologySignature(
        ((0, 0),),
        (True, False),
        (True, True),
        ((0, 0, vertices, vertices, 1),),
    )


def _scan(*, area_shift: float = 0.0, vertices: int = 4) -> KinematicTopologyScan:
    frames = tuple(
        KinematicTopologyFrame(
            parameter=parameter,
            phase="rotation",
            phase_parameter=parameter,
            displacement=np.zeros(6),
            contacts=(
                KinematicContactTopologySample(
                    interface=0,
                    signature=_signature(vertices=vertices),
                    overlap_area=1.0 + parameter + area_shift,
                    supported_rows=(0, 1),
                    active_rows=(0,),
                    maximum_pressure=2.0 + parameter,
                ),
            ),
        )
        for parameter in (0.25, 0.5)
    )
    transitions = (
        KinematicTopologyTransition(
            0.25,
            0.5,
            (KinematicTopologyChange("clipping_vertex_edge", 0, (0, 0), "test"),),
        ),
    )
    return KinematicTopologyScan(frames, transitions, 1.0e-12)


def test_equal_scans_pass_with_zero_error() -> None:
    comparison = compare_kinematic_topology_scans(_scan(), _scan())

    assert comparison.passed
    assert comparison.divergence is None
    assert comparison.frame_count == 2
    assert comparison.transition_count == 1
    assert comparison.maximum_absolute_error == 0.0


def test_small_numeric_difference_uses_tolerances() -> None:
    comparison = compare_kinematic_topology_scans(
        _scan(),
        _scan(area_shift=5.0e-13),
        tolerances=RepetitionTolerances(absolute=1.0e-12, relative=0.0),
    )

    assert comparison.passed
    assert comparison.maximum_absolute_error == pytest.approx(5.0e-13)


def test_numeric_divergence_names_field_and_path_state() -> None:
    comparison = compare_kinematic_topology_scans(
        _scan(),
        _scan(area_shift=1.0e-5),
        tolerances=RepetitionTolerances(absolute=1.0e-12, relative=1.0e-12),
    )

    assert not comparison.passed
    assert comparison.divergence is not None
    assert comparison.divergence.field == "overlap_area"
    assert comparison.divergence.frame == 0
    assert comparison.divergence.parameter == 0.25
    assert comparison.divergence.interface == 0
    assert comparison.divergence.absolute_error == pytest.approx(1.0e-5)


def test_discrete_signature_difference_is_exact() -> None:
    comparison = compare_kinematic_topology_scans(_scan(), _scan(vertices=5))

    assert not comparison.passed
    assert comparison.divergence is not None
    assert comparison.divergence.field == "topology_signature"
    assert comparison.divergence.left != comparison.divergence.right


def test_transition_difference_reports_first_record() -> None:
    left = _scan()
    changed = replace(
        left.transitions[0],
        changes=(
            KinematicTopologyChange("pallet_transition", 0, (0, 0), "changed"),
        ),
    )
    right = replace(left, transitions=(changed,))

    comparison = compare_kinematic_topology_scans(left, right)

    assert not comparison.passed
    assert comparison.divergence is not None
    assert comparison.divergence.field == "transition_changes"
    assert comparison.divergence.frame == 0
    assert comparison.divergence.parameter == pytest.approx(0.375)


def test_tolerances_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="absolute"):
        RepetitionTolerances(absolute=-1.0)
    with pytest.raises(ValueError, match="relative"):
        RepetitionTolerances(relative=float("nan"))
