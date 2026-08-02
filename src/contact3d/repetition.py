"""Deterministic comparison of repeated contact-topology path scans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .topology_scan import KinematicTopologyScan
from .topology_signature import topology_signature_hash


@dataclass(frozen=True, slots=True)
class RepetitionTolerances:
    """Absolute and relative limits for continuous scan fields."""

    absolute: float = 1.0e-12
    relative: float = 1.0e-10

    def __post_init__(self) -> None:
        absolute = float(self.absolute)
        relative = float(self.relative)
        if not np.isfinite(absolute) or absolute < 0.0:
            raise ValueError("absolute repetition tolerance must be finite and nonnegative")
        if not np.isfinite(relative) or relative < 0.0:
            raise ValueError("relative repetition tolerance must be finite and nonnegative")
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "relative", relative)


@dataclass(frozen=True, slots=True)
class RepetitionDivergence:
    """First field that differs between two repeated scans."""

    field: str
    frame: int | None
    parameter: float | None
    interface: int | None
    left: object
    right: object
    absolute_error: float | None = None
    relative_error: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "frame": self.frame,
            "parameter": self.parameter,
            "interface": self.interface,
            "left": self.left,
            "right": self.right,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
        }


@dataclass(frozen=True, slots=True)
class RepetitionComparison:
    """Outcome of comparing two complete topology scans."""

    passed: bool
    divergence: RepetitionDivergence | None
    frame_count: int
    transition_count: int
    maximum_absolute_error: float
    maximum_relative_error: float

    def summary(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "frame_count": self.frame_count,
            "transition_count": self.transition_count,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "first_divergence": (
                None if self.divergence is None else self.divergence.as_dict()
            ),
        }


def _numeric_errors(left: float, right: float) -> tuple[float, float]:
    absolute = abs(float(left) - float(right))
    scale = max(abs(float(left)), abs(float(right)))
    relative = 0.0 if scale == 0.0 else absolute / scale
    return absolute, relative


def _numeric_equal(
    left: float,
    right: float,
    tolerances: RepetitionTolerances,
) -> tuple[bool, float, float]:
    absolute, relative = _numeric_errors(left, right)
    return (
        absolute <= tolerances.absolute + tolerances.relative * abs(float(right)),
        absolute,
        relative,
    )


def _divergence(
    field: str,
    left: object,
    right: object,
    *,
    frame: int | None = None,
    parameter: float | None = None,
    interface: int | None = None,
    absolute_error: float | None = None,
    relative_error: float | None = None,
) -> RepetitionComparison:
    return RepetitionComparison(
        False,
        RepetitionDivergence(
            field,
            frame,
            parameter,
            interface,
            left,
            right,
            absolute_error,
            relative_error,
        ),
        0,
        0,
        0.0 if absolute_error is None else absolute_error,
        0.0 if relative_error is None else relative_error,
    )


def compare_kinematic_topology_scans(
    left: KinematicTopologyScan,
    right: KinematicTopologyScan,
    *,
    tolerances: RepetitionTolerances | None = None,
) -> RepetitionComparison:
    """Compare repeated scans using exact discrete and tolerant numeric fields."""

    settings = RepetitionTolerances() if tolerances is None else tolerances
    if len(left.frames) != len(right.frames):
        return _divergence("frame_count", len(left.frames), len(right.frames))
    if len(left.transitions) != len(right.transitions):
        return _divergence(
            "transition_count", len(left.transitions), len(right.transitions)
        )

    maximum_absolute = 0.0
    maximum_relative = 0.0

    for frame_index, (left_frame, right_frame) in enumerate(
        zip(left.frames, right.frames, strict=True)
    ):
        parameter = left_frame.parameter
        numeric_fields = (
            ("parameter", left_frame.parameter, right_frame.parameter),
            (
                "phase_parameter",
                left_frame.phase_parameter,
                right_frame.phase_parameter,
            ),
        )
        for field, left_value, right_value in numeric_fields:
            equal, absolute, relative = _numeric_equal(
                left_value, right_value, settings
            )
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            if not equal:
                result = _divergence(
                    field,
                    left_value,
                    right_value,
                    frame=frame_index,
                    parameter=parameter,
                    absolute_error=absolute,
                    relative_error=relative,
                )
                return RepetitionComparison(
                    False,
                    result.divergence,
                    len(left.frames),
                    len(left.transitions),
                    maximum_absolute,
                    maximum_relative,
                )
        if left_frame.phase != right_frame.phase:
            return RepetitionComparison(
                False,
                RepetitionDivergence(
                    "phase",
                    frame_index,
                    parameter,
                    None,
                    left_frame.phase,
                    right_frame.phase,
                ),
                len(left.frames),
                len(left.transitions),
                maximum_absolute,
                maximum_relative,
            )
        if len(left_frame.contacts) != len(right_frame.contacts):
            return RepetitionComparison(
                False,
                RepetitionDivergence(
                    "interface_count",
                    frame_index,
                    parameter,
                    None,
                    len(left_frame.contacts),
                    len(right_frame.contacts),
                ),
                len(left.frames),
                len(left.transitions),
                maximum_absolute,
                maximum_relative,
            )

        for interface, (left_sample, right_sample) in enumerate(
            zip(left_frame.contacts, right_frame.contacts, strict=True)
        ):
            left_hash = topology_signature_hash(left_sample.signature)
            right_hash = topology_signature_hash(right_sample.signature)
            if left_hash != right_hash:
                return RepetitionComparison(
                    False,
                    RepetitionDivergence(
                        "topology_signature",
                        frame_index,
                        parameter,
                        interface,
                        left_hash,
                        right_hash,
                    ),
                    len(left.frames),
                    len(left.transitions),
                    maximum_absolute,
                    maximum_relative,
                )
            for field, left_value, right_value in (
                ("overlap_area", left_sample.overlap_area, right_sample.overlap_area),
                (
                    "maximum_pressure",
                    left_sample.maximum_pressure,
                    right_sample.maximum_pressure,
                ),
            ):
                equal, absolute, relative = _numeric_equal(
                    left_value, right_value, settings
                )
                maximum_absolute = max(maximum_absolute, absolute)
                maximum_relative = max(maximum_relative, relative)
                if not equal:
                    return RepetitionComparison(
                        False,
                        RepetitionDivergence(
                            field,
                            frame_index,
                            parameter,
                            interface,
                            left_value,
                            right_value,
                            absolute,
                            relative,
                        ),
                        len(left.frames),
                        len(left.transitions),
                        maximum_absolute,
                        maximum_relative,
                    )

    for transition_index, (left_item, right_item) in enumerate(
        zip(left.transitions, right.transitions, strict=True)
    ):
        for field, left_value, right_value in (
            ("transition_left_parameter", left_item.left_parameter, right_item.left_parameter),
            (
                "transition_right_parameter",
                left_item.right_parameter,
                right_item.right_parameter,
            ),
        ):
            equal, absolute, relative = _numeric_equal(
                left_value, right_value, settings
            )
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            if not equal:
                return RepetitionComparison(
                    False,
                    RepetitionDivergence(
                        field,
                        transition_index,
                        left_item.midpoint,
                        None,
                        left_value,
                        right_value,
                        absolute,
                        relative,
                    ),
                    len(left.frames),
                    len(left.transitions),
                    maximum_absolute,
                    maximum_relative,
                )
        left_changes = tuple(
            (change.kind, change.interface, change.entity, change.detail)
            for change in left_item.changes
        )
        right_changes = tuple(
            (change.kind, change.interface, change.entity, change.detail)
            for change in right_item.changes
        )
        if left_changes != right_changes:
            return RepetitionComparison(
                False,
                RepetitionDivergence(
                    "transition_changes",
                    transition_index,
                    left_item.midpoint,
                    None,
                    left_changes,
                    right_changes,
                ),
                len(left.frames),
                len(left.transitions),
                maximum_absolute,
                maximum_relative,
            )

    return RepetitionComparison(
        True,
        None,
        len(left.frames),
        len(left.transitions),
        maximum_absolute,
        maximum_relative,
    )
