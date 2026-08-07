"""Multiplier transport across contact-support topology changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...mortar.enforcement import AugmentedLagrangeState
from ...topology_model import BranchSignature


@dataclass(frozen=True, slots=True)
class MultiplierTransportRecord:
    """One interface multiplier update caused by a support transition."""

    interface: int
    released_rows: tuple[int, ...]
    activated_rows: tuple[int, ...]
    changed_rows: tuple[int, ...]
    values_before: tuple[float, ...]
    values_after: tuple[float, ...]
    maximum_unsupported_before: float
    maximum_unsupported_after: float
    initialization_rule: str = "zero"

    def __post_init__(self) -> None:
        if self.interface < 0:
            raise ValueError("transport interface must be nonnegative")
        if self.initialization_rule != "zero":
            raise ValueError("only zero initialization is supported")
        if len(self.changed_rows) != len(self.values_before):
            raise ValueError("changed rows and before values must align")
        if len(self.changed_rows) != len(self.values_after):
            raise ValueError("changed rows and after values must align")
        if self.maximum_unsupported_before < 0.0:
            raise ValueError("unsupported multiplier norm must be nonnegative")
        if self.maximum_unsupported_after < 0.0:
            raise ValueError("unsupported multiplier norm must be nonnegative")

    def as_dict(self) -> dict[str, object]:
        """Return a strict-JSON-compatible event-history record."""

        return {
            "interface": self.interface,
            "released_rows": list(self.released_rows),
            "activated_rows": list(self.activated_rows),
            "changed_rows": list(self.changed_rows),
            "values_before": list(self.values_before),
            "values_after": list(self.values_after),
            "maximum_unsupported_before": self.maximum_unsupported_before,
            "maximum_unsupported_after": self.maximum_unsupported_after,
            "initialization_rule": self.initialization_rule,
        }


def _support(signature: BranchSignature, size: int) -> np.ndarray:
    values = np.asarray(signature.supported_rows, dtype=bool)
    if values.shape != (size,):
        raise ValueError("support signature must match multiplier state size")
    return values


def transport_multiplier_states(
    states: tuple[AugmentedLagrangeState, ...],
    left_signatures: tuple[BranchSignature, ...],
    right_signatures: tuple[BranchSignature, ...],
) -> tuple[
    tuple[AugmentedLagrangeState, ...],
    tuple[MultiplierTransportRecord, ...],
]:
    """Transport accepted multipliers onto a selected post-event support branch.

    Rows released by the selected branch are zeroed before the post-event Newton
    restart. Newly supported rows are also initialized to zero rather than
    inheriting a value from an earlier support interval. Rows supported on both
    branches retain their accepted multiplier.
    """

    states = tuple(states)
    left_signatures = tuple(left_signatures)
    right_signatures = tuple(right_signatures)
    if not len(states) == len(left_signatures) == len(right_signatures):
        raise ValueError("states and topology signatures must have equal length")

    transported: list[AugmentedLagrangeState] = []
    records: list[MultiplierTransportRecord] = []
    for interface, (state, left, right) in enumerate(
        zip(states, left_signatures, right_signatures, strict=True)
    ):
        if not isinstance(state, AugmentedLagrangeState):
            transported.append(state)  # type: ignore[arg-type]
            continue
        values = np.asarray(state.multipliers, dtype=float)
        left_support = _support(left, len(values))
        right_support = _support(right, len(values))
        released = np.flatnonzero(left_support & ~right_support)
        activated = np.flatnonzero(~left_support & right_support)
        changed = np.flatnonzero(left_support != right_support)

        before_unsupported = float(
            np.max(np.abs(values[~right_support]), initial=0.0)
        )
        updated = values.copy()
        updated[~right_support] = 0.0
        updated[activated] = 0.0
        after_unsupported = float(
            np.max(np.abs(updated[~right_support]), initial=0.0)
        )
        transported.append(AugmentedLagrangeState(updated, state.augmentation))

        if len(changed):
            records.append(
                MultiplierTransportRecord(
                    interface=interface,
                    released_rows=tuple(int(row) for row in released),
                    activated_rows=tuple(int(row) for row in activated),
                    changed_rows=tuple(int(row) for row in changed),
                    values_before=tuple(float(values[row]) for row in changed),
                    values_after=tuple(float(updated[row]) for row in changed),
                    maximum_unsupported_before=before_unsupported,
                    maximum_unsupported_after=after_unsupported,
                )
            )
    return tuple(transported), tuple(records)


__all__ = [
    "MultiplierTransportRecord",
    "transport_multiplier_states",
]
