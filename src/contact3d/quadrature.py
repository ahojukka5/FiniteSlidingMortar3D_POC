"""Symmetric quadrature rules on a triangle."""

from __future__ import annotations

import numpy as np

from .model import FloatArray


def triangle_rule(point_count: int) -> tuple[FloatArray, FloatArray]:
    """Return barycentric points and weights normalized to sum to one."""

    if point_count == 1:
        return np.array([[1.0 / 3.0] * 3]), np.array([1.0])
    if point_count == 3:
        return (
            np.array(
                [
                    [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                    [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                    [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
                ]
            ),
            np.full(3, 1.0 / 3.0),
        )
    if point_count == 7:
        a = 0.059715871789770
        b = 0.470142064105115
        c = 0.797426985353087
        d = 0.101286507323456
        return (
            np.array(
                [
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    [a, b, b],
                    [b, a, b],
                    [b, b, a],
                    [c, d, d],
                    [d, c, d],
                    [d, d, c],
                ]
            ),
            np.array(
                [
                    0.225,
                    0.132394152788506,
                    0.132394152788506,
                    0.132394152788506,
                    0.125939180544827,
                    0.125939180544827,
                    0.125939180544827,
                ]
            ),
        )
    raise ValueError("triangle quadrature supports 1, 3, or 7 points")
