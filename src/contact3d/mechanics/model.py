"""Array type aliases owned by the mechanics subsystem."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

__all__ = ["FloatArray", "IntArray"]
