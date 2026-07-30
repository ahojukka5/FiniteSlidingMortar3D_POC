"""Small deterministic CSR matrix utilities for verification-sized assemblies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class CSRMatrix:
    """Immutable compressed-sparse-row matrix with deterministic ordering."""

    shape: tuple[int, int]
    indptr: IntArray
    indices: IntArray
    data: FloatArray

    def __post_init__(self) -> None:
        rows, columns = self.shape
        indptr = np.asarray(self.indptr, dtype=np.int64)
        indices = np.asarray(self.indices, dtype=np.int64)
        data = np.asarray(self.data, dtype=float)
        if rows < 0 or columns < 0:
            raise ValueError("matrix dimensions must be nonnegative")
        if indptr.shape != (rows + 1,):
            raise ValueError("indptr must have length row_count + 1")
        if indptr[0] != 0 or np.any(indptr[1:] < indptr[:-1]):
            raise ValueError("indptr must be monotone and start at zero")
        if indptr[-1] != len(indices) or len(indices) != len(data):
            raise ValueError("CSR arrays have incompatible lengths")
        if np.any(indices < 0) or np.any(indices >= columns):
            raise ValueError("CSR column index is out of range")
        if not np.all(np.isfinite(data)):
            raise ValueError("CSR data must be finite")
        for row in range(rows):
            start = int(indptr[row])
            stop = int(indptr[row + 1])
            if stop - start > 1 and np.any(
                indices[start + 1 : stop] <= indices[start : stop - 1]
            ):
                raise ValueError("CSR column indices must be strictly increasing per row")
        object.__setattr__(self, "indptr", indptr.copy())
        object.__setattr__(self, "indices", indices.copy())
        object.__setattr__(self, "data", data.copy())

    @property
    def nnz(self) -> int:
        return len(self.data)

    def to_dense(self) -> FloatArray:
        dense = np.zeros(self.shape, dtype=float)
        for row in range(self.shape[0]):
            start = int(self.indptr[row])
            stop = int(self.indptr[row + 1])
            dense[row, self.indices[start:stop]] = self.data[start:stop]
        return dense

    def matvec(self, vector: FloatArray) -> FloatArray:
        values = np.asarray(vector, dtype=float)
        if values.shape != (self.shape[1],):
            raise ValueError("vector length must match the matrix column count")
        result = np.zeros(self.shape[0], dtype=float)
        for row in range(self.shape[0]):
            start = int(self.indptr[row])
            stop = int(self.indptr[row + 1])
            result[row] = float(np.dot(self.data[start:stop], values[self.indices[start:stop]]))
        return result

    def extract_dense(self, rows: IntArray, columns: IntArray) -> FloatArray:
        row_indices = np.asarray(rows, dtype=np.int64)
        column_indices = np.asarray(columns, dtype=np.int64)
        if np.any(row_indices < 0) or np.any(row_indices >= self.shape[0]):
            raise ValueError("requested row is out of range")
        if np.any(column_indices < 0) or np.any(column_indices >= self.shape[1]):
            raise ValueError("requested column is out of range")
        lookup = {int(column): position for position, column in enumerate(column_indices)}
        result = np.zeros((len(row_indices), len(column_indices)), dtype=float)
        for output_row, row in enumerate(row_indices):
            start = int(self.indptr[int(row)])
            stop = int(self.indptr[int(row) + 1])
            for column, value in zip(
                self.indices[start:stop], self.data[start:stop], strict=True
            ):
                output_column = lookup.get(int(column))
                if output_column is not None:
                    result[output_row, output_column] = value
        return result


class SparseAccumulator:
    """Dictionary-backed block accumulator converted once to deterministic CSR."""

    def __init__(self, shape: tuple[int, int]) -> None:
        rows, columns = shape
        if rows < 0 or columns < 0:
            raise ValueError("matrix dimensions must be nonnegative")
        self.shape = shape
        self._rows: list[dict[int, float]] = [dict() for _ in range(rows)]

    def add_block(
        self,
        row_indices: IntArray,
        column_indices: IntArray,
        block: FloatArray,
    ) -> None:
        rows = np.asarray(row_indices, dtype=np.int64)
        columns = np.asarray(column_indices, dtype=np.int64)
        values = np.asarray(block, dtype=float)
        if values.shape != (len(rows), len(columns)):
            raise ValueError("block shape must match the supplied row and column indices")
        if np.any(rows < 0) or np.any(rows >= self.shape[0]):
            raise ValueError("block row index is out of range")
        if np.any(columns < 0) or np.any(columns >= self.shape[1]):
            raise ValueError("block column index is out of range")
        if not np.all(np.isfinite(values)):
            raise ValueError("block values must be finite")
        for local_row, row in enumerate(rows):
            target = self._rows[int(row)]
            for local_column, column in enumerate(columns):
                key = int(column)
                target[key] = target.get(key, 0.0) + float(values[local_row, local_column])

    def to_csr(self, *, drop_tolerance: float = 0.0) -> CSRMatrix:
        if not np.isfinite(drop_tolerance) or drop_tolerance < 0.0:
            raise ValueError("drop_tolerance must be finite and nonnegative")
        indptr = [0]
        indices: list[int] = []
        data: list[float] = []
        for row in self._rows:
            for column in sorted(row):
                value = row[column]
                if abs(value) > drop_tolerance:
                    indices.append(column)
                    data.append(value)
            indptr.append(len(indices))
        return CSRMatrix(
            self.shape,
            np.asarray(indptr, dtype=np.int64),
            np.asarray(indices, dtype=np.int64),
            np.asarray(data, dtype=float),
        )
