from __future__ import annotations

import array as _array
from typing import Dict, List, Tuple

from pysol.types import Type


_TYPECODE: Dict[Type, str] = {
    Type.Int: "i",
    Type.Float: "f",
    Type.Double: "d",
}

_DEFAULT: Dict[Type, object] = {
    Type.Int: 0,
    Type.Float: 0.0,
    Type.Double: 0.0,
    Type.String: "",
}


class ArrayElement:
    """
    A C-style (row-major) N-dimensional block of homogeneous values.
    Indexing: element[i, j, k] or element[(i, j, k)].
    """

    def __init__(self, n_dims: int, size: List[int], type: Type) -> None:
        if len(size) != n_dims:
            raise ValueError(f"size must have {n_dims} entries, got {len(size)}")
        self._n_dims = n_dims
        self._size = list(size)
        self._type = type
        total = 1
        for s in size:
            total *= s
        self._total = total
        default = _DEFAULT[type]
        if type in _TYPECODE:
            self._data: object = _array.array(_TYPECODE[type], [default] * total)
        else:
            self._data = [default] * total

    # ------------------------------------------------------------------
    def _flat(self, key) -> int:
        coords: List[int] = [key] if isinstance(key, int) else list(key)
        if len(coords) != self._n_dims:
            raise IndexError(f"expected {self._n_dims} indices, got {len(coords)}")
        idx = 0
        for dim, (c, s) in enumerate(zip(coords, self._size)):
            if not (0 <= c < s):
                raise IndexError(f"index {c} out of range [0, {s}) at dim {dim}")
            idx = idx * s + c
        return idx

    def __getitem__(self, key):
        return self._data[self._flat(key)]

    def __setitem__(self, key, value) -> None:
        self._data[self._flat(key)] = value

    # ------------------------------------------------------------------
    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def size(self) -> List[int]:
        return list(self._size)

    @property
    def type(self) -> Type:
        return self._type


class Array:
    """
    Blocked N-dimensional versioned array.

    Each cell is identified by (version, *coordinates).  Two cells with the
    same coordinates but different versions are independent ArrayElements.
    Elements are created lazily on first access.

    Indexing:  arr[version, i, j]  or  arr.get(version, [i, j])

    Example
    -------
    arr = Array(2, 3, [10, 20], [30, 40, 50], Type.Float)
    # 2-D array of 10×20 cells; each cell is a 3-D 30×40×50 Float block.
    block = arr[0, 3, 7]          # version 0, row 3, col 7
    block[1, 2, 4] = 3.14
    """

    def __init__(
        self,
        n_dims: int,
        n_block_dims: int,
        array_size: List[int],
        block_size: List[int],
        type: Type,
    ) -> None:
        if len(array_size) != n_dims:
            raise ValueError(f"array_size must have {n_dims} entries")
        if len(block_size) != n_block_dims:
            raise ValueError(f"block_size must have {n_block_dims} entries")
        self._n_dims = n_dims
        self._n_block_dims = n_block_dims
        self._array_size = list(array_size)
        self._block_size = list(block_size)
        self._type = type
        self._store: Dict[Tuple, ArrayElement] = {}

    # ------------------------------------------------------------------
    def _validate(self, coords: List[int]) -> None:
        if len(coords) != self._n_dims:
            raise IndexError(f"expected {self._n_dims} coordinates, got {len(coords)}")
        for dim, (c, s) in enumerate(zip(coords, self._array_size)):
            if not (0 <= c < s):
                raise IndexError(f"coordinate {c} out of range [0, {s}) at dim {dim}")

    def get(self, version: int, coordinates: List[int]) -> ArrayElement:
        """Return the ArrayElement at (version, *coordinates), creating it if absent."""
        self._validate(coordinates)
        key: Tuple = (version, *coordinates)
        if key not in self._store:
            self._store[key] = ArrayElement(self._n_block_dims, self._block_size, self._type)
        return self._store[key]

    def __getitem__(self, key) -> ArrayElement:
        """arr[version, i, j, ...]"""
        if not isinstance(key, tuple) or len(key) != self._n_dims + 1:
            raise IndexError(
                f"key must be a tuple (version, *{self._n_dims} coordinates), got {key!r}"
            )
        return self.get(int(key[0]), list(key[1:]))

    # ------------------------------------------------------------------
    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def n_block_dims(self) -> int:
        return self._n_block_dims

    @property
    def array_size(self) -> List[int]:
        return list(self._array_size)

    @property
    def block_size(self) -> List[int]:
        return list(self._block_size)

    @property
    def type(self) -> Type:
        return self._type
