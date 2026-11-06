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


class _LatestVersion:
    """
    Sentinel for the version slot of an Array subscript — e.g.
    arr[LATEST_VERSION, i][j] — meaning "the most recently written version
    for these coordinates". Recognized by name at the AST level (see
    pysol.sol_generator._SolWriter._expr) and rendered as the .sol token
    %latest; use the module-level LATEST_VERSION instance, never this class
    directly.
    """

    def __repr__(self) -> str:
        return "LATEST_VERSION"


LATEST_VERSION = _LatestVersion()


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

    def __add__(self, other: object) -> "ArrayElement":
        del other
        return self

    def __radd__(self, other: object) -> "ArrayElement":
        del other
        return self

    def __mul__(self, other: object) -> "ArrayElement":
        del other
        return self

    def __rmul__(self, other: object) -> "ArrayElement":
        del other
        return self

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


class GrowingArray:
    """
    N-dimensional versioned array whose block dimension is always 1 — each
    (version, *coordinates) cell holds a single scalar value rather than a
    nested ArrayElement block.

    Indexing:  arr[version, i, j] = value  /  value = arr[version, i, j]

    Example
    -------
    arr = GrowingArray(3, [10, 20, 30], Type.Double)
    arr[0, 1, 2, 3] = 3.14
    """

    def __init__(self, n_dims: int, array_size: List[int], type: Type) -> None:
        if len(array_size) != n_dims:
            raise ValueError(f"array_size must have {n_dims} entries")
        self._n_dims = n_dims
        self._array_size = list(array_size)
        self._type = type
        self._store: Dict[Tuple, object] = {}

    # ------------------------------------------------------------------
    def _validate(self, coords: List[int]) -> None:
        if len(coords) != self._n_dims:
            raise IndexError(f"expected {self._n_dims} coordinates, got {len(coords)}")
        for dim, (c, s) in enumerate(zip(coords, self._array_size)):
            if not (0 <= c < s):
                raise IndexError(f"coordinate {c} out of range [0, {s}) at dim {dim}")

    def get(self, version: int, coordinates: List[int]):
        """Return the scalar at (version, *coordinates), defaulting if unset."""
        self._validate(coordinates)
        key: Tuple = (version, *coordinates)
        return self._store.get(key, _DEFAULT[self._type])

    def set(self, version: int, coordinates: List[int], value) -> None:
        self._validate(coordinates)
        key: Tuple = (version, *coordinates)
        self._store[key] = value

    def __getitem__(self, key):
        """arr[version, i, j, ...]"""
        if not isinstance(key, tuple) or len(key) != self._n_dims + 1:
            raise IndexError(
                f"key must be a tuple (version, *{self._n_dims} coordinates), got {key!r}"
            )
        return self.get(int(key[0]), list(key[1:]))

    def __setitem__(self, key, value) -> None:
        """arr[version, i, j, ...] = value"""
        if not isinstance(key, tuple) or len(key) != self._n_dims + 1:
            raise IndexError(
                f"key must be a tuple (version, *{self._n_dims} coordinates), got {key!r}"
            )
        self.set(int(key[0]), list(key[1:]), value)

    def __add__(self, other: object) -> "GrowingArray":
        del other
        return self

    def __radd__(self, other: object) -> "GrowingArray":
        del other
        return self

    def __mul__(self, other: object) -> "GrowingArray":
        del other
        return self

    def __rmul__(self, other: object) -> "GrowingArray":
        del other
        return self

    # ------------------------------------------------------------------
    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def array_size(self) -> List[int]:
        return list(self._array_size)

    @property
    def type(self) -> Type:
        return self._type


class CSRBlock:
    """
    Mock runtime value returned by CSRMatrix.__getitem__ — one block of a
    blocked CSR matrix. Like ArrayElement, this exists only so Python-level
    code referencing a CSRMatrix stays a valid, inspectable object; the real
    CSR block lives in the generated C++ (see pysol.types.Type.CSRBlock and
    sol_generator._gen_import_arg_decl).
    """

    def __init__(self, matrix: "CSRMatrix", i: int, j: int) -> None:
        self._matrix = matrix
        self._i = i
        self._j = j
        self._type = matrix.type

    @property
    def type(self) -> Type:
        return self._type

    def __repr__(self) -> str:
        return f"CSRBlock(matrix={self._matrix!r}, i={self._i}, j={self._j})"


class CSRMatrix:
    """
    A blocked 2-D array of CSR (Compressed Sparse Row) matrix blocks.

    Indexing:  arr[i, j]  or  arr[i][j]  →  a CSRBlock

    Example
    -------
    ia = Array(1, 1, [dim1 + 1], [1], Type.Int)
    ja = Array(1, 1, [nnz], [1], Type.Int)
    a  = Array(1, 1, [nnz], [1], Type.Double)
    A = CSRMatrix([dim1, dim2], [n_blocks_1, n_blocks_2], nnz, ia, ja, a, Type.Double)
    # A 2x2-blocked CSR matrix of shape dim1 x dim2.
    block = A[0, 1]          # block (0,1)
    """

    def __init__(
        self,
        size: List[int],
        n_blocks: List[int],
        nnz: int,
        ia: "Array",
        ja: "Array",
        a: "Array",
        type: Type,
    ) -> None:
        if len(size) != 2:
            raise ValueError(f"size must have 2 entries [dim1, dim2], got {len(size)}")
        if len(n_blocks) != 2:
            raise ValueError(
                f"n_blocks must have 2 entries [n_blocks_1, n_blocks_2], got {len(n_blocks)}"
            )
        self._size = list(size)
        self._n_blocks = list(n_blocks)
        self._nnz = nnz
        self._ia = ia
        self._ja = ja
        self._a = a
        self._type = type

    # ------------------------------------------------------------------
    def _validate(self, i: int, j: int) -> None:
        if not (0 <= i < self._n_blocks[0]):
            raise IndexError(f"block row {i} out of range [0, {self._n_blocks[0]})")
        if not (0 <= j < self._n_blocks[1]):
            raise IndexError(f"block col {j} out of range [0, {self._n_blocks[1]})")

    def __getitem__(self, key) -> CSRBlock:
        """arr[i, j] or arr[i][j]"""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(f"expected 2 indices (i, j), got {len(key)}")
            i, j = key
            self._validate(i, j)
            return CSRBlock(self, i, j)
        # arr[i] — return a partially-applied row accessor supporting arr[i][j]
        i = key
        if not (0 <= i < self._n_blocks[0]):
            raise IndexError(f"block row {i} out of range [0, {self._n_blocks[0]})")
        matrix = self

        class _Row:
            def __getitem__(self_row, j: int) -> CSRBlock:
                matrix._validate(i, j)
                return CSRBlock(matrix, i, j)

        return _Row()  # type: ignore[return-value]

    # ------------------------------------------------------------------
    @property
    def size(self) -> List[int]:
        return list(self._size)

    @property
    def n_blocks(self) -> List[int]:
        return list(self._n_blocks)

    @property
    def nnz(self) -> int:
        return self._nnz

    @property
    def ia(self) -> "Array":
        return self._ia

    @property
    def ja(self) -> "Array":
        return self._ja

    @property
    def a(self) -> "Array":
        return self._a

    @property
    def type(self) -> Type:
        return self._type


def is_zero(block: CSRBlock) -> bool:
    """
    Predicate for a CSRMatrix block: is_zero(b) / is_zero(A[i][j]).

    Usable directly inside a @pysol if-condition ("if not is_zero(b):" or
    "if is_zero(A[i][j]):"), where it is recognized by name (see
    pysol.decorator._is_zero_call) and rendered as the .sol macro %zero(...)
    (see pysol.sol_generator._SolWriter._cond) rather than ever being called.
    This Python-level version exists only so the name resolves outside a
    @pysol function.
    """
    if not isinstance(block, CSRBlock):
        raise TypeError(f"is_zero() expects a CSRBlock, got {type(block).__name__}")
    return False


def reset(x) -> None:
    """
    Reset operator: reset(A) — A may be an Array, GrowingArray, or CSRMatrix.

    Usable directly as a @pysol statement, where it is recognized by name
    (see pysol.decorator._is_reset_call) and rendered as the .sol statement
    "A = {};" (see pysol.sol_generator._SolWriter._reset_stmt) rather than
    ever being called. This Python-level version exists only so reset() is
    callable outside a @pysol function; for Array/GrowingArray it clears
    the mock's lazily-created store back to defaults, and is a no-op for
    CSRMatrix (which holds no per-block store of its own).
    """
    if isinstance(x, (Array, GrowingArray)):
        x._store.clear()
    elif isinstance(x, CSRMatrix):
        pass
    else:
        raise TypeError(
            f"reset() expects an Array, GrowingArray, or CSRMatrix, got {type(x).__name__}"
        )


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

    is_single_version=True declares a "SingleVersionArray<...>" in the
    generated .sol instead of "Array<...>" (see
    pysol.sol_generator._SolWriter._array_decl_prefix) - the RTS holder
    that keeps only its most-recently-written version per cell rather than
    every version ever written, matching bicg.sol's own block-vectors
    (declared as SingleVersionArray there). This Python-level mock still
    keeps every version in self._store regardless - the distinction only
    affects the generated declaration's .sol text.
    """

    def __init__(
        self,
        n_dims: int,
        n_block_dims: int,
        array_size: List[int],
        block_size: List[int],
        type: Type,
        is_single_version: bool = False,
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
        self._is_single_version = is_single_version
        self._store: Dict[Tuple, ArrayElement] = {}

    # ------------------------------------------------------------------
    def _validate(self, coords: List[int]) -> None:
        if len(coords) != self._n_dims:
            raise IndexError(f"expected {self._n_dims} coordinates, got {len(coords)}")
        for dim, (c, s) in enumerate(zip(coords, self._array_size)):
            if not (0 <= c < s):
                raise IndexError(f"coordinate {c} out of range [0, {s}) at dim {dim}")

    def _latest_version(self, coordinates: List[int]) -> int:
        matches = [k[0] for k in self._store if list(k[1:]) == coordinates]
        return max(matches) if matches else 0

    def get(self, version, coordinates: List[int]) -> ArrayElement:
        """Return the ArrayElement at (version, *coordinates), creating it if absent.

        version may be LATEST_VERSION, resolving to the highest version
        already stored for these coordinates (0 if none yet).
        """
        self._validate(coordinates)
        if version is LATEST_VERSION:
            version = self._latest_version(coordinates)
        key: Tuple = (version, *coordinates)
        if key not in self._store:
            self._store[key] = ArrayElement(self._n_block_dims, self._block_size, self._type)
        return self._store[key]

    def __getitem__(self, key) -> ArrayElement:
        """arr[version, i, j, ...] — version may be LATEST_VERSION"""
        if not isinstance(key, tuple) or len(key) != self._n_dims + 1:
            raise IndexError(
                f"key must be a tuple (version, *{self._n_dims} coordinates), got {key!r}"
            )
        version = key[0] if key[0] is LATEST_VERSION else int(key[0])
        return self.get(version, list(key[1:]))

    def __add__(self, other: object) -> "Array":
        del other
        return self

    def __radd__(self, other: object) -> "Array":
        del other
        return self

    def __mul__(self, other: object) -> "Array":
        del other
        return self

    def __rmul__(self, other: object) -> "Array":
        del other
        return self

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

    @property
    def is_single_version(self) -> bool:
        return self._is_single_version
