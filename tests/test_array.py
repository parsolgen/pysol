import pytest
from pysol import (
    Array, ArrayElement, CSRBlock, CSRMatrix, GrowingArray, LATEST_VERSION, Type, is_zero,
    pysol_update_array, reset,
)


# ---------------------------------------------------------------------------
# ArrayElement
# ---------------------------------------------------------------------------

def test_element_1d_read_write():
    el = ArrayElement(1, [5], Type.Int)
    el[3] = 42
    assert el[3] == 42


def test_element_3d_row_major():
    el = ArrayElement(3, [2, 3, 4], Type.Double)
    el[1, 2, 3] = 1.5
    assert el[1, 2, 3] == 1.5
    assert el[0, 0, 0] == 0.0


def test_element_string_type():
    el = ArrayElement(1, [3], Type.String)
    el[2] = "hello"
    assert el[2] == "hello"
    assert el[0] == ""


def test_element_index_out_of_range():
    el = ArrayElement(2, [4, 4], Type.Float)
    with pytest.raises(IndexError):
        _ = el[4, 0]


def test_element_wrong_ndims():
    el = ArrayElement(2, [4, 4], Type.Float)
    with pytest.raises(IndexError):
        _ = el[0]  # 1 index for 2-D element


# ---------------------------------------------------------------------------
# Array construction
# ---------------------------------------------------------------------------

def test_array_1d_example():
    # Array(1, 1, [10], [10], Type.Float)
    arr = Array(1, 1, [10], [10], Type.Float)
    assert arr.n_dims == 1
    assert arr.n_block_dims == 1
    assert arr.array_size == [10]
    assert arr.block_size == [10]


def test_array_2d_example():
    # Array(2, 3, [10, 20], [30, 40, 50], Type.Float)
    arr = Array(2, 3, [10, 20], [30, 40, 50], Type.Float)
    assert arr.n_dims == 2
    assert arr.n_block_dims == 3
    assert arr.array_size == [10, 20]
    assert arr.block_size == [30, 40, 50]


def test_array_bad_sizes():
    with pytest.raises(ValueError):
        Array(2, 1, [10], [5], Type.Int)   # array_size length mismatch
    with pytest.raises(ValueError):
        Array(1, 2, [10], [5], Type.Int)   # block_size length mismatch


def test_array_is_single_version_defaults_false():
    arr = Array(1, 1, [10], [10], Type.Float)
    assert arr.is_single_version is False


def test_array_is_single_version_positional():
    arr = Array(1, 1, [10], [10], Type.Float, True)
    assert arr.is_single_version is True


def test_array_is_single_version_keyword():
    arr = Array(1, 1, [10], [10], Type.Float, is_single_version=True)
    assert arr.is_single_version is True


# ---------------------------------------------------------------------------
# Array element access
# ---------------------------------------------------------------------------

def test_array_get_returns_element():
    arr = Array(1, 1, [10], [10], Type.Float)
    el = arr.get(0, [3])
    assert isinstance(el, ArrayElement)
    assert el.size == [10]


def test_array_bracket_equals_get():
    arr = Array(1, 1, [10], [10], Type.Float)
    assert arr[0, 3] is arr.get(0, [3])


def test_array_lazy_creation():
    arr = Array(2, 2, [5, 5], [3, 3], Type.Int)
    el = arr[0, 1, 2]
    el[0, 0] = 7
    assert arr[0, 1, 2][0, 0] == 7


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

def test_different_versions_are_independent():
    arr = Array(1, 1, [5], [4], Type.Double)
    arr[0, 2][1] = 1.0
    arr[1, 2][1] = 2.0
    assert arr[0, 2][1] == 1.0
    assert arr[1, 2][1] == 2.0


def test_same_version_same_object():
    arr = Array(1, 1, [5], [4], Type.Int)
    assert arr[3, 0] is arr.get(3, [0])


def test_array_coord_out_of_range():
    arr = Array(1, 1, [5], [4], Type.Int)
    with pytest.raises(IndexError):
        arr[0, 5]


# ---------------------------------------------------------------------------
# ArrayElement operator overloads
# ---------------------------------------------------------------------------

def test_array_element_add_returns_element():
    el = ArrayElement(1, [4], Type.Float)
    result = el + el
    assert isinstance(result, ArrayElement)


def test_array_element_radd_returns_element():
    el = ArrayElement(1, [4], Type.Float)
    result = 1 + el
    assert isinstance(result, ArrayElement)


def test_array_element_mul_returns_element():
    el = ArrayElement(1, [4], Type.Float)
    result = el * el
    assert isinstance(result, ArrayElement)


def test_array_element_rmul_returns_element():
    el = ArrayElement(1, [4], Type.Float)
    result = 2.0 * el
    assert isinstance(result, ArrayElement)


# ---------------------------------------------------------------------------
# Array operator overloads
# ---------------------------------------------------------------------------

def test_array_add_returns_array():
    a = Array(1, 1, [5], [3], Type.Int)
    b = Array(1, 1, [5], [3], Type.Int)
    result = a + b
    assert isinstance(result, Array)


def test_array_radd_returns_array():
    a = Array(1, 1, [5], [3], Type.Int)
    result = 1 + a
    assert isinstance(result, Array)


def test_array_mul_returns_array():
    a = Array(1, 1, [5], [3], Type.Float)
    b = Array(1, 1, [5], [3], Type.Float)
    result = a * b
    assert isinstance(result, Array)


def test_array_rmul_returns_array():
    a = Array(1, 1, [5], [3], Type.Float)
    result = 3.0 * a
    assert isinstance(result, Array)


# ---------------------------------------------------------------------------
# pysol_update_array
# ---------------------------------------------------------------------------

def test_pysol_update_array_returns_none():
    src = Array(1, 1, [5], [3], Type.Int)
    dst = Array(1, 1, [5], [3], Type.Int)
    assert pysol_update_array(src, 0, dst, 1) is None


def test_pysol_update_array_does_not_raise():
    src = Array(2, 1, [4, 4], [2], Type.Float)
    dst = Array(2, 1, [4, 4], [2], Type.Float)
    pysol_update_array(src, 0, dst, 1)


# ---------------------------------------------------------------------------
# GrowingArray construction
# ---------------------------------------------------------------------------

def test_growing_array_1d_example():
    arr = GrowingArray(1, [10], Type.Float)
    assert arr.n_dims == 1
    assert arr.array_size == [10]
    assert arr.type == Type.Float


def test_growing_array_3d_example():
    arr = GrowingArray(3, [10, 20, 30], Type.Double)
    assert arr.n_dims == 3
    assert arr.array_size == [10, 20, 30]


def test_growing_array_bad_size():
    with pytest.raises(ValueError):
        GrowingArray(2, [10], Type.Int)  # array_size length mismatch


def test_growing_array_has_no_block_params():
    """GrowingArray's ctor takes no n_block_dims / block_size — unlike Array."""
    assert not hasattr(GrowingArray(1, [5], Type.Int), "n_block_dims")
    assert not hasattr(GrowingArray(1, [5], Type.Int), "block_size")


# ---------------------------------------------------------------------------
# GrowingArray element access
# ---------------------------------------------------------------------------

def test_growing_array_get_default():
    arr = GrowingArray(1, [10], Type.Float)
    assert arr.get(0, [3]) == 0.0


def test_growing_array_set_and_get():
    arr = GrowingArray(1, [10], Type.Int)
    arr.set(0, [3], 42)
    assert arr.get(0, [3]) == 42


def test_growing_array_bracket_equals_get_set():
    arr = GrowingArray(2, [5, 5], Type.Double)
    arr[0, 1, 2] = 3.14
    assert arr[0, 1, 2] == 3.14
    assert arr[0, 1, 2] == arr.get(0, [1, 2])


def test_growing_array_versions_independent():
    arr = GrowingArray(1, [5], Type.Double)
    arr[0, 2] = 1.0
    arr[1, 2] = 2.0
    assert arr[0, 2] == 1.0
    assert arr[1, 2] == 2.0


def test_growing_array_coord_out_of_range():
    arr = GrowingArray(1, [5], Type.Int)
    with pytest.raises(IndexError):
        arr[0, 5]


def test_growing_array_wrong_key_shape():
    arr = GrowingArray(2, [5, 5], Type.Int)
    with pytest.raises(IndexError):
        _ = arr[0, 1]  # missing one coordinate


# ---------------------------------------------------------------------------
# GrowingArray operator overloads
# ---------------------------------------------------------------------------

def test_growing_array_add_returns_growing_array():
    a = GrowingArray(1, [5], Type.Int)
    b = GrowingArray(1, [5], Type.Int)
    assert isinstance(a + b, GrowingArray)


def test_growing_array_mul_returns_growing_array():
    a = GrowingArray(1, [5], Type.Float)
    assert isinstance(a * 2.0, GrowingArray)


# ---------------------------------------------------------------------------
# CSRMatrix / CSRBlock
# ---------------------------------------------------------------------------

def _make_csr(dim1=4, dim2=4, n_blocks_1=2, n_blocks_2=2, nnz=6, type=Type.Double):
    ia = Array(1, 1, [dim1 + 1], [1], Type.Int)
    ja = Array(1, 1, [nnz], [1], Type.Int)
    a = Array(1, 1, [nnz], [1], type)
    return CSRMatrix([dim1, dim2], [n_blocks_1, n_blocks_2], nnz, ia, ja, a, type), ia, ja, a


def test_csr_matrix_attributes():
    A, ia, ja, a = _make_csr()
    assert A.size == [4, 4]
    assert A.n_blocks == [2, 2]
    assert A.nnz == 6
    assert A.ia is ia
    assert A.ja is ja
    assert A.a is a
    assert A.type is Type.Double


def test_csr_matrix_bad_size():
    ia = Array(1, 1, [5], [1], Type.Int)
    ja = Array(1, 1, [6], [1], Type.Int)
    a = Array(1, 1, [6], [1], Type.Double)
    with pytest.raises(ValueError):
        CSRMatrix([4], [2, 2], 6, ia, ja, a, Type.Double)


def test_csr_matrix_bad_n_blocks():
    ia = Array(1, 1, [5], [1], Type.Int)
    ja = Array(1, 1, [6], [1], Type.Int)
    a = Array(1, 1, [6], [1], Type.Double)
    with pytest.raises(ValueError):
        CSRMatrix([4, 4], [2], 6, ia, ja, a, Type.Double)


def test_csr_matrix_getitem_tuple_returns_csr_block():
    A, *_ = _make_csr()
    block = A[0, 1]
    assert isinstance(block, CSRBlock)
    assert block.type is Type.Double


def test_csr_matrix_getitem_double_subscript_returns_csr_block():
    A, *_ = _make_csr()
    block = A[1][0]
    assert isinstance(block, CSRBlock)


def test_csr_matrix_getitem_out_of_range_tuple():
    A, *_ = _make_csr()
    with pytest.raises(IndexError):
        _ = A[2, 0]
    with pytest.raises(IndexError):
        _ = A[0, 2]


def test_csr_matrix_getitem_out_of_range_double_subscript():
    A, *_ = _make_csr()
    with pytest.raises(IndexError):
        _ = A[2][0]
    with pytest.raises(IndexError):
        _ = A[0][2]


def test_csr_matrix_getitem_wrong_tuple_len():
    A, *_ = _make_csr()
    with pytest.raises(IndexError):
        _ = A[0, 1, 0]


def test_csr_block_repr():
    A, *_ = _make_csr()
    r = repr(A[0, 1])
    assert "CSRBlock" in r


# ---------------------------------------------------------------------------
# is_zero
# ---------------------------------------------------------------------------

def test_is_zero_accepts_csr_block():
    A, *_ = _make_csr()
    assert is_zero(A[0, 1]) in (True, False)


def test_is_zero_rejects_non_csr_block():
    with pytest.raises(TypeError):
        is_zero(42)


# ---------------------------------------------------------------------------
# LATEST_VERSION
# ---------------------------------------------------------------------------

def test_latest_version_defaults_to_zero_when_unset():
    arr = Array(1, 1, [10], [4], Type.Float)
    el = arr[LATEST_VERSION, 3]
    assert el is arr[0, 3]


def test_latest_version_resolves_to_highest_written_version():
    arr = Array(1, 1, [10], [4], Type.Int)
    arr[0, 3][0] = 1
    arr[5, 3][0] = 2
    arr[2, 3][0] = 3
    assert arr[LATEST_VERSION, 3] is arr[5, 3]


def test_latest_version_is_per_coordinate():
    arr = Array(1, 1, [10], [4], Type.Int)
    arr[7, 3][0] = 1
    # Coordinate 4 has no versions beyond the default 0.
    assert arr[LATEST_VERSION, 4] is arr[0, 4]
    assert arr[LATEST_VERSION, 3] is arr[7, 3]


def test_latest_version_via_get():
    arr = Array(1, 1, [10], [4], Type.Int)
    arr.get(9, [1])
    assert arr.get(LATEST_VERSION, [1]) is arr.get(9, [1])


def test_latest_version_repr():
    assert repr(LATEST_VERSION) == "LATEST_VERSION"


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------

def test_reset_array_clears_store():
    arr = Array(1, 1, [10], [4], Type.Int)
    el = arr[0, 3]
    el[0] = 42
    reset(arr)
    assert arr[0, 3] is not el
    assert arr[0, 3][0] == 0


def test_reset_growing_array_clears_store():
    arr = GrowingArray(1, [10], Type.Int)
    arr.set(0, [3], 42)
    reset(arr)
    assert arr.get(0, [3]) == 0


def test_reset_csr_matrix_is_noop():
    A, *_ = _make_csr()
    reset(A)  # should not raise


def test_reset_rejects_other_types():
    with pytest.raises(TypeError):
        reset(42)
