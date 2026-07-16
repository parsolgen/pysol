import pytest
from pysol import Array, ArrayElement, Type, pysol_update_array


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
