import pytest
from pysol import ArrayElementType, ArrayType, CSRBlockType, Type


def test_type_members():
    assert Type.Int.value == "Int"
    assert Type.Float.value == "Float"
    assert Type.Double.value == "Double"
    assert Type.String.value == "String"


def test_type_lookup():
    assert Type("Int") is Type.Int
    assert Type("String") is Type.String


# ---------------------------------------------------------------------------
# ArrayType
# ---------------------------------------------------------------------------

def test_array_type_attributes():
    at = ArrayType(2, 3, Type.Float)
    assert at.n_dims == 2
    assert at.n_block_dims == 3
    assert at.element_type is Type.Float


def test_array_type_equality():
    assert ArrayType(1, 1, Type.Int) == ArrayType(1, 1, Type.Int)
    assert ArrayType(1, 1, Type.Int) != ArrayType(2, 1, Type.Int)
    assert ArrayType(1, 1, Type.Int) != ArrayType(1, 1, Type.Float)


def test_array_type_hashable():
    s = {ArrayType(1, 2, Type.Double), ArrayType(1, 2, Type.Double)}
    assert len(s) == 1


def test_array_type_repr():
    r = repr(ArrayType(2, 3, Type.String))
    assert "n_dims=2" in r
    assert "n_block_dims=3" in r
    assert "String" in r


def test_array_type_invalid_dims():
    with pytest.raises(ValueError):
        ArrayType(0, 1, Type.Int)
    with pytest.raises(ValueError):
        ArrayType(1, 0, Type.Int)


# ---------------------------------------------------------------------------
# ArrayElementType
# ---------------------------------------------------------------------------

def test_array_element_type_attributes():
    aet = ArrayElementType(3, Type.Double)
    assert aet.n_dims == 3
    assert aet.element_type is Type.Double


def test_array_element_type_equality():
    assert ArrayElementType(2, Type.Float) == ArrayElementType(2, Type.Float)
    assert ArrayElementType(2, Type.Float) != ArrayElementType(3, Type.Float)
    assert ArrayElementType(2, Type.Float) != ArrayElementType(2, Type.Int)


def test_array_element_type_hashable():
    s = {ArrayElementType(1, Type.Int), ArrayElementType(1, Type.Int)}
    assert len(s) == 1


def test_array_element_type_repr():
    r = repr(ArrayElementType(2, Type.String))
    assert "n_dims=2" in r
    assert "String" in r


def test_array_element_type_invalid_dims():
    with pytest.raises(ValueError):
        ArrayElementType(0, Type.Float)


# ---------------------------------------------------------------------------
# Type.CSRBlock / CSRBlockType
# ---------------------------------------------------------------------------

def test_type_csr_block_returns_csr_block_type():
    cb = Type.CSRBlock(Type.Double)
    assert isinstance(cb, CSRBlockType)
    assert cb.element_type is Type.Double


def test_csr_block_type_equality():
    assert Type.CSRBlock(Type.Double) == Type.CSRBlock(Type.Double)
    assert Type.CSRBlock(Type.Double) != Type.CSRBlock(Type.Int)


def test_csr_block_type_hashable():
    s = {Type.CSRBlock(Type.Float), Type.CSRBlock(Type.Float)}
    assert len(s) == 1


def test_csr_block_type_repr():
    r = repr(Type.CSRBlock(Type.Double))
    assert "CSRBlockType" in r
    assert "Double" in r
