import pytest
from pysol import (
    Array, CSRMatrix, NotSupportedIfStatement, NotSupportedLoop, Type, pysol,
    pysol_struct, UnsupportedStructFieldType,
)
from pysol.decorator import _analyse, _struct_registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(func):
    """Return the _Analysis of func without executing it."""
    return _analyse(func)


# Module-level names referenced inside inner fn() bodies used as AST fixtures.
# Most of those functions are never actually called — they exist so the linter
# can resolve every name that appears in the source the decorator will parse.
a = b = c = x = flag = None
xs: list = []

def func(*_): return True   # type: ignore[return]
def check(*_): return True  # type: ignore[return]
def pred(*_): return True   # type: ignore[return]
def f(*_): return True      # type: ignore[return]
def g(*_): return True      # type: ignore[return]


# ---------------------------------------------------------------------------
# Array detection
# ---------------------------------------------------------------------------

def test_detects_bare_array():
    def fn():
        a = Array(1, 1, [10], [10], Type.Float)

    a = _run(fn)
    assert len(a.arrays) == 1
    assert a.arrays[0].var == "a"
    assert a.arrays[0].n_dims == "1"
    assert a.arrays[0].n_block_dims == "1"
    assert a.arrays[0].array_size == "[10]"
    assert a.arrays[0].block_size == "[10]"
    assert a.arrays[0].elem_type == "Type.Float"
    assert a.arrays[0].is_single_version is False


def test_detects_single_version_array_positional():
    def fn():
        a = Array(1, 1, [10], [10], Type.Float, True)

    a = _run(fn)
    assert a.arrays[0].is_single_version is True


def test_detects_single_version_array_keyword():
    def fn():
        a = Array(1, 1, [10], [10], Type.Float, is_single_version=True)

    a = _run(fn)
    assert a.arrays[0].is_single_version is True


def test_detects_qualified_array():
    import pysol as _pysol

    def fn():
        import pysol as _pysol
        b = _pysol.Array(2, 3, [10, 20], [30, 40, 50], Type.Double)

    a = _run(fn)
    assert len(a.arrays) == 1
    assert a.arrays[0].var == "b"
    assert a.arrays[0].n_dims == "2"
    assert a.arrays[0].block_size == "[30, 40, 50]"
    assert a.arrays[0].elem_type == "Type.Double"


def test_detects_multiple_arrays():
    def fn():
        x = Array(1, 1, [5], [5], Type.Int)
        y = Array(2, 2, [3, 4], [2, 2], Type.String)

    a = _run(fn)
    assert len(a.arrays) == 2
    assert a.arrays[0].var == "x"
    assert a.arrays[1].var == "y"


def test_detects_annotated_assignment():
    def fn():
        arr: Array = Array(1, 1, [8], [8], Type.Double)

    a = _run(fn)
    assert len(a.arrays) == 1
    assert a.arrays[0].var == "arr"


def test_no_arrays():
    def fn():
        x = 1 + 2

    a = _run(fn)
    assert a.arrays == []


# ---------------------------------------------------------------------------
# CSRMatrix detection
# ---------------------------------------------------------------------------

def test_detects_bare_csr_matrix():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)

    analysis = _run(fn)
    assert len(analysis.csr_matrices) == 1
    info = analysis.csr_matrices[0]
    assert info.var == "A"
    assert info.dim1 == "4"
    assert info.dim2 == "4"
    assert info.n_blocks_1 == "2"
    assert info.n_blocks_2 == "2"
    assert info.nnz == "6"
    assert info.ia == "ia"
    assert info.ja == "ja"
    assert info.a == "a"
    assert info.elem_type == "Type.Double"


def test_detects_annotated_csr_matrix():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Float)
        A: CSRMatrix = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Float)

    analysis = _run(fn)
    assert len(analysis.csr_matrices) == 1
    assert analysis.csr_matrices[0].var == "A"
    assert analysis.csr_matrices[0].elem_type == "Type.Float"


def test_no_csr_matrices():
    def fn():
        x = 1 + 2

    analysis = _run(fn)
    assert analysis.csr_matrices == []


def test_csr_matrix_bad_size_literal_raises():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4], [2, 2], 6, ia, ja, a, Type.Double)

    with pytest.raises(ValueError):
        _run(fn)


# ---------------------------------------------------------------------------
# For-loop detection
# ---------------------------------------------------------------------------

def test_detects_for_loop():
    def fn():
        for i in range(10):
            pass

    a = _run(fn)
    assert len(a.for_loops) == 1
    assert a.for_loops[0].target == "i"
    assert a.for_loops[0].iterable == "range(10)"


def test_detects_nested_for_loops():
    def fn():
        for i in range(5):
            for j in range(3):
                pass

    a = _run(fn)
    assert len(a.for_loops) == 2
    assert a.for_loops[0].target == "i"
    assert a.for_loops[1].target == "j"


def test_for_loop_tuple_target_raises():
    def fn():
        for i, j in [(1, 2)]:
            pass

    with pytest.raises(NotSupportedLoop):
        _run(fn)


def test_for_loop_non_range_iterable_raises():
    def fn():
        for _ in [1, 2, 3]:
            pass

    with pytest.raises(NotSupportedLoop):
        _run(fn)


def test_for_loop_variable_iterable_raises():
    def fn():
        items = [1, 2, 3]
        for _ in items:
            pass

    with pytest.raises(NotSupportedLoop):
        _run(fn)


def test_not_supported_loop_message():
    def fn():
        for i, j in [(1, 2)]:
            pass

    with pytest.raises(NotSupportedLoop, match="only 'for <var> in range"):
        _run(fn)


def test_not_supported_loop_at_decoration_time():
    def fn():
        for _ in [1, 2]:
            pass

    with pytest.raises(NotSupportedLoop):
        pysol(fn)


# ---------------------------------------------------------------------------
# While-loop detection
# ---------------------------------------------------------------------------

def test_detects_while_loop():
    def fn():
        i = 0
        while i < 10:
            i += 1

    a = _run(fn)
    assert len(a.while_loops) == 1
    assert a.while_loops[0].condition == "i < 10"


def test_detects_while_true():
    def fn():
        while True:
            break

    a = _run(fn)
    assert len(a.while_loops) == 1
    assert a.while_loops[0].condition == "True"


# ---------------------------------------------------------------------------
# Mixed + decorator execution
# ---------------------------------------------------------------------------

def test_mixed_analysis():
    def fn():
        arr = Array(1, 1, [10], [10], Type.Int)
        for i in range(10):
            pass
        j = 0
        while j < 5:
            j += 1

    a = _run(fn)
    assert len(a.arrays) == 1
    assert len(a.for_loops) == 1
    assert len(a.while_loops) == 1


def test_decorator_runs_function():
    @pysol
    def greet():
        return "hello"

    assert greet() == "hello"


# ---------------------------------------------------------------------------
# If-statement detection
# ---------------------------------------------------------------------------

def test_detects_valid_if():
    def fn():
        if func(a, b):
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "func(a, b)"


def test_detects_if_no_args():
    def fn():
        if check():
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "check()"


def test_detects_if_single_var():
    def fn():
        if pred(x):
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "pred(x)"


def test_detects_multiple_if_stmts():
    def fn():
        if f(a):
            pass
        if g(b, c):
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 2
    assert result.if_stmts[0].condition == "f(a)"
    assert result.if_stmts[1].condition == "g(b, c)"


def test_if_literal_arg_raises():
    def fn():
        if check(1):
            pass

    with pytest.raises(NotSupportedIfStatement):
        _run(fn)


def test_if_expression_arg_raises():
    def fn():
        if check(a + b):
            pass

    with pytest.raises(NotSupportedIfStatement):
        _run(fn)


# ---------------------------------------------------------------------------
# is_zero(...) if-conditions
# ---------------------------------------------------------------------------

def test_if_is_zero_var():
    def fn():
        if is_zero(b):
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "is_zero(b)"


def test_if_not_is_zero_var():
    def fn():
        if not is_zero(b):
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "not is_zero(b)"


def test_if_is_zero_csr_block_subscript():
    def fn():
        if is_zero(A[i][j]):
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "is_zero(A[i][j])"


def test_if_not_is_zero_csr_block_subscript():
    def fn():
        if not is_zero(A[i][j]):
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "not is_zero(A[i][j])"


def test_if_is_zero_too_many_args_raises():
    def fn():
        if is_zero(a, b):
            pass

    with pytest.raises(NotSupportedIfStatement):
        _run(fn)


def test_if_is_zero_literal_arg_raises():
    def fn():
        if is_zero(1):
            pass

    with pytest.raises(NotSupportedIfStatement):
        _run(fn)


def test_if_not_other_call_still_requires_variable_args():
    """The 'not expr(...)' relaxation is general, but expr's own argument
    rules (plain variables only, unless it's specifically is_zero) still
    apply."""
    def fn():
        if not check(a + b):
            pass

    with pytest.raises(NotSupportedIfStatement):
        _run(fn)


def test_if_not_other_call_with_var_args_is_valid():
    def fn():
        if not check(a, b):
            pass

    result = _run(fn)
    assert result.if_stmts[0].condition == "not check(a, b)"


# --- binary comparison forms (now valid) ------------------------------------

def test_if_var_op_var():
    def fn():
        if a < b:
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "a < b"


def test_if_var_op_const():
    def fn():
        if a > 0:
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "a > 0"


def test_if_const_op_var():
    def fn():
        if 0 < a:
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "0 < a"


def test_if_and_operator():
    def fn():
        if a and b:
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "a and b"


def test_if_or_operator():
    def fn():
        if a or b:
            pass

    result = _run(fn)
    assert len(result.if_stmts) == 1
    assert result.if_stmts[0].condition == "a or b"


# --- still-invalid forms -----------------------------------------------------

def test_if_plain_variable_condition_raises():
    def fn():
        if flag:
            pass

    with pytest.raises(NotSupportedIfStatement):
        _run(fn)


def test_if_nested_call_arg_raises():
    def fn():
        if check(len(xs)):
            pass

    with pytest.raises(NotSupportedIfStatement):
        _run(fn)


def test_if_const_op_const_raises():
    def fn():
        if 1 < 2:
            pass

    with pytest.raises(NotSupportedIfStatement):
        _run(fn)


def test_if_chained_comparison_raises():
    def fn():
        if a < b < c:
            pass

    with pytest.raises(NotSupportedIfStatement):
        _run(fn)


def test_if_three_operand_bool_raises():
    def fn():
        if a and b and c:
            pass

    with pytest.raises(NotSupportedIfStatement):
        _run(fn)


def test_not_supported_if_message():
    def fn():
        if flag:
            pass

    with pytest.raises(NotSupportedIfStatement, match="only 'if expr"):
        _run(fn)


def test_not_supported_if_at_decoration_time():
    def fn():
        if 1 < 2:
            pass

    with pytest.raises(NotSupportedIfStatement):
        pysol(fn)


def test_sorted_by_line():
    def fn():
        a = Array(1, 1, [5], [5], Type.Int)
        for i in range(5):
            b = Array(2, 2, [2, 2], [2, 2], Type.Float)

    a = _run(fn)
    lines = [info.line for info in a.arrays]
    assert lines == sorted(lines)


# ---------------------------------------------------------------------------
# @pysol_struct
# ---------------------------------------------------------------------------

def test_pysol_struct_registers_fields():
    @pysol_struct
    class _Vec2Struct:
        x: Type.Double
        y: Type.Float

    assert _struct_registry["_Vec2Struct"] == [("x", Type.Double), ("y", Type.Float)]


def test_pysol_struct_preserves_declaration_order():
    @pysol_struct
    class _OrderedStruct:
        c: Type.Int
        a: Type.Double
        b: Type.Float

    names = [name for name, _ in _struct_registry["_OrderedStruct"]]
    assert names == ["c", "a", "b"]


def test_pysol_struct_returns_usable_dataclass():
    """The decorated class stays a real, constructible Python object."""
    @pysol_struct
    class _PointStruct:
        x: Type.Double
        y: Type.Double

    p = _PointStruct(x=1.0, y=2.0)
    assert p.x == 1.0
    assert p.y == 2.0
    assert p == _PointStruct(x=1.0, y=2.0)


def test_pysol_struct_exposes_fields_on_class():
    @pysol_struct
    class _TaggedStruct:
        v: Type.Int

    assert _TaggedStruct.__pysol_struct_fields__ == [("v", Type.Int)]


def test_pysol_struct_rejects_string_field():
    with pytest.raises(UnsupportedStructFieldType, match="'s'"):
        @pysol_struct
        class _BadStruct:
            s: Type.String


def test_pysol_struct_rejects_bool_field():
    with pytest.raises(UnsupportedStructFieldType):
        @pysol_struct
        class _BadBoolStruct:
            flag: Type.Bool


def test_pysol_struct_rejects_empty_class():
    with pytest.raises(UnsupportedStructFieldType, match="no fields"):
        @pysol_struct
        class _EmptyStruct:
            pass


def test_pysol_struct_message_names_class():
    with pytest.raises(UnsupportedStructFieldType, match="_NamedBadStruct"):
        @pysol_struct
        class _NamedBadStruct:
            s: Type.String
