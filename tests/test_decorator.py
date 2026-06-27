import pytest
from pysol import Array, Type, pysol
from pysol.decorator import _analyse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(func):
    """Return the _Analysis of func without executing it."""
    return _analyse(func)


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


def test_for_loop_tuple_target():
    def fn():
        for i, j in [(1, 2)]:
            pass

    a = _run(fn)
    assert len(a.for_loops) == 1
    assert "i" in a.for_loops[0].target
    assert "j" in a.for_loops[0].target


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


def test_decorator_runs_function(capsys):
    @pysol
    def greet():
        return "hello"

    result = greet()
    assert result == "hello"
    out = capsys.readouterr().out
    assert "[pysol] greet" in out


def test_decorator_prints_array_info(capsys):
    @pysol
    def fn():
        arr = Array(1, 1, [10], [10], Type.Float)

    fn()
    out = capsys.readouterr().out
    assert "arr" in out
    assert "n_dims=1" in out
    assert "array_size=[10]" in out
    assert "Type.Float" in out


def test_decorator_prints_loop_info(capsys):
    @pysol
    def fn():
        for i in range(3):
            pass
        j = 0
        while j < 3:
            j += 1

    fn()
    out = capsys.readouterr().out
    assert "for i in range(3)" in out
    assert "while j < 3" in out


def test_function_name_in_output(capsys):
    @pysol
    def my_special_function():
        pass

    my_special_function()
    assert "[pysol] my_special_function" in capsys.readouterr().out


def test_sorted_by_line():
    def fn():
        a = Array(1, 1, [5], [5], Type.Int)
        for i in range(5):
            b = Array(2, 2, [2, 2], [2, 2], Type.Float)

    a = _run(fn)
    lines = [info.line for info in a.arrays]
    assert lines == sorted(lines)
