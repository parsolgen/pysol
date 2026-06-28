"""Tests for .sol file generation."""
import os
import textwrap

import pytest

from pysol import Array, KernelArgument, NotImportedKernel, Type, import_kernel, pysol
from pysol.sol_generator import (
    generate_sol, _collect_kernel_imports, _parse_size_list, _gen_import_arg_decl,
)
from pysol.decorator import _analyse
import ast


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sol(func) -> str:
    """Return the .sol content for func without writing a file."""
    return generate_sol(func, _analyse(func))


def _lines(func) -> list:
    return _sol(func).splitlines()


# ---------------------------------------------------------------------------
# _parse_size_list
# ---------------------------------------------------------------------------

def test_parse_size_list_single():
    assert _parse_size_list("[10]") == ["10"]


def test_parse_size_list_multiple():
    assert _parse_size_list("[30, 40, 50]") == ["30", "40", "50"]


# ---------------------------------------------------------------------------
# Import declarations
# ---------------------------------------------------------------------------

def test_import_section_empty():
    def fn():
        pass

    assert "import" not in _sol(fn)


def test_import_single_kernel():
    def fn():
        k = import_kernel("compute", [])

    out = _sol(fn)
    assert "import compute();" in out


def test_import_multiple_kernels():
    def fn():
        ka = import_kernel("alpha", [])
        kb = import_kernel("beta", [])

    out = _sol(fn)
    assert "import alpha();" in out
    assert "import beta();" in out


# ---------------------------------------------------------------------------
# _gen_import_arg_decl unit tests
# ---------------------------------------------------------------------------

def test_gen_import_arg_scalar_in():
    decls = _gen_import_arg_decl(KernelArgument("x", True, Type.Int))
    assert decls == ["in int x"]


def test_gen_import_arg_scalar_out():
    decls = _gen_import_arg_decl(KernelArgument("y", False, Type.Double))
    assert decls == ["out double y"]


def test_gen_import_arg_scalar_float():
    decls = _gen_import_arg_decl(KernelArgument("v", True, Type.Float))
    assert decls == ["in float v"]


def test_gen_import_arg_scalar_string():
    decls = _gen_import_arg_decl(KernelArgument("s", False, Type.String))
    assert decls == ["out string s"]


def test_gen_import_arg_scalar_bool():
    decls = _gen_import_arg_decl(KernelArgument("flag", True, Type.Bool))
    assert decls == ["in bool flag"]


def test_gen_import_arg_array_element_1d():
    # KernelArgument("a", True, Type.Float, n_dims=1)
    # → ["in a_n1", "in Array<float> a[a_n1]"]
    decls = _gen_import_arg_decl(KernelArgument("a", True, Type.Float, n_dims=1))
    assert decls == ["in a_n1", "in Array<float> a[a_n1]"]


def test_gen_import_arg_array_element_2d_out():
    # KernelArgument("a", False, Type.Float, n_dims=2)
    # → ["in a_n1", "in a_n2", "out Array<float> a[a_n1][a_n2]"]
    decls = _gen_import_arg_decl(KernelArgument("a", False, Type.Float, n_dims=2))
    assert decls == ["in a_n1", "in a_n2", "out Array<float> a[a_n1][a_n2]"]


def test_gen_import_arg_array_element_3d_double():
    decls = _gen_import_arg_decl(KernelArgument("blk", True, Type.Double, n_dims=3))
    assert decls == [
        "in blk_n1", "in blk_n2", "in blk_n3",
        "in Array<double> blk[blk_n1][blk_n2][blk_n3]",
    ]


# ---------------------------------------------------------------------------
# Import declarations with argument lists
# ---------------------------------------------------------------------------

def test_import_with_scalar_args():
    def fn():
        k = import_kernel("add", [
            KernelArgument("x", True, Type.Int),
            KernelArgument("y", False, Type.Double),
        ])

    out = _sol(fn)
    assert "import add(in int x, out double y);" in out


def test_import_with_array_element_arg():
    def fn():
        k = import_kernel("process", [
            KernelArgument("a", False, Type.Float, n_dims=2),
        ])

    out = _sol(fn)
    assert "import process(in a_n1, in a_n2, out Array<float> a[a_n1][a_n2]);" in out


def test_import_mixed_scalar_and_array():
    def fn():
        k = import_kernel("kern", [
            KernelArgument("n",   True,  Type.Int),
            KernelArgument("blk", False, Type.Double, n_dims=3),
            KernelArgument("s",   True,  Type.String),
        ])

    out = _sol(fn)
    assert (
        "import kern("
        "in int n, "
        "in blk_n1, in blk_n2, in blk_n3, out Array<double> blk[blk_n1][blk_n2][blk_n3], "
        "in string s"
        ");"
    ) in out


# ---------------------------------------------------------------------------
# Function declaration
# ---------------------------------------------------------------------------

def test_function_declaration():
    def my_func():
        pass

    out = _sol(my_func)
    assert "export sub my_func()" in out


def test_function_has_braces():
    def fn():
        pass

    lines = _lines(fn)
    assert "{" in lines
    assert "}" in lines


# ---------------------------------------------------------------------------
# Array declarations
# ---------------------------------------------------------------------------

def test_array_decl_1d_float():
    def fn():
        arr = Array(1, 1, [10], [10], Type.Float)

    out = _sol(fn)
    assert "Array<float[10]> arr[10];" in out


def test_array_decl_2d_double():
    def fn():
        b = Array(2, 3, [10, 20], [30, 40, 50], Type.Double)

    out = _sol(fn)
    assert "Array<double[30][40][50]> b[10][20];" in out


def test_array_decl_int_type():
    def fn():
        x = Array(1, 2, [5], [3, 4], Type.Int)

    out = _sol(fn)
    assert "Array<int[3][4]> x[5];" in out


def test_array_decl_string_type():
    def fn():
        s = Array(1, 1, [8], [8], Type.String)

    out = _sol(fn)
    assert "Array<string[8]> s[8];" in out


# ---------------------------------------------------------------------------
# For-loop generation
# ---------------------------------------------------------------------------

def test_for_range_one_arg():
    def fn():
        for i in range(10):
            pass

    out = _sol(fn)
    assert "for (int i = 0; i < 10; i++)" in out


def test_for_range_two_args():
    def fn():
        for i in range(2, 8):
            pass

    out = _sol(fn)
    assert "for (int i = 2; i < 8; i++)" in out


def test_for_range_three_args():
    def fn():
        for i in range(0, 10, 2):
            pass

    out = _sol(fn)
    assert "for (int i = 0; i < 10; i += 2)" in out


def test_for_body_indented():
    def fn():
        k = import_kernel("work", [])
        for i in range(3):
            k()

    out = _sol(fn)
    # The kernel call inside the loop must be indented further (skip the import line)
    inner = [l for l in out.splitlines() if "work()" in l and not l.strip().startswith("import")][0]
    assert inner.startswith("        ")  # 8 spaces = 2 levels


# ---------------------------------------------------------------------------
# While-loop generation
# ---------------------------------------------------------------------------

def test_while_var_op_const():
    def fn():
        while i < 10:
            pass

    out = _sol(fn)
    assert "while (i < 10) {" in out


def test_while_bool_cond():
    def fn():
        while flag:
            pass

    # 'if flag' would fail validation, but while(flag) is a raw Name — not
    # subject to @pysol if-statement rules, just generated as-is.
    out = _sol(fn)
    assert "while (flag) {" in out


# ---------------------------------------------------------------------------
# If-statement generation
# ---------------------------------------------------------------------------

def test_if_call_condition():
    def fn():
        if check(a, b):
            pass

    out = _sol(fn)
    assert "if (check(a, b)) {" in out


def test_if_comparison():
    def fn():
        if a < b:
            pass

    out = _sol(fn)
    assert "if (a < b) {" in out


def test_if_and_condition():
    def fn():
        if a and b:
            pass

    out = _sol(fn)
    assert "if (a && b) {" in out


def test_if_or_condition():
    def fn():
        if a or b:
            pass

    out = _sol(fn)
    assert "if (a || b) {" in out


# ---------------------------------------------------------------------------
# Kernel call generation
# ---------------------------------------------------------------------------

def test_kernel_call_no_args():
    def fn():
        k = import_kernel("noop", [])
        k()

    out = _sol(fn)
    assert "noop();" in out


def test_kernel_call_scalar_args():
    def fn():
        k = import_kernel("add", [])
        k(n + 1, 123)

    out = _sol(fn)
    assert "add(n + 1, 123);" in out


def test_kernel_call_not_imported_raises():
    def fn():
        mystery()

    with pytest.raises(NotImportedKernel, match="mystery"):
        _sol(fn)


# ---------------------------------------------------------------------------
# Array argument expansion
# ---------------------------------------------------------------------------

def test_array_ref_subscript_expansion():
    """a[version, i][j] → <block_sizes...>, a{version}[i][j]"""
    def fn():
        arr = Array(1, 2, [10], [4, 8], Type.Float)
        k = import_kernel("kern", [])
        k(arr[version, i][j])

    out = _sol(fn)
    # block sizes 4 and 8 should appear as extra args before the ref
    assert "4, 8, arr{version}[i][j]" in out


def test_array_get_expansion():
    """arr.get(v, [i, j]) → <block_sizes...>, arr{v}[i][j]"""
    def fn():
        arr = Array(2, 1, [5, 5], [3], Type.Int)
        k = import_kernel("kern", [])
        k(arr.get(v, [i, j]))

    out = _sol(fn)
    assert "3, arr{v}[i][j]" in out


# ---------------------------------------------------------------------------
# .sol file written to disk
# ---------------------------------------------------------------------------

def test_sol_file_created(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Write a temp source file so inspect.getfile works
    src = tmp_path / "sample.py"
    src.write_text(textwrap.dedent("""
        from pysol import Array, Type, import_kernel, pysol

        @pysol
        def my_kernel():
            arr = Array(1, 1, [5], [5], Type.Int)
    """))

    import importlib.util
    spec = importlib.util.spec_from_file_location("sample", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    sol_file = tmp_path / "my_kernel.sol"
    assert sol_file.exists()
    content = sol_file.read_text()
    assert "export sub my_kernel()" in content
    assert "Array<int[5]> arr[5];" in content
