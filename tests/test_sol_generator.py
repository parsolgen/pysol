"""Tests for .sol file generation."""
import os
import textwrap

import pytest

import pysol as _pysol_mod
from pysol import (
    Array, ArrayOperationError, CPU, CSRMatrix, GOOPAX, GrowingArray, KernelArgument,
    LATEST_VERSION, NotImportedKernel, UnsupportedCallError, UnsupportedVariableType, Type,
    import_kernel, pysol, pysol_struct, pysol_update_array, reset,
)
from pysol.sol_generator import (
    generate_sol, _collect_kernel_imports, _parse_size_list, _gen_import_arg_decl,
    _annotation_to_sol_type,
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


# Module-level stubs referenced inside inner fn() bodies used as AST fixtures.
a = b = i = j = v = var = version = e1 = e2 = flag = x = y = z = n = None

def check(*_): return True  # type: ignore[return]


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
# Import declarations — supported_backends_list → @cpu / @goopax annotation
# ---------------------------------------------------------------------------

def test_import_backend_cpu_and_goopax():
    def fn():
        print_k = import_kernel("print", [KernelArgument("value", True, Type.Int)], [CPU, GOOPAX])

    out = _sol(fn)
    assert "import print(in int value) : @cpu, @goopax;" in out


def test_import_backend_cpu_only_explicit():
    def fn():
        k = import_kernel("cpu_only", [KernelArgument("value", True, Type.Int)], [CPU])

    out = _sol(fn)
    assert "import cpu_only(in int value) : @cpu;" in out


def test_import_backend_goopax_only():
    def fn():
        k = import_kernel("goopax_only", [KernelArgument("value", True, Type.Int)], [GOOPAX])

    out = _sol(fn)
    assert "import goopax_only(in int value) : @goopax;" in out


def test_import_backend_omitted_generates_no_annotation():
    def fn():
        k = import_kernel("legacy", [KernelArgument("value", True, Type.Int)])

    out = _sol(fn)
    assert "import legacy(in int value);" in out
    assert "@cpu" not in out
    assert "@goopax" not in out


def test_import_backend_empty_list_generates_no_annotation():
    def fn():
        k = import_kernel("empty", [KernelArgument("value", True, Type.Int)], [])

    out = _sol(fn)
    assert "import empty(in int value);" in out
    assert "@cpu" not in out
    assert "@goopax" not in out


def test_import_backend_order_independent():
    def fn():
        k = import_kernel("flipped", [KernelArgument("value", True, Type.Int)], [GOOPAX, CPU])

    out = _sol(fn)
    assert "import flipped(in int value) : @cpu, @goopax;" in out


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
    # → ["in int a_n1", "in Array<float> a[a_n1]"]
    decls = _gen_import_arg_decl(KernelArgument("a", True, Type.Float, n_dims=1))
    assert decls == ["in int a_n1", "in Array<float> a[a_n1]"]


def test_gen_import_arg_array_element_2d_out():
    # KernelArgument("a", False, Type.Float, n_dims=2)
    # → ["in int a_n1", "in int a_n2", "out Array<float> a[a_n1][a_n2]"]
    decls = _gen_import_arg_decl(KernelArgument("a", False, Type.Float, n_dims=2))
    assert decls == ["in int a_n1", "in int a_n2", "out Array<float> a[a_n1][a_n2]"]


def test_gen_import_arg_array_element_3d_double():
    decls = _gen_import_arg_decl(KernelArgument("blk", True, Type.Double, n_dims=3))
    assert decls == [
        "in int blk_n1", "in int blk_n2", "in int blk_n3",
        "in Array<double> blk[blk_n1][blk_n2][blk_n3]",
    ]


def test_gen_import_arg_csr_block_in():
    decls = _gen_import_arg_decl(KernelArgument("a", True, Type.CSRBlock(Type.Double)))
    assert decls == ["in CSRBlock<double> a"]


def test_gen_import_arg_csr_block_out():
    decls = _gen_import_arg_decl(KernelArgument("b", False, Type.CSRBlock(Type.Float)))
    assert decls == ["out CSRBlock<float> b"]


def test_gen_import_arg_csr_block_int():
    decls = _gen_import_arg_decl(KernelArgument("c", True, Type.CSRBlock(Type.Int)))
    assert decls == ["in CSRBlock<int> c"]


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
    assert "import process(in int a_n1, in int a_n2, out Array<float> a[a_n1][a_n2]);" in out


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
        "in int blk_n1, in int blk_n2, in int blk_n3, out Array<double> blk[blk_n1][blk_n2][blk_n3], "
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
# is_single_version -> SingleVersionArray<...> instead of Array<...>
# ---------------------------------------------------------------------------

def test_array_decl_single_version_positional():
    def fn():
        b = Array(1, 1, [9], [2], Type.Double, True)

    out = _sol(fn)
    assert "SingleVersionArray<double[2]> b[9];" in out


def test_array_decl_single_version_keyword():
    def fn():
        b = Array(1, 1, [9], [2], Type.Double, is_single_version=True)

    out = _sol(fn)
    assert "SingleVersionArray<double[2]> b[9];" in out


def test_array_decl_single_version_false_is_plain_array():
    def fn():
        b = Array(1, 1, [9], [2], Type.Double, is_single_version=False)

    out = _sol(fn)
    assert "Array<double[2]> b[9];" in out
    assert "SingleVersionArray" not in out


def test_array_decl_single_version_default_is_plain_array():
    def fn():
        b = Array(1, 1, [9], [2], Type.Double)

    out = _sol(fn)
    assert "Array<double[2]> b[9];" in out
    assert "SingleVersionArray" not in out


# ---------------------------------------------------------------------------
# CSRMatrix declaration
# ---------------------------------------------------------------------------

def test_csr_matrix_decl_bare_assign():
    """
    .sol argument order (dim1, dim2, nnz, n_blocks_1, n_blocks_2, ia, ja, a)
    differs from the Python constructor's own order (size, n_blocks, nnz,
    ia, ja, a, type).
    """
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)

    out = _sol(fn)
    assert "BlockedCSRMatrix<double[][]> A = {4, 4, 6, 2, 2, ia, ja, a};" in out


def test_csr_matrix_decl_ann_assign():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Float)
        A: CSRMatrix = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Float)

    out = _sol(fn)
    assert "BlockedCSRMatrix<float[][]> A = {4, 4, 6, 2, 2, ia, ja, a};" in out


def test_csr_matrix_decl_int_type():
    def fn():
        ia = Array(1, 1, [3], [1], Type.Int)
        ja = Array(1, 1, [4], [1], Type.Int)
        a = Array(1, 1, [4], [1], Type.Int)
        A = CSRMatrix([2, 2], [1, 1], 4, ia, ja, a, Type.Int)

    out = _sol(fn)
    assert "BlockedCSRMatrix<int[][]> A = {2, 2, 4, 1, 1, ia, ja, a};" in out


def test_csr_matrix_ia_wrong_type_raises():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Double)  # should be Type.Int
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)

    with pytest.raises(ArrayOperationError):
        _sol(fn)


def test_csr_matrix_ja_wrong_type_raises():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Float)  # should be Type.Int
        a = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)

    with pytest.raises(ArrayOperationError):
        _sol(fn)


def test_csr_matrix_a_wrong_type_raises():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Float)  # matrix type is Type.Double
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)

    with pytest.raises(ArrayOperationError):
        _sol(fn)


def test_csr_matrix_ia_wrong_ndims_raises():
    def fn():
        ia = Array(2, 1, [5, 5], [1], Type.Int)  # must be 1-D
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)

    with pytest.raises(ArrayOperationError):
        _sol(fn)


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
# is_zero(...) condition generation — renders as the .sol macro %zero(...)
# ---------------------------------------------------------------------------

def test_if_is_zero_var():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        arr = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, arr, Type.Double)
        bl: Type.CSRBlock(Type.Double) = A[0][1]
        if is_zero(bl):
            pass

    out = _sol(fn)
    assert "if (%zero(bl)) {" in out


def test_if_not_is_zero_var():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        arr = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, arr, Type.Double)
        bl: Type.CSRBlock(Type.Double) = A[0][1]
        if not is_zero(bl):
            pass

    out = _sol(fn)
    assert "if (!%zero(bl)) {" in out


def test_if_is_zero_csr_block_subscript():
    """if is_zero(A[i][j]): — the block subscript is passed inline, unexpanded."""
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        arr = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, arr, Type.Double)
        if is_zero(A[i][j]):
            pass

    out = _sol(fn)
    assert "if (%zero(A[i][j])) {" in out


def test_if_not_is_zero_csr_block_subscript():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        arr = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, arr, Type.Double)
        if not is_zero(A[i][j]):
            pass

    out = _sol(fn)
    assert "if (!%zero(A[i][j])) {" in out


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
    """a[version, i][j] → %block_n(a, 0), %block_n(a, 1), a{version}[i][j]"""
    def fn():
        arr = Array(1, 2, [10], [4, 8], Type.Float)
        k = import_kernel("kern", [])
        k(arr[version, i][j])

    out = _sol(fn)
    assert "%block_n(arr, 0), %block_n(arr, 1), arr{version}[i][j]" in out


# ---------------------------------------------------------------------------
# LATEST_VERSION — renders as the .sol token %latest
# ---------------------------------------------------------------------------

def test_latest_version_known_array_kernel_arg():
    """arr[LATEST_VERSION, i][j] → %block_n(arr, 0), %block_n(arr, 1), arr{%latest}[i][j]"""
    def fn():
        arr = Array(1, 2, [10], [4, 8], Type.Float)
        k = import_kernel("kern", [])
        k(arr[LATEST_VERSION, i][j])

    out = _sol(fn)
    assert "%block_n(arr, 0), %block_n(arr, 1), arr{%latest}[i][j]" in out


def test_latest_version_known_array_get_expansion():
    """arr.get(LATEST_VERSION, [i, j]) → %block_n(arr, 0), arr{%latest}[i][j]"""
    def fn():
        arr = Array(2, 1, [5, 5], [3], Type.Int)
        k = import_kernel("kern", [])
        k(arr.get(LATEST_VERSION, [i, j]))

    out = _sol(fn)
    assert "kern(%block_n(arr, 0), arr{%latest}[i][j]);" in out


def test_latest_version_csr_matrix_kernel_arg():
    """A[LATEST_VERSION, i][j][k] → A{%latest}[i][j][k] — no %block_n expansion for a CSRMatrix."""
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        arr = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, arr, Type.Double)
        foo = import_kernel("foo", [])
        foo(A[LATEST_VERSION, i][j][k])

    out = _sol(fn)
    assert "foo(A{%latest}[i][j][k]);" in out


def test_latest_version_bare_expression():
    """b = A[LATEST_VERSION, i][j] as a direct (non-kernel-call) expression."""
    def fn():
        arr = Array(1, 1, [10], [4], Type.Double)
        x: Type.Double = arr[LATEST_VERSION, i][0]

    out = _sol(fn)
    assert "double x = arr{%latest}[i][0];" in out


def test_array_get_expansion():
    """arr.get(v, [i, j]) → %block_n(arr, 0), arr{v}[i][j]"""
    def fn():
        arr = Array(2, 1, [5, 5], [3], Type.Int)
        k = import_kernel("kern", [])
        k(arr.get(v, [i, j]))

    out = _sol(fn)
    assert "kern(%block_n(arr, 0), arr{v}[i][j]);" in out


def test_array_get_scalar_and_array_args():
    """kernel(1, var, arr.get(version, [e1, e2])) → kernel(1, var, %block_n(arr, 0), arr{version}[e1][e2])"""
    def fn():
        arr = Array(2, 1, [5, 5], [3], Type.Double)
        k = import_kernel("kernel", [])
        k(1, var, arr.get(version, [e1, e2]))

    out = _sol(fn)
    assert "kernel(1, var, %block_n(arr, 0), arr{version}[e1][e2]);" in out


def test_array_get_single_coord():
    """arr.get(v, [i]) → %block_n(arr, 0), arr{v}[i]"""
    def fn():
        arr = Array(1, 1, [5], [3], Type.Int)
        k = import_kernel("kern", [])
        k(arr.get(v, [i]))

    out = _sol(fn)
    assert "kern(%block_n(arr, 0), arr{v}[i]);" in out


def test_plain_subscript_single_index():
    """arr[i] (no version) → %block_n(arr, 0), arr[i]"""
    def fn():
        arr = Array(1, 1, [10], [4], Type.Float)
        k = import_kernel("kern", [])
        k(arr[i])

    out = _sol(fn)
    assert "kern(%block_n(arr, 0), arr[i]);" in out


def test_plain_subscript_two_indices():
    """arr[i][j] (no version, two subscripts) → %block_n * n_block_dims, arr[i][j]"""
    def fn():
        arr = Array(2, 2, [5, 5], [3, 3], Type.Double)
        k = import_kernel("kern", [])
        k(arr[i][j])

    out = _sol(fn)
    assert "kern(%block_n(arr, 0), %block_n(arr, 1), arr[i][j]);" in out


def test_plain_subscript_mixed_with_scalar_args():
    """Scalar args around a plain array subscript are passed through unchanged."""
    def fn():
        arr = Array(1, 1, [8], [4], Type.Int)
        k = import_kernel("kern", [])
        k(n, arr[i], n)

    out = _sol(fn)
    assert "kern(n, %block_n(arr, 0), arr[i], n);" in out


# ---------------------------------------------------------------------------
# CSRMatrix / CSRBlock usage: A[i][j] as a kernel-call argument and as the
# RHS of a Type.CSRBlock(...)-annotated variable. A is never added to
# self._arrays, so A[i][j] falls through to the generic Subscript renderer
# (unlike Array's special %block_n expansion above) and is emitted bare.
# ---------------------------------------------------------------------------

def test_csr_block_inline_kernel_call_arg():
    """foo(A[i][j]) → foo(A[i][j]) — no %block_n expansion for a CSRMatrix."""
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)
        foo = import_kernel("foo", [KernelArgument("a", True, Type.CSRBlock(Type.Double))])
        foo(A[i][j])

    out = _sol(fn)
    assert "foo(A[i][j]);" in out


def test_csr_block_inline_kernel_call_mixed_with_scalar():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)
        foo = import_kernel("foo", [])
        foo(n, A[i][j], n)

    out = _sol(fn)
    assert "foo(n, A[i][j], n);" in out


def test_csr_block_variable_declaration_from_subscript():
    """b : Type.CSRBlock(Type.Double) = A[i][j] → CSRBlock<double> b = A[i][j];"""
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)
        b: Type.CSRBlock(Type.Double) = A[i][j]

    out = _sol(fn)
    assert "CSRBlock<double> b = A[i][j];" in out


def test_csr_matrix_import_line_with_csr_block_arg():
    def fn():
        foo = import_kernel("foo", [KernelArgument("a", True, Type.CSRBlock(Type.Double))])

    out = _sol(fn)
    assert "import foo(in CSRBlock<double> a);" in out


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


# ---------------------------------------------------------------------------
# _annotation_to_sol_type unit tests
# ---------------------------------------------------------------------------

def test_annotation_type_int():
    import ast
    node = ast.parse("x: Type.Int = 0", mode="single").body[0]
    assert _annotation_to_sol_type(node.annotation) == "int"


def test_annotation_type_float():
    import ast
    node = ast.parse("x: Type.Float = 0.0", mode="single").body[0]
    assert _annotation_to_sol_type(node.annotation) == "float"


def test_annotation_type_double():
    import ast
    node = ast.parse("x: Type.Double = 0.0", mode="single").body[0]
    assert _annotation_to_sol_type(node.annotation) == "double"


def test_annotation_type_string():
    import ast
    node = ast.parse("x: Type.String = ''", mode="single").body[0]
    assert _annotation_to_sol_type(node.annotation) == "string"


def test_annotation_type_bool():
    import ast
    node = ast.parse("x: Type.Bool = True", mode="single").body[0]
    assert _annotation_to_sol_type(node.annotation) == "bool"


def test_annotation_pysol_qualified():
    """pysol.Type.Int and pysol.Int both map to 'int'."""
    import ast
    for src in ("x: pysol.Type.Int = 0", "x: pysol.Int = 0"):
        node = ast.parse(src, mode="single").body[0]
        assert _annotation_to_sol_type(node.annotation) == "int", src


def test_annotation_csr_block_double():
    import ast
    node = ast.parse("x: Type.CSRBlock(Type.Double) = y", mode="single").body[0]
    assert _annotation_to_sol_type(node.annotation) == "CSRBlock<double>"


def test_annotation_csr_block_int():
    import ast
    node = ast.parse("x: Type.CSRBlock(Type.Int) = y", mode="single").body[0]
    assert _annotation_to_sol_type(node.annotation) == "CSRBlock<int>"


def test_annotation_unknown_returns_none():
    import ast
    node = ast.parse("x: int = 0", mode="single").body[0]
    assert _annotation_to_sol_type(node.annotation) is None


# ---------------------------------------------------------------------------
# Simple variable declaration generation
# ---------------------------------------------------------------------------

def test_simple_var_int_annotation():
    def fn():
        x: Type.Int = 5

    out = _sol(fn)
    assert "int x = 5;" in out


def test_simple_var_float_annotation():
    def fn():
        y: Type.Float = 1.5

    out = _sol(fn)
    assert "float y = 1.5;" in out


def test_simple_var_double_annotation():
    def fn():
        z: Type.Double = 3.14

    out = _sol(fn)
    assert "double z = 3.14;" in out


def test_simple_var_string_annotation():
    def fn():
        n: Type.String = "hello"

    out = _sol(fn)
    assert 'string n = "hello";' in out


def test_simple_var_bool_annotation_true():
    def fn():
        flag: Type.Bool = True

    out = _sol(fn)
    assert "bool flag = true;" in out


def test_simple_var_bool_annotation_false():
    def fn():
        flag: Type.Bool = False

    out = _sol(fn)
    assert "bool flag = false;" in out


def test_simple_var_unknown_annotation_raises():
    def fn():
        x: int = 42

    with pytest.raises(UnsupportedVariableType, match="'x'"):
        _sol(fn)


def test_simple_var_no_annotation_raises():
    def fn():
        x = 42

    with pytest.raises(UnsupportedVariableType, match="'x'"):
        _sol(fn)


def test_unsupported_var_message_lists_supported_types():
    def fn():
        x = 42

    with pytest.raises(UnsupportedVariableType, match="Type.Int"):
        _sol(fn)


def test_simple_var_pysol_qualified_annotation():
    """pysol.Type.Int annotation works the same as Type.Int."""
    def fn():
        x: _pysol_mod.Type.Int = 10

    out = _sol(fn)
    assert "int x = 10;" in out


# ---------------------------------------------------------------------------
# Binary-expression parenthesisation
# ---------------------------------------------------------------------------

def test_binop_complex_nested_parens():
    """User's example: dx2*dy2 / (2.0*a*(dx2+dy2)) preserves both paren groups."""
    def fn():
        dt: Type.Double = x * y / (2.0 * z * (x + y))

    out = _sol(fn)
    assert "double dt = x * y / (2.0 * z * (x + y));" in out


def test_binop_lower_prec_left_child_gets_parens():
    """(x + y) * z — addition on the left of multiplication needs parens."""
    def fn():
        n: Type.Int = (x + y) * z

    out = _sol(fn)
    assert "int n = (x + y) * z;" in out


def test_binop_higher_prec_right_child_no_parens():
    """x + y * z — multiplication on the right of addition needs no parens."""
    def fn():
        n: Type.Int = x + y * z

    out = _sol(fn)
    assert "int n = x + y * z;" in out


def test_binop_right_div_same_prec_gets_parens():
    """x / (y / z) — equal-prec right child of / needs parens."""
    def fn():
        dt: Type.Double = x / (y / z)

    out = _sol(fn)
    assert "double dt = x / (y / z);" in out


def test_binop_right_sub_same_prec_gets_parens():
    """x - (y - z) — equal-prec right child of - needs parens."""
    def fn():
        n: Type.Int = x - (y - z)

    out = _sol(fn)
    assert "int n = x - (y - z);" in out


def test_binop_right_add_same_prec_no_parens():
    """x + y + z — right-assoc addition is fine without parens."""
    def fn():
        n: Type.Int = x + y + z

    out = _sol(fn)
    assert "int n = x + y + z;" in out


# ---------------------------------------------------------------------------
# @ (dot product) operator generation
# ---------------------------------------------------------------------------

def test_matmul_bare_arrays():
    """a @ b -> (a, b) for two whole blocked-vector Arrays."""
    def fn():
        Rl = Array(1, 1, [9], [2], Type.Double)
        Rr = Array(1, 1, [9], [2], Type.Double)
        rho: Type.Double = Rl @ Rr

    out = _sol(fn)
    assert "double rho = (Rl, Rr);" in out


def test_matmul_versioned_operands():
    """a[v,] @ b[v,] -> (a{v}, b{v}) — whole vector at a given version."""
    def fn():
        Rl = Array(1, 1, [9], [2], Type.Double)
        Rr = Array(1, 1, [9], [2], Type.Double)
        it: Type.Int = 1
        rho: Type.Double = Rl[it,] @ Rr[it,]

    out = _sol(fn)
    assert "double rho = (Rl{it}, Rr{it});" in out


def test_matmul_mixed_bare_and_versioned_operands():
    def fn():
        Rl = Array(1, 1, [9], [2], Type.Double)
        Rr = Array(1, 1, [9], [2], Type.Double)
        it: Type.Int = 1
        rho: Type.Double = Rl[it,] @ Rr

    out = _sol(fn)
    assert "double rho = (Rl{it}, Rr);" in out


def test_matmul_reassignment():
    def fn():
        Rl = Array(1, 1, [9], [2], Type.Double)
        Rr = Array(1, 1, [9], [2], Type.Double)
        rho: Type.Double = 0.0
        rho = Rl @ Rr

    out = _sol(fn)
    assert "rho = (Rl, Rr);" in out


# ---------------------------------------------------------------------------
# +, -, * on vectors (version-tagged 1-D arrays of 1-D blocks)
# ---------------------------------------------------------------------------

def test_vector_add():
    """a[v,] + b[v,] -> a{v} + b{v}"""
    def fn():
        a = Array(1, 1, [9], [2], Type.Double)
        b = Array(1, 1, [9], [2], Type.Double)
        k = import_kernel("k", [KernelArgument("v", True, Type.Double, 1)])
        it: Type.Int = 1
        k(a[it,] + b[it,])

    out = _sol(fn)
    assert "k(a{it} + b{it});" in out


def test_vector_sub():
    def fn():
        a = Array(1, 1, [9], [2], Type.Double)
        b = Array(1, 1, [9], [2], Type.Double)
        k = import_kernel("k", [KernelArgument("v", True, Type.Double, 1)])
        it: Type.Int = 1
        k(a[it,] - b[it,])

    out = _sol(fn)
    assert "k(a{it} - b{it});" in out


def test_vector_mul_vector():
    def fn():
        a = Array(1, 1, [9], [2], Type.Double)
        b = Array(1, 1, [9], [2], Type.Double)
        k = import_kernel("k", [KernelArgument("v", True, Type.Double, 1)])
        it: Type.Int = 1
        k(a[it,] * b[it,])

    out = _sol(fn)
    assert "k(a{it} * b{it});" in out


def test_vector_mul_constant():
    """a[v,] * c -> a{v} * c, where c is a plain scalar of the same type."""
    def fn():
        a = Array(1, 1, [9], [2], Type.Double)
        k = import_kernel("k", [KernelArgument("v", True, Type.Double, 1)])
        it: Type.Int = 1
        c: Type.Double = 2.0
        k(a[it,] * c)

    out = _sol(fn)
    assert "k(a{it} * c);" in out


def test_vector_constant_mul_left():
    """c * a[v,] -> c * a{v} (scalar on the left)."""
    def fn():
        a = Array(1, 1, [9], [2], Type.Double)
        k = import_kernel("k", [KernelArgument("v", True, Type.Double, 1)])
        it: Type.Int = 1
        c: Type.Double = 2.0
        k(c * a[it,])

    out = _sol(fn)
    assert "k(c * a{it});" in out


def test_vector_nested_axpy_style():
    """a[v+1,] + c * b[v,] -> a{v + 1} + c * b{v} - precedence keeps this unparenthesized."""
    def fn():
        a = Array(1, 1, [9], [2], Type.Double)
        b = Array(1, 1, [9], [2], Type.Double)
        k = import_kernel("k", [KernelArgument("v", True, Type.Double, 1)])
        it: Type.Int = 1
        c: Type.Double = 2.0
        k(a[it + 1,] + c * b[it,])

    out = _sol(fn)
    assert "k(a{it + 1} + c * b{it});" in out


def test_vector_axpy_as_kernel_arg():
    """x[v+1,] + a_c * Pr[v,] used as a kernel-call argument."""
    def fn():
        x = Array(1, 1, [9], [2], Type.Double)
        Pr = Array(1, 1, [9], [2], Type.Double)
        k = import_kernel("k", [KernelArgument("v", True, Type.Double, 1)])
        it: Type.Int = 1
        a_c: Type.Double = 0.5
        k(x[it,] + a_c * Pr[it,])

    out = _sol(fn)
    assert "k(x{it} + a_c * Pr{it});" in out


def test_vector_whole_assignment_statement():
    """x[v,] = <vector expr>; -> x{v} = <expr>; - bicg.sol's own
    "x{1} = x + a_c * Pr;" style, no per-block loop or kernel call needed."""
    def fn():
        x = Array(1, 1, [9], [2], Type.Double)
        Pr = Array(1, 1, [9], [2], Type.Double)
        it: Type.Int = 1
        a_c: Type.Double = 0.5
        x[it + 1,] = x[it,] + a_c * Pr[it,]

    out = _sol(fn)
    assert "x{it + 1} = x{it} + a_c * Pr{it};" in out


def test_vector_whole_assignment_subtraction():
    def fn():
        Rr = Array(1, 1, [9], [2], Type.Double)
        Zr = Array(1, 1, [9], [2], Type.Double)
        it: Type.Int = 1
        a_c: Type.Double = 0.5
        Rr[it + 1,] = Rr[it,] - a_c * Zr[9,]

    out = _sol(fn)
    assert "Rr{it + 1} = Rr{it} - a_c * Zr{9};" in out


# ---------------------------------------------------------------------------
# "=" plain whole-vector copy (no arithmetic operator)
# ---------------------------------------------------------------------------

def test_vector_plain_copy_from_versioned_source():
    """A = B[1,] -> A = B{1}; (bicg.sol's own "Rr = B;" style - A stays bare)."""
    def fn():
        A = Array(1, 1, [9], [2], Type.Double)
        B = Array(1, 1, [9], [2], Type.Double)
        A = B[1,]

    out = _sol(fn)
    assert "A = B{1};" in out


def test_vector_plain_copy_both_bare():
    """A = B -> A = B; - matches bicg.sol's "Rr = B;" exactly, no braces at all."""
    def fn():
        A = Array(1, 1, [9], [2], Type.Double)
        B = Array(1, 1, [9], [2], Type.Double)
        A = B

    out = _sol(fn)
    assert "A = B;" in out


def test_vector_plain_copy_versioned_target():
    """A[0,] = B[1,] -> A{0} = B{1};"""
    def fn():
        A = Array(1, 1, [9], [2], Type.Double)
        B = Array(1, 1, [9], [2], Type.Double)
        A[0,] = B[1,]

    out = _sol(fn)
    assert "A{0} = B{1};" in out


def test_vector_plain_copy_multiple_assignments():
    """Several whole-vector copies in sequence, matching bicg.sol's setup
    phase (Rr = B; Rl = Rr; Zr = Rr; ...)."""
    def fn():
        B = Array(1, 1, [9], [2], Type.Double)
        Rr = Array(1, 1, [9], [2], Type.Double)
        Rl = Array(1, 1, [9], [2], Type.Double)
        Rr = B
        Rl = Rr

    out = _sol(fn)
    assert "Rr = B;" in out
    assert "Rl = Rr;" in out


def test_simple_var_bare_reassignment_no_redecl():
    """Plain x = expr after x: Type.Int = ... emits bare assignment, no second declaration."""
    def fn():
        x: Type.Int = 5
        x = 10

    out = _sol(fn)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    decl_lines = [l for l in lines if "int x" in l]
    assert decl_lines == ["int x = 5;"]
    assert "x = 10;" in out


def test_simple_var_annotated_reassignment_no_redecl():
    """x: Type.Int = expr a second time must not re-declare x."""
    def fn():
        x: Type.Int = 1
        x: Type.Int = 2

    out = _sol(fn)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    decl_lines = [l for l in lines if "int x" in l]
    assert decl_lines == ["int x = 1;"]
    assert "x = 2;" in out


def test_simple_var_reassignment_order():
    """Declaration comes before the bare reassignment in the output."""
    def fn():
        n: Type.Int = 0
        n = 5

    out = _sol(fn)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    decl_idx   = next(i for i, l in enumerate(lines) if l == "int n = 0;")
    assign_idx = next(i for i, l in enumerate(lines) if l == "n = 5;")
    assert decl_idx < assign_idx


# ---------------------------------------------------------------------------
# Array binary-operation generation
# ---------------------------------------------------------------------------

def test_array_add_infers_c_type():
    """C = A + B: C gets same array declaration as A, expression is A + B."""
    def fn():
        A = Array(2, 1, [5, 5], [3], Type.Float)
        B = Array(2, 1, [5, 5], [3], Type.Float)
        C = A + B

    out = _sol(fn)
    assert "Array<float[3]> C[5][5] = A + B;" in out


def test_array_mul_infers_c_type():
    """C = A * B: same shape as A."""
    def fn():
        A = Array(1, 1, [10], [4], Type.Int)
        B = Array(1, 1, [10], [4], Type.Int)
        C = A * B

    out = _sol(fn)
    assert "Array<int[4]> C[10] = A * B;" in out


def test_array_scalar_mul_left():
    """C = 2 * A: scalar on left, array on right."""
    def fn():
        A = Array(1, 1, [8], [4], Type.Float)
        C = 2 * A

    out = _sol(fn)
    assert "Array<float[4]> C[8] = 2 * A;" in out


def test_array_scalar_mul_right():
    """C = A * 3: array on left, scalar on right."""
    def fn():
        A = Array(1, 1, [6], [2], Type.Double)
        C = A * 3

    out = _sol(fn)
    assert "Array<double[2]> C[6] = A * 3;" in out


def test_array_binop_c_already_declared():
    """If C is already declared as an Array, emit plain assignment (no redeclaration)."""
    def fn():
        A = Array(1, 1, [5], [3], Type.Int)
        B = Array(1, 1, [5], [3], Type.Int)
        C = Array(1, 1, [5], [3], Type.Int)
        C = A + B

    out = _sol(fn)
    lines = out.splitlines()
    decl_lines = [l for l in lines if "Array<int[3]> C" in l]
    assert len(decl_lines) == 1, "C should only be declared once"
    assert "C = A + B;" in out


def test_array_binop_annotated_uses_array_type():
    """C: SomeAnnotation = A + B is handled as array binop (annotation ignored)."""
    def fn():
        A = Array(1, 2, [4], [2, 3], Type.Double)
        B = Array(1, 2, [4], [2, 3], Type.Double)
        C: Type.Double = A + B  # annotation present but overridden by array inference

    out = _sol(fn)
    assert "Array<double[2][3]> C[4] = A + B;" in out


def test_array_add_incompatible_dims_raises():
    def fn():
        A = Array(1, 1, [5], [3], Type.Float)
        B = Array(2, 1, [5, 5], [3], Type.Float)
        C = A + B

    with pytest.raises(ArrayOperationError, match="n_dims"):
        _sol(fn)


def test_array_add_incompatible_size_raises():
    def fn():
        A = Array(1, 1, [5], [3], Type.Float)
        B = Array(1, 1, [8], [3], Type.Float)
        C = A + B

    with pytest.raises(ArrayOperationError, match="array_size"):
        _sol(fn)


def test_array_add_incompatible_block_size_raises():
    def fn():
        A = Array(1, 1, [5], [3], Type.Float)
        B = Array(1, 1, [5], [4], Type.Float)
        C = A + B

    with pytest.raises(ArrayOperationError, match="block_size"):
        _sol(fn)


def test_array_add_incompatible_type_raises():
    def fn():
        A = Array(1, 1, [5], [3], Type.Float)
        B = Array(1, 1, [5], [3], Type.Int)
        C = A + B

    with pytest.raises(ArrayOperationError, match="type"):
        _sol(fn)


def test_array_add_incompatible_message_names_arrays():
    def fn():
        A = Array(1, 1, [5], [3], Type.Float)
        B = Array(1, 1, [5], [3], Type.Int)
        C = A + B

    with pytest.raises(ArrayOperationError, match="'A'"):
        _sol(fn)


# ---------------------------------------------------------------------------
# reset(...) .sol generation
# ---------------------------------------------------------------------------

def test_reset_array():
    def fn():
        arr = Array(1, 1, [10], [4], Type.Double)
        reset(arr)

    out = _sol(fn)
    assert "arr = {};" in out


def test_reset_growing_array():
    def fn():
        arr = GrowingArray(1, [10], Type.Int)
        reset(arr)

    out = _sol(fn)
    assert "arr = {};" in out


def test_reset_csr_matrix():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)
        reset(A)

    out = _sol(fn)
    assert "A = {};" in out


def test_reset_inside_for_loop_indented():
    def fn():
        arr = Array(1, 1, [10], [4], Type.Double)
        for i in range(3):
            reset(arr)

    out = _sol(fn)
    inner = [l for l in out.splitlines() if l.strip() == "arr = {};"][0]
    assert inner.startswith("        ")  # 8 spaces = 2 levels


def test_reset_qualified_name():
    def fn():
        import pysol as _pysol
        arr = Array(1, 1, [10], [4], Type.Double)
        _pysol.reset(arr)

    out = _sol(fn)
    assert "arr = {};" in out


def test_reset_wrong_arg_count_raises():
    def fn():
        arr = Array(1, 1, [10], [4], Type.Double)
        reset(arr, arr)

    with pytest.raises(UnsupportedCallError):
        _sol(fn)


def test_reset_no_args_raises():
    def fn():
        reset()

    with pytest.raises(UnsupportedCallError):
        _sol(fn)


def test_reset_non_name_arg_raises():
    def fn():
        ia = Array(1, 1, [5], [1], Type.Int)
        ja = Array(1, 1, [6], [1], Type.Int)
        a = Array(1, 1, [6], [1], Type.Double)
        A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)
        reset(A[0][1])

    with pytest.raises(UnsupportedCallError):
        _sol(fn)


# ---------------------------------------------------------------------------
# pysol_update_array .sol generation
# ---------------------------------------------------------------------------

import pysol.sol_generator as _sol_mod


@pytest.fixture(autouse=False)
def _pat_env(tmp_path, monkeypatch):
    """Write a two-parameter pattern file and point _EXCHANGE_PATTERN at it."""
    pat = tmp_path / "update_array.pat"
    pat.write_text(
        'replace "pysol_update_array(src, src_ver, dst, dst_ver)":\n'
        "copy($src{$src_ver}, $dst{$dst_ver});\n"
    )
    monkeypatch.setattr(_sol_mod, "_EXCHANGE_PATTERN", str(pat))
    yield str(pat)


def test_update_array_happy_path(_pat_env):
    """Valid 2-D source and destination generate the pattern-expanded line."""
    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Float)
        B = Array(2, 1, [4, 4], [3], Type.Float)
        pysol_update_array(A, 0, B, 1)

    out = _sol(fn)
    assert "copy(A{0}, B{1});" in out


def test_update_array_version_expression(_pat_env):
    """Version arguments are inlined as expressions in the pattern_str."""
    def fn():
        A = Array(2, 2, [8, 8], [4, 4], Type.Double)
        B = Array(2, 2, [8, 8], [4, 4], Type.Double)
        pysol_update_array(A, 2, B, 3)

    out = _sol(fn)
    assert "copy(A{2}, B{3});" in out


def test_update_array_multiline_pattern(tmp_path, monkeypatch):
    """Pattern bodies with multiple lines are each emitted at correct indentation."""
    pat = tmp_path / "multi.pat"
    pat.write_text(
        'replace "pysol_update_array(src, src_ver, dst, dst_ver)":\n'
        "// update\n"
        "copy($src{$src_ver}, $dst{$dst_ver});\n"
    )
    monkeypatch.setattr(_sol_mod, "_EXCHANGE_PATTERN", str(pat))

    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Int)
        B = Array(2, 1, [4, 4], [3], Type.Int)
        pysol_update_array(A, 0, B, 1)

    lines = _sol(fn).splitlines()
    body = [l.strip() for l in lines if l.strip() in ("// update", "copy(A{0}, B{1});")]
    assert body == ["// update", "copy(A{0}, B{1});"]


def test_update_array_src_ndims_not_2_raises(_pat_env):
    """Source array with n_dims != 2 raises UnsupportedCallError."""
    def fn():
        A = Array(1, 1, [4], [3], Type.Float)
        B = Array(2, 1, [4, 4], [3], Type.Float)
        pysol_update_array(A, 0, B, 1)

    with pytest.raises(UnsupportedCallError, match="source"):
        _sol(fn)


def test_update_array_dst_ndims_not_2_raises(_pat_env):
    """Destination array with n_dims != 2 raises UnsupportedCallError."""
    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Float)
        B = Array(3, 1, [4, 4, 4], [3], Type.Float)
        pysol_update_array(A, 0, B, 1)

    with pytest.raises(UnsupportedCallError, match="destination"):
        _sol(fn)


def test_update_array_incompatible_size_raises(_pat_env):
    """Structurally different arrays raise ArrayOperationError."""
    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Float)
        B = Array(2, 1, [8, 8], [3], Type.Float)
        pysol_update_array(A, 0, B, 1)

    with pytest.raises(ArrayOperationError, match="array_size"):
        _sol(fn)


def test_update_array_incompatible_type_raises(_pat_env):
    """Arrays with different element types raise ArrayOperationError."""
    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Float)
        B = Array(2, 1, [4, 4], [3], Type.Int)
        pysol_update_array(A, 0, B, 1)

    with pytest.raises(ArrayOperationError, match="type"):
        _sol(fn)


def test_update_array_incompatible_block_size_raises(_pat_env):
    """Arrays with different block_size raise ArrayOperationError."""
    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Float)
        B = Array(2, 1, [4, 4], [5], Type.Float)
        pysol_update_array(A, 0, B, 1)

    with pytest.raises(ArrayOperationError, match="block_size"):
        _sol(fn)



def test_update_array_wrong_arg_count_raises(_pat_env):
    """Wrong number of arguments raises UnsupportedCallError."""
    # We test this by directly calling _update_array_stmt via sol generation.
    # Build a synthetic AST call with only 2 args.
    import ast
    from pysol.sol_generator import _SolWriter
    from pysol.decorator import _ArrayInfo

    src_info = _ArrayInfo(line=1, var="A", n_dims="2", n_block_dims="1",
                          array_size="[4, 4]", block_size="[3]", elem_type="Type.Float")
    arrays = {"A": src_info}
    writer = _SolWriter("fn", ast.parse("pass").body[0],  # type: ignore[arg-type]
                        {}, arrays)

    call = ast.parse("f(A, 0)").body[0].value  # type: ignore[attr-defined]
    with pytest.raises(UnsupportedCallError, match="4 positional"):
        writer._update_array_stmt(call, "    ")


def test_update_array_unknown_src_raises(_pat_env):
    """First argument not a known array raises UnsupportedCallError."""
    import ast
    from pysol.sol_generator import _SolWriter
    from pysol.decorator import _ArrayInfo

    dst_info = _ArrayInfo(line=1, var="B", n_dims="2", n_block_dims="1",
                          array_size="[4, 4]", block_size="[3]", elem_type="Type.Float")
    arrays = {"B": dst_info}
    writer = _SolWriter("fn", ast.parse("pass").body[0],  # type: ignore[arg-type]
                        {}, arrays)

    call = ast.parse("f(X, 0, B, 1)").body[0].value  # type: ignore[attr-defined]
    with pytest.raises(UnsupportedCallError, match="source"):
        writer._update_array_stmt(call, "    ")


# ---------------------------------------------------------------------------
# pysol_update_array — $$declare: hoisting into the sub
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def _pat_env_decl(tmp_path, monkeypatch):
    """Pattern that declares a temp array and uses it in the body."""
    pat = tmp_path / "decl.pat"
    pat.write_text(
        'replace "pysol_update_array(src, src_ver, dst, dst_ver)":\n'
        "$$declare: Array<float[3]> tmp[4, 4];\n"
        "load(tmp, $src{$src_ver});\n"
        "store($dst{$dst_ver}, tmp);\n"
    )
    monkeypatch.setattr(_sol_mod, "_EXCHANGE_PATTERN", str(pat))
    yield str(pat)


def test_declare_hoisted_after_last_array_decl(_pat_env_decl):
    """$$declare: insertion is after the last referenced array declaration, before body stmts."""
    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Float)
        B = Array(2, 1, [4, 4], [3], Type.Float)
        pysol_update_array(A, 0, B, 1)

    out = _sol(fn)
    stripped = [l.strip() for l in out.splitlines() if l.strip()]
    b_decl_idx  = next(i for i, l in enumerate(stripped) if l == "Array<float[3]> B[4][4];")
    tmp_decl_idx = next(i for i, l in enumerate(stripped) if l == "Array<float[3]> tmp_1[4, 4];")
    load_idx    = next(i for i, l in enumerate(stripped) if l.startswith("load("))
    assert b_decl_idx < tmp_decl_idx < load_idx


def test_declare_body_uses_renamed_tmp(_pat_env_decl):
    """Body lines reference the renamed (suffixed) declaration name."""
    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Float)
        B = Array(2, 1, [4, 4], [3], Type.Float)
        pysol_update_array(A, 0, B, 1)

    out = _sol(fn)
    assert "load(tmp_1, A{0});" in out
    assert "store(B{1}, tmp_1);" in out
    assert "tmp;" not in out  # bare un-renamed name must not appear


def test_declare_call_number_increments(tmp_path, monkeypatch):
    """Second pysol_update_array call gets suffix _2."""
    pat = tmp_path / "two.pat"
    pat.write_text(
        'replace "pysol_update_array(src, src_ver, dst, dst_ver)":\n'
        "$$declare: int buf;\n"
        "buf = 0;\n"
    )
    monkeypatch.setattr(_sol_mod, "_EXCHANGE_PATTERN", str(pat))

    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Int)
        B = Array(2, 1, [4, 4], [3], Type.Int)
        C = Array(2, 1, [4, 4], [3], Type.Int)
        pysol_update_array(A, 0, B, 1)
        pysol_update_array(B, 1, C, 2)

    out = _sol(fn)
    assert "int buf_1;" in out
    assert "int buf_2;" in out
    assert "buf_1 = 0;" in out
    assert "buf_2 = 0;" in out


def test_declare_hoisted_before_first_body_stmt(_pat_env_decl):
    """Hoisted declarations come before regular body statements in the sub."""
    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Float)
        B = Array(2, 1, [4, 4], [3], Type.Float)
        pysol_update_array(A, 0, B, 1)

    out = _sol(fn)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    decl_idx  = next(i for i, l in enumerate(lines) if "tmp_1" in l and "Array" in l)
    store_idx = next(i for i, l in enumerate(lines) if l.startswith("store("))
    assert decl_idx < store_idx


# ---------------------------------------------------------------------------
# pysol_update_array — $$type() resolved from array info
# ---------------------------------------------------------------------------

def test_type_token_resolves_to_sol_type(tmp_path, monkeypatch):
    """$$type($src) in the pattern is replaced with the .sol type of the source array."""
    pat = tmp_path / "typed.pat"
    pat.write_text(
        'replace "pysol_update_array(src, src_ver, dst, dst_ver)":\n'
        "$$declare: Array<$$type($src)[3]> tmp[4, 4];\n"
        "copy_typed($$type($src), $src{$src_ver}, tmp_1, $dst{$dst_ver});\n"
    )
    monkeypatch.setattr(_sol_mod, "_EXCHANGE_PATTERN", str(pat))

    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Float)
        B = Array(2, 1, [4, 4], [3], Type.Float)
        pysol_update_array(A, 0, B, 1)

    out = _sol(fn)
    assert "Array<float[3]> tmp_1[4, 4];" in out
    assert "copy_typed(float, A{0}, tmp_1, B{1});" in out


def test_type_token_int_type(tmp_path, monkeypatch):
    """$$type($src) resolves to 'int' for Type.Int arrays."""
    pat = tmp_path / "t.pat"
    pat.write_text(
        'replace "pysol_update_array(src, src_ver, dst, dst_ver)":\n'
        "$$type($src) val;\n"
    )
    monkeypatch.setattr(_sol_mod, "_EXCHANGE_PATTERN", str(pat))

    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Int)
        B = Array(2, 1, [4, 4], [3], Type.Int)
        pysol_update_array(A, 0, B, 1)
    assert "int val;" in _sol(fn)


def test_type_token_double_type(tmp_path, monkeypatch):
    """$$type($src) resolves to 'double' for Type.Double arrays."""
    pat = tmp_path / "t.pat"
    pat.write_text(
        'replace "pysol_update_array(src, src_ver, dst, dst_ver)":\n'
        "$$type($src) val;\n"
    )
    monkeypatch.setattr(_sol_mod, "_EXCHANGE_PATTERN", str(pat))

    def fn():
        A = Array(2, 1, [4, 4], [3], Type.Double)
        B = Array(2, 1, [4, 4], [3], Type.Double)
        pysol_update_array(A, 0, B, 1)
    assert "double val;" in _sol(fn)


# ---------------------------------------------------------------------------
# Array reduction operators: sum(A) / max(A) / min(A) → reduce(%<op>, A[:]...)
# ---------------------------------------------------------------------------

def test_reduce_sum_1d():
    def fn():
        A = Array(1, 1, [10], [12], Type.Double)
        s: Type.Double = sum(A)

    out = _sol(fn)
    assert "double s = reduce(%sum, A[:]);" in out


def test_reduce_sum_2d():
    def fn():
        A = Array(2, 2, [10, 10], [12, 12], Type.Double)
        s: Type.Double = sum(A)

    out = _sol(fn)
    assert "double s = reduce(%sum, A[:][:]);" in out


def test_reduce_sum_3d():
    def fn():
        A = Array(3, 3, [10, 10, 10], [12, 12, 12], Type.Double)
        s: Type.Double = sum(A)

    out = _sol(fn)
    assert "double s = reduce(%sum, A[:][:][:]);" in out


def test_reduce_max_1d():
    def fn():
        A = Array(1, 1, [10], [12], Type.Float)
        m: Type.Float = max(A)

    out = _sol(fn)
    assert "float m = reduce(%max, A[:]);" in out


def test_reduce_max_2d():
    def fn():
        A = Array(2, 2, [10, 10], [12, 12], Type.Float)
        m: Type.Float = max(A)

    out = _sol(fn)
    assert "float m = reduce(%max, A[:][:]);" in out


def test_reduce_max_3d():
    def fn():
        A = Array(3, 3, [10, 10, 10], [12, 12, 12], Type.Float)
        m: Type.Float = max(A)

    out = _sol(fn)
    assert "float m = reduce(%max, A[:][:][:]);" in out


def test_reduce_min_1d():
    def fn():
        A = Array(1, 1, [10], [12], Type.Int)
        m: Type.Int = min(A)

    out = _sol(fn)
    assert "int m = reduce(%min, A[:]);" in out


def test_reduce_min_2d():
    def fn():
        A = Array(2, 2, [10, 10], [12, 12], Type.Int)
        m: Type.Int = min(A)

    out = _sol(fn)
    assert "int m = reduce(%min, A[:][:]);" in out


def test_reduce_min_3d():
    def fn():
        A = Array(3, 3, [10, 10, 10], [12, 12, 12], Type.Int)
        m: Type.Int = min(A)

    out = _sol(fn)
    assert "int m = reduce(%min, A[:][:][:]);" in out


def test_reduce_qualified_name():
    """pysol.sum(A) (Attribute call) is recognised the same as sum(A)."""
    def fn():
        A = Array(2, 2, [10, 10], [12, 12], Type.Double)
        s: Type.Double = pysol.sum(A)

    out = _sol(fn)
    assert "double s = reduce(%sum, A[:][:]);" in out


def test_reduce_nested_in_expression():
    """reduce() calls compose with the rest of expression generation."""
    def fn():
        A = Array(1, 1, [10], [12], Type.Double)
        s: Type.Double = sum(A) + 1.0

    out = _sol(fn)
    assert "double s = reduce(%sum, A[:]) + 1.0;" in out


def test_reduce_unknown_array_raises():
    """sum() on a non-array variable raises UnsupportedCallError."""
    def fn():
        x: Type.Int = 5
        s: Type.Int = sum(x)

    with pytest.raises(UnsupportedCallError, match="sum"):
        _sol(fn)


def test_reduce_wrong_arg_count_raises():
    """sum() called with two arguments raises UnsupportedCallError."""
    def fn():
        A = Array(1, 1, [10], [12], Type.Double)
        B = Array(1, 1, [10], [12], Type.Double)
        s: Type.Double = sum(A, B)

    with pytest.raises(UnsupportedCallError, match="sum"):
        _sol(fn)


def test_reduce_no_args_raises():
    """sum() called with no arguments raises UnsupportedCallError."""
    def fn():
        s: Type.Int = sum()

    with pytest.raises(UnsupportedCallError, match="sum"):
        _sol(fn)


def test_reduce_unsupported_ndims_raises():
    """An array with an n_dims outside {1, 2, 3} raises UnsupportedCallError."""
    from pysol.sol_generator import _SolWriter
    from pysol.decorator import _ArrayInfo

    info = _ArrayInfo(line=1, var="A", n_dims="4", n_block_dims="1",
                       array_size="[4, 4, 4, 4]", block_size="[3]", elem_type="Type.Float")
    writer = _SolWriter("fn", ast.parse("pass").body[0],  # type: ignore[arg-type]
                         {}, {"A": info})

    call = ast.parse("sum(A)").body[0].value  # type: ignore[attr-defined]
    with pytest.raises(UnsupportedCallError, match="n_dims"):
        writer._reduce_expr("sum", call)


# ---------------------------------------------------------------------------
# GrowingArray declaration generation
# ---------------------------------------------------------------------------

def test_growing_array_decl_3d_example():
    """The spec example: GrowingArray(3, [nb, mb, lb], Type.Double) → GrowingArray<double> A[nb][mb][lb];"""
    def fn():
        nb: Type.Int = 10
        mb: Type.Int = 10
        lb: Type.Int = 10
        A = GrowingArray(3, [nb, mb, lb], Type.Double)

    out = _sol(fn)
    assert "GrowingArray<double> A[nb][mb][lb];" in out


def test_growing_array_decl_1d():
    def fn():
        G = GrowingArray(1, [10], Type.Int)

    out = _sol(fn)
    assert "GrowingArray<int> G[10];" in out


def test_growing_array_decl_2d():
    def fn():
        G = GrowingArray(2, [10, 20], Type.Float)

    out = _sol(fn)
    assert "GrowingArray<float> G[10][20];" in out


def test_growing_array_decl_literal_dims():
    def fn():
        G = GrowingArray(3, [4, 5, 6], Type.Double)

    out = _sol(fn)
    assert "GrowingArray<double> G[4][5][6];" in out


def test_growing_array_decl_ann_assign():
    """AnnAssign form: G: SomeAnnotation = GrowingArray(...) is handled the same as Assign."""
    def fn():
        G: Type.Double = GrowingArray(2, [10, 10], Type.Double)

    out = _sol(fn)
    assert "GrowingArray<double> G[10][10];" in out


def test_growing_array_no_block_dims_in_decl():
    """Unlike Array, the declaration never includes block-size brackets on the type."""
    def fn():
        G = GrowingArray(1, [10], Type.Double)

    out = _sol(fn)
    assert "GrowingArray<double[" not in out


def test_growing_array_alongside_regular_array():
    """GrowingArray and Array declarations coexist and keep their own forms."""
    def fn():
        A = Array(1, 1, [10], [12], Type.Double)
        G = GrowingArray(1, [10], Type.Double)

    out = _sol(fn)
    assert "Array<double[12]> A[10];" in out
    assert "GrowingArray<double> G[10];" in out


def test_growing_array_binop_infers_growing_c_type():
    """C = A + B where A, B are GrowingArray: C is declared as GrowingArray too."""
    def fn():
        A = GrowingArray(1, [10], Type.Float)
        B = GrowingArray(1, [10], Type.Float)
        C = A + B

    out = _sol(fn)
    assert "GrowingArray<float> C[10] = A + B;" in out


# ---------------------------------------------------------------------------
# @pysol_struct type declarations
# ---------------------------------------------------------------------------

def test_struct_decl_matches_spec_example():
    """The exact worked example: class T(a: Double, b: Float) → type T : struct { a : double, b : float };"""
    @pysol_struct
    class _T_Spec:
        a: Type.Double
        b: Type.Float

    def fn():
        G = GrowingArray(1, [10], _T_Spec)

    out = _sol(fn)
    assert "type _T_Spec : struct { a : double, b : float };" in out


def test_struct_decl_as_growing_array_element_type():
    @pysol_struct
    class _Vec3Struct:
        x: Type.Double
        y: Type.Double
        z: Type.Double

    def fn():
        nb: Type.Int = 10
        mb: Type.Int = 10
        lb: Type.Int = 10
        A = GrowingArray(3, [nb, mb, lb], _Vec3Struct)

    out = _sol(fn)
    assert "type _Vec3Struct : struct { x : double, y : double, z : double };" in out
    assert "GrowingArray<_Vec3Struct> A[nb][mb][lb];" in out


def test_struct_decl_as_regular_array_element_type():
    @pysol_struct
    class _PairStruct:
        a: Type.Int
        b: Type.Int

    def fn():
        A = Array(1, 1, [10], [12], _PairStruct)

    out = _sol(fn)
    assert "type _PairStruct : struct { a : int, b : int };" in out
    assert "Array<_PairStruct[12]> A[10];" in out


def test_struct_decl_as_variable_annotation():
    @pysol_struct
    class _ScalarStruct:
        v: Type.Double

    def fn():
        G = GrowingArray(1, [10], _ScalarStruct)
        x: _ScalarStruct = G

    out = _sol(fn)
    assert "type _ScalarStruct : struct { v : double };" in out
    assert "_ScalarStruct x = G;" in out


def test_struct_decl_appears_before_imports():
    @pysol_struct
    class _OrderStruct:
        v: Type.Int

    def fn():
        k = import_kernel("noop", [])
        G = GrowingArray(1, [10], _OrderStruct)
        k()

    out = _sol(fn)
    lines = [l for l in out.splitlines() if l.strip()]
    struct_idx = next(i for i, l in enumerate(lines) if l.startswith("type "))
    import_idx = next(i for i, l in enumerate(lines) if l.startswith("import "))
    assert struct_idx < import_idx


def test_struct_decl_emitted_once_when_used_twice():
    """A struct referenced by two arrays only gets one type declaration."""
    @pysol_struct
    class _SharedStruct:
        v: Type.Double

    def fn():
        A = GrowingArray(1, [10], _SharedStruct)
        B = GrowingArray(1, [20], _SharedStruct)

    out = _sol(fn)
    assert out.count("type _SharedStruct : struct") == 1


def test_struct_decl_not_emitted_when_unused():
    """A registered struct that isn't referenced in the function produces no decl."""
    @pysol_struct
    class _UnusedStruct:
        v: Type.Int

    def fn():
        x: Type.Int = 5

    out = _sol(fn)
    assert "_UnusedStruct" not in out
