"""Tests for .sol file generation."""
import os
import textwrap

import pytest

import pysol as _pysol_mod
from pysol import (
    Array, ArrayOperationError, KernelArgument, NotImportedKernel,
    UnsupportedCallError, UnsupportedVariableType, Type, import_kernel, pysol,
    pysol_update_array,
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
    """arr.get(v, [i, j]) → arr{v}[i][j]  (no block-size prefix)"""
    def fn():
        arr = Array(2, 1, [5, 5], [3], Type.Int)
        k = import_kernel("kern", [])
        k(arr.get(v, [i, j]))

    out = _sol(fn)
    assert "kern(arr{v}[i][j]);" in out


def test_array_get_scalar_and_array_args():
    """kernel(1, var, arr.get(version, [e1, e2])) → kernel(1, var, arr{version}[e1][e2])"""
    def fn():
        arr = Array(2, 1, [5, 5], [3], Type.Double)
        k = import_kernel("kernel", [])
        k(1, var, arr.get(version, [e1, e2]))

    out = _sol(fn)
    assert "kernel(1, var, arr{version}[e1][e2]);" in out


def test_array_get_single_coord():
    """arr.get(v, [i]) → arr{v}[i]"""
    def fn():
        arr = Array(1, 1, [5], [3], Type.Int)
        k = import_kernel("kern", [])
        k(arr.get(v, [i]))

    out = _sol(fn)
    assert "kern(arr{v}[i]);" in out


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
