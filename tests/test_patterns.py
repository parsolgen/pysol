"""Tests for _pysol_apply_pattern."""
import pytest
from pysol.decorator import _pysol_apply_pattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path, content: str, name: str = "pat.txt"):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def _apply(pattern_str: str, path: str, call_number: int = 1,
           array_types=None) -> str:
    """Return just the body portion of _pysol_apply_pattern (ignores hoists and imports)."""
    body, _, _ = _pysol_apply_pattern(pattern_str, path, call_number, array_types)
    return body


# ---------------------------------------------------------------------------
# Happy path — single parameter
# ---------------------------------------------------------------------------

def test_single_param_substituted(tmp_path):
    path = _write(tmp_path, (
        'replace "echo(x)":\n'
        'print($x);\n'
    ))
    result = _apply("echo(hello)", path)
    assert result == "print(hello);"


def test_single_param_multiple_lines(tmp_path):
    path = _write(tmp_path, (
        'replace "greet(name)":\n'
        'string s = $name;\n'
        'write(s);\n'
    ))
    result = _apply("greet(world)", path)
    assert result == "string s = world;\nwrite(s);"


def test_single_param_multiple_occurrences_on_one_line(tmp_path):
    path = _write(tmp_path, (
        'replace "dup(v)":\n'
        '$v = $v;\n'
    ))
    result = _apply("dup(x)", path)
    assert result == "x = x;"


# ---------------------------------------------------------------------------
# Happy path — multiple parameters
# ---------------------------------------------------------------------------

def test_two_params_substituted(tmp_path):
    path = _write(tmp_path, (
        'replace "add(a, b)":\n'
        'result = $a + $b;\n'
    ))
    result = _apply("add(x_val, y_val)", path)
    assert result == "result = x_val + y_val;"


def test_three_params(tmp_path):
    path = _write(tmp_path, (
        'replace "fma(a, b, c)":\n'
        'out = $a * $b + $c;\n'
    ))
    result = _apply("fma(alpha, src, bias)", path)
    assert result == "out = alpha * src + bias;"


def test_params_used_independently_per_line(tmp_path):
    path = _write(tmp_path, (
        'replace "copy(src, dst)":\n'
        'load $src;\n'
        'store $dst;\n'
    ))
    result = _apply("copy(A, B)", path)
    assert result == "load A;\nstore B;"


# ---------------------------------------------------------------------------
# Lines without parameter tokens are kept unchanged
# ---------------------------------------------------------------------------

def test_plain_line_unchanged(tmp_path):
    path = _write(tmp_path, (
        'replace "nop(x)":\n'
        'int result = 0;\n'
        'result = $x;\n'
        'return result;\n'
    ))
    result = _apply("nop(val)", path)
    lines = result.splitlines()
    assert lines[0] == "int result = 0;"
    assert lines[1] == "result = val;"
    assert lines[2] == "return result;"


def test_unknown_dollar_token_kept_as_is(tmp_path):
    """$unknown (not in pattern params) must pass through unchanged."""
    path = _write(tmp_path, (
        'replace "set(x)":\n'
        '$x = $unknown;\n'
    ))
    result = _apply("set(v)", path)
    assert result == "v = $unknown;"


# ---------------------------------------------------------------------------
# Empty body
# ---------------------------------------------------------------------------

def test_empty_body_returns_empty_string(tmp_path):
    path = _write(tmp_path, 'replace "noop(x)":\n')
    result = _apply("noop(v)", path)
    assert result == ""


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_empty_file_raises(tmp_path):
    path = _write(tmp_path, "")
    with pytest.raises(ValueError, match="empty"):
        _pysol_apply_pattern("foo(x)", path)


def test_invalid_header_raises(tmp_path):
    path = _write(tmp_path, "this is not a valid header\nsome body\n")
    with pytest.raises(ValueError, match="header"):
        _pysol_apply_pattern("foo(x)", path)


def test_arity_mismatch_raises(tmp_path):
    path = _write(tmp_path, 'replace "add(a, b)":\nresult = $a + $b;\n')
    with pytest.raises(ValueError, match="2"):
        _pysol_apply_pattern("add(only_one_arg)", path)


def test_invalid_pattern_str_raises(tmp_path):
    path = _write(tmp_path, 'replace "f(x)":\n$x;\n')
    with pytest.raises(ValueError, match="pattern string"):
        _pysol_apply_pattern("not_a_call", path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _pysol_apply_pattern("f(x)", str(tmp_path / "nonexistent.txt"))


# ---------------------------------------------------------------------------
# Prefix-collision safety
# ---------------------------------------------------------------------------

def test_prefix_param_not_clobbered(tmp_path):
    """$src must not corrupt $src_ver when both appear in the body."""
    path = _write(tmp_path, (
        'replace "ua(src, src_ver, dst, dst_ver)":\n'
        'copy($src{$src_ver}, $dst{$dst_ver});\n'
    ))
    result = _apply("ua(A, 0, B, 1)", path)
    assert result == "copy(A{0}, B{1});"


# ---------------------------------------------------------------------------
# $$declare: hoisting
# ---------------------------------------------------------------------------

def test_declare_array_is_hoisted(tmp_path):
    """$$declare: line is removed from body and returned as hoist."""
    path = _write(tmp_path, (
        'replace "f(a)":\n'
        '$$declare: Array<float[4]> tmp[8, 8];\n'
        'use(tmp);\n'
    ))
    body, hoist, _ = _pysol_apply_pattern("f(x)", path, call_number=1)
    assert hoist == ["Array<float[4]> tmp_1[8, 8];"]
    assert body == "use(tmp_1);"


def test_declare_variable_is_hoisted(tmp_path):
    """$$declare: with a primitive variable declaration."""
    path = _write(tmp_path, (
        'replace "f(a)":\n'
        '$$declare: int counter;\n'
        'counter = $a;\n'
    ))
    body, hoist, _ = _pysol_apply_pattern("f(0)", path, call_number=3)
    assert hoist == ["int counter_3;"]
    assert body == "counter_3 = 0;"


def test_declare_call_number_in_suffix(tmp_path):
    """The suffix uses the exact call_number passed in."""
    path = _write(tmp_path, (
        'replace "g(v)":\n'
        '$$declare: float acc;\n'
        'acc = $v;\n'
    ))
    _, hoist2, _ = _pysol_apply_pattern("g(x)", path, call_number=2)
    _, hoist7, _ = _pysol_apply_pattern("g(x)", path, call_number=7)
    assert hoist2 == ["float acc_2;"]
    assert hoist7 == ["float acc_7;"]


def test_declare_body_references_renamed(tmp_path):
    """All occurrences of the declared name in body lines are renamed."""
    path = _write(tmp_path, (
        'replace "f(a)":\n'
        '$$declare: int tmp;\n'
        'tmp = $a;\n'
        'result = tmp + 1;\n'
    ))
    body, _, _ = _pysol_apply_pattern("f(5)", path, call_number=2)
    assert body == "tmp_2 = 5;\nresult = tmp_2 + 1;"


def test_declare_param_substituted_in_declaration(tmp_path):
    """$param tokens inside $$declare: are substituted before name extraction."""
    path = _write(tmp_path, (
        'replace "f(src)":\n'
        '$$declare: int buf;\n'
        'buf = $src;\n'
    ))
    body, hoist, _ = _pysol_apply_pattern("f(A)", path, call_number=1)
    assert hoist == ["int buf_1;"]
    assert body == "buf_1 = A;"


def test_declare_multiple_declarations(tmp_path):
    """Multiple $$declare: lines each become a separate hoist entry."""
    path = _write(tmp_path, (
        'replace "f(a)":\n'
        '$$declare: int x;\n'
        '$$declare: float y;\n'
        'x = $a;\n'
        'y = x;\n'
    ))
    body, hoist, _ = _pysol_apply_pattern("f(1)", path, call_number=4)
    assert hoist == ["int x_4;", "float y_4;"]
    assert body == "x_4 = 1;\ny_4 = x_4;"


def test_declare_no_declarations_empty_hoist(tmp_path):
    """Pattern with no $$declare: lines returns an empty hoist list."""
    path = _write(tmp_path, (
        'replace "f(a)":\n'
        'use($a);\n'
    ))
    body, hoist, _ = _pysol_apply_pattern("f(v)", path, call_number=1)
    assert hoist == []
    assert body == "use(v);"


def test_declare_unknown_name_raises(tmp_path):
    """$$declare: line whose name cannot be extracted raises ValueError."""
    path = _write(tmp_path, (
        'replace "f(a)":\n'
        '$$declare: ???;\n'
        'use($a);\n'
    ))
    with pytest.raises(ValueError, match="Cannot extract name"):
        _pysol_apply_pattern("f(x)", path, call_number=1)


# ---------------------------------------------------------------------------
# $$type() substitution
# ---------------------------------------------------------------------------

def test_type_token_in_body_line(tmp_path):
    """$$type(A) in a normal body line is replaced with the array's .sol type."""
    path = _write(tmp_path, (
        'replace "f(src)":\n'
        '$$type($src) tmp = 0;\n'
    ))
    result = _apply("f(A)", path, array_types={"A": "float"})
    assert result == "float tmp = 0;"


def test_type_token_via_direct_name(tmp_path):
    """$$type(name) where name is a literal (not a $param) is resolved."""
    path = _write(tmp_path, (
        'replace "f(x)":\n'
        'cast($$type(myarr), $x);\n'
    ))
    result = _apply("f(v)", path, array_types={"myarr": "double"})
    assert result == "cast(double, v);"


def test_type_token_via_param_substitution(tmp_path):
    """$$type($src) resolves $src first, then looks up the resulting name."""
    path = _write(tmp_path, (
        'replace "f(src, dst)":\n'
        '$$type($src) x; $$type($dst) y;\n'
    ))
    result = _apply("f(A, B)", path, array_types={"A": "int", "B": "double"})
    assert result == "int x; double y;"


def test_type_token_in_declare_line(tmp_path):
    """$$type inside $$declare: is resolved before name extraction."""
    path = _write(tmp_path, (
        'replace "f(a)":\n'
        '$$declare: Array<$$type($a)[4]> tmp[8, 8];\n'
        'use(tmp);\n'
    ))
    body, hoist, _ = _pysol_apply_pattern("f(A)", path, 1, {"A": "float"})
    assert hoist == ["Array<float[4]> tmp_1[8, 8];"]
    assert body == "use(tmp_1);"


def test_type_token_unknown_left_unchanged(tmp_path):
    """$$type(X) where X is not in array_types is emitted unchanged."""
    path = _write(tmp_path, (
        'replace "f(a)":\n'
        'use($$type(unknown), $a);\n'
    ))
    result = _apply("f(v)", path, array_types={"other": "int"})
    assert result == "use($$type(unknown), v);"


def test_type_token_no_array_types_left_unchanged(tmp_path):
    """$$type(A) with no array_types dict is left unchanged."""
    path = _write(tmp_path, (
        'replace "f(a)":\n'
        'use($$type(A), $a);\n'
    ))
    result = _apply("f(v)", path)  # array_types=None
    assert result == "use($$type(A), v);"


def test_type_token_multiple_occurrences(tmp_path):
    """Multiple $$type() tokens on the same line are all replaced."""
    path = _write(tmp_path, (
        'replace "f(src, dst)":\n'
        'conv($$type($src), $$type($dst));\n'
    ))
    result = _apply("f(A, B)", path, array_types={"A": "float", "B": "int"})
    assert result == "conv(float, int);"


# ---------------------------------------------------------------------------
# $$import directive
# ---------------------------------------------------------------------------

def test_import_collected_not_in_body(tmp_path):
    """$$import line is removed from body and returned in imports list."""
    path = _write(tmp_path, (
        'replace "f(x)":\n'
        '$$import helper(in int n);\n'
        'helper($x);\n'
    ))
    body, _, imports = _pysol_apply_pattern("f(v)", path)
    assert body == "helper(v);"
    assert imports == ["import helper(in int n);"]


def test_import_param_substituted(tmp_path):
    """$param tokens inside $$import are substituted."""
    path = _write(tmp_path, (
        'replace "f(src)":\n'
        '$$import read_$src(in int n);\n'
        'read_$src(0);\n'
    ))
    body, _, imports = _pysol_apply_pattern("f(A)", path)
    assert imports == ["import read_A(in int n);"]
    assert body == "read_A(0);"


def test_import_type_substituted(tmp_path):
    """$$type() inside $$import is resolved."""
    path = _write(tmp_path, (
        'replace "f(arr)":\n'
        '$$import load(in Array<$$type($arr)> x[n]);\n'
        'load(data);\n'
    ))
    body, _, imports = _pysol_apply_pattern("f(A)", path, array_types={"A": "float"})
    assert imports == ["import load(in Array<float> x[n]);"]
    assert body == "load(data);"


def test_import_multiple(tmp_path):
    """Multiple $$import lines each produce a separate import entry."""
    path = _write(tmp_path, (
        'replace "f(x)":\n'
        '$$import alpha(in int n);\n'
        '$$import beta(in int n);\n'
        'do_work($x);\n'
    ))
    _, _, imports = _pysol_apply_pattern("f(v)", path)
    assert imports == ["import alpha(in int n);", "import beta(in int n);"]


def test_import_no_imports_empty_list(tmp_path):
    """Pattern with no $$import lines returns an empty imports list."""
    path = _write(tmp_path, (
        'replace "f(a)":\n'
        'use($a);\n'
    ))
    _, _, imports = _pysol_apply_pattern("f(v)", path)
    assert imports == []
