from __future__ import annotations

import ast
import ctypes
import _ctypes
import dataclasses
import functools
import inspect
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from pysol.array import Array

from pysol.config import get_config
from pysol.types import Type


# ── Exceptions ──────────────────────────────────────────────────────────────

class NotSupportedLoop(Exception):
    """Raised at decoration time when a for-loop is not of the form 'for <var> in range(...)'."""


class NotSupportedIfStatement(Exception):
    """Raised at decoration time when an if-condition is not of the form 'expr(v1, ..., vn)'
    where every argument is a plain variable."""


class NotImportedKernel(Exception):
    """Raised during .sol generation when a call targets a function not imported via import_kernel."""


class SolLibraryNotFoundError(Exception):
    """Raised by pysol_run when the .so shared library cannot be found or loaded."""


class SolFunctionNotFoundError(Exception):
    """Raised by pysol_run when the function symbol is absent from the .so shared library."""


class SolInitError(Exception):
    """Raised by pysol_run when init_pslib returns a non-zero error code."""


class SolCompileError(Exception):
    """Raised by pysol_compile when the C++ compiler returns a non-zero exit code."""


class UnsupportedVariableType(Exception):
    """Raised during .sol generation when a variable lacks a recognised pysol type annotation."""


class ArrayOperationError(Exception):
    """Raised during .sol generation when two array operands are incompatible."""


class UnsupportedCallError(Exception):
    """Raised during .sol generation when a call is not supported (e.g. wrong array dimensionality)."""


class UnsupportedStructFieldType(Exception):
    """Raised by @pysol_struct when a field's annotation isn't Type.Int/Float/Double."""


# Regex matching $$type(array_name) tokens in pattern bodies
_DOLLAR_TYPE_RE = re.compile(r'\$\$type\((\w+)\)')

# ast.Index was removed in Python 3.9; capture it once to avoid deprecation hints
_AST_INDEX = getattr(ast, "Index", None)


# ── Minimal AST → string (Python 3.8 compatible) ────────────────────────────

def _unparse(node: ast.AST) -> str:
    if node is None:
        return "None"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Attribute):
        return f"{_unparse(node.value)}.{node.attr}"
    if isinstance(node, ast.List):
        return "[" + ", ".join(_unparse(e) for e in node.elts) + "]"
    if isinstance(node, ast.Tuple):
        return "(" + ", ".join(_unparse(e) for e in node.elts) + ")"
    if _AST_INDEX is not None and isinstance(node, _AST_INDEX):
        return _unparse(node.value)  # type: ignore[arg-type]
    if isinstance(node, ast.Call):
        args = [_unparse(a) for a in node.args]
        kwargs = [f"{kw.arg}={_unparse(kw.value)}" for kw in node.keywords]
        return f"{_unparse(node.func)}({', '.join(args + kwargs)})"
    if isinstance(node, ast.Subscript):
        return f"{_unparse(node.value)}[{_unparse(node.slice)}]"
    if isinstance(node, ast.UnaryOp):
        ops = {ast.USub: "-", ast.UAdd: "+", ast.Not: "not ", ast.Invert: "~"}
        return f"{ops.get(type(node.op), '?')}{_unparse(node.operand)}"
    if isinstance(node, ast.BinOp):
        ops = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
               ast.Mod: "%", ast.FloorDiv: "//", ast.Pow: "**"}
        return f"{_unparse(node.left)} {ops.get(type(node.op), '?')} {_unparse(node.right)}"
    if isinstance(node, ast.Compare):
        cmpops = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
                  ast.Eq: "==", ast.NotEq: "!=", ast.In: "in", ast.NotIn: "not in",
                  ast.Is: "is", ast.IsNot: "is not"}
        parts = [_unparse(node.left)]
        for op, comp in zip(node.ops, node.comparators):
            parts += [cmpops.get(type(op), "?"), _unparse(comp)]
        return " ".join(parts)
    if isinstance(node, ast.BoolOp):
        op_str = " and " if isinstance(node.op, ast.And) else " or "
        return op_str.join(_unparse(v) for v in node.values)
    if isinstance(node, ast.IfExp):
        return f"{_unparse(node.body)} if {_unparse(node.test)} else {_unparse(node.orelse)}"
    return f"<{type(node).__name__}>"


def _target_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Tuple):
        return ", ".join(_target_name(e) for e in node.elts)
    return _unparse(node)


def _is_range_for(node: ast.For) -> bool:
    """True iff the loop is exactly: for <single_var> in range(...)"""
    return (
        isinstance(node.target, ast.Name)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
    )


def _is_valid_if(node: ast.If) -> bool:
    """
    Valid conditions:
      1. expr(v1, ..., vn)  — call whose every argument is a plain variable
      2. var op var         — comparison or boolean op between two variables
      3. var op const       — comparison or boolean op, variable on the left
      4. const op var       — comparison or boolean op, constant on the left
      5. not expr(v1, ..., vn) — negation of form 1 (e.g. "not is_zero(b)")
    At least one side of a binary expression must be a variable.

    is_zero(...) is special-cased within form 1/5: its argument may also be
    a CSR matrix block subscript (is_zero(A[i][j])), not just a plain Name.
    """

    def _call_args_ok(call: ast.Call) -> bool:
        if _is_zero_call(call):
            return (
                len(call.args) == 1
                and isinstance(call.args[0], (ast.Name, ast.Subscript))
                and not call.keywords
            )
        return all(isinstance(a, ast.Name) for a in call.args) and all(
            isinstance(kw.value, ast.Name) for kw in call.keywords
        )

    test = node.test

    # Form 1: all-variable call (or is_zero(block-subscript))
    if isinstance(test, ast.Call):
        return _call_args_ok(test)

    def _is_atom(n: ast.AST) -> bool:
        return isinstance(n, (ast.Name, ast.Constant))

    def _valid_binary(left: ast.AST, right: ast.AST) -> bool:
        return (
            _is_atom(left)
            and _is_atom(right)
            and (isinstance(left, ast.Name) or isinstance(right, ast.Name))
        )

    # Forms 2–4: single comparison operator  (<, >, <=, >=, ==, !=)
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        return _valid_binary(test.left, test.comparators[0])

    # Forms 2–4: boolean operator with exactly two operands  (and, or)
    if isinstance(test, ast.BoolOp) and len(test.values) == 2:
        return _valid_binary(test.values[0], test.values[1])

    # Form 5: not expr(v1, ..., vn)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and isinstance(
        test.operand, ast.Call
    ):
        return _call_args_ok(test.operand)

    return False


def _is_array_call(node: ast.AST) -> bool:
    """True for either Array(...) or GrowingArray(...) constructor calls."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.id if isinstance(func, ast.Name) else (
        func.attr if isinstance(func, ast.Attribute) else None
    )
    return name in ("Array", "GrowingArray")


def _is_growing_array_call(node: ast.AST) -> bool:
    """True specifically for GrowingArray(...) constructor calls."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "GrowingArray") or (
        isinstance(func, ast.Attribute) and func.attr == "GrowingArray"
    )


def _is_csr_matrix_call(node: ast.AST) -> bool:
    """True for CSRMatrix(...) constructor calls."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "CSRMatrix") or (
        isinstance(func, ast.Attribute) and func.attr == "CSRMatrix"
    )


def _is_csr_block_annotation(node: ast.AST) -> bool:
    """True for a Type.CSRBlock(...) annotation/type expression."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "CSRBlock"
    )


def _is_zero_call(node: ast.AST) -> bool:
    """True for is_zero(...) calls, e.g. is_zero(b) or is_zero(A[i][j])."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "is_zero") or (
        isinstance(func, ast.Attribute) and func.attr == "is_zero"
    )


def _is_reset_call(node: ast.AST) -> bool:
    """True for reset(...) calls, e.g. reset(A) where A is an Array or CSRMatrix."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "reset") or (
        isinstance(func, ast.Attribute) and func.attr == "reset"
    )


# ── Info containers ──────────────────────────────────────────────────────────

# Array(n_dims, n_block_dims, array_size, block_size, type, is_single_version)
_ARRAY_PARAMS = (
    "n_dims", "n_block_dims", "array_size", "block_size", "type", "is_single_version",
)
# GrowingArray(n_dims, array_size, type) — block dimension is always 1, so it
# takes no n_block_dims / block_size arguments.
_GROWING_ARRAY_PARAMS = ("n_dims", "array_size", "type")


@dataclass
class _ArrayInfo:
    line: int
    var: str
    n_dims: str
    n_block_dims: str
    array_size: str
    block_size: str
    elem_type: str
    is_growing: bool = False
    is_single_version: bool = False


# CSRMatrix([dim1, dim2], [n_blocks_1, n_blocks_2], nnz, ia, ja, a, type)
_CSR_MATRIX_PARAMS = ("size", "n_blocks", "nnz", "ia", "ja", "a", "type")


@dataclass
class _CSRMatrixInfo:
    line: int
    var: str
    dim1: str
    dim2: str
    n_blocks_1: str
    n_blocks_2: str
    nnz: str
    ia: str
    ja: str
    a: str
    elem_type: str


@dataclass
class _ForInfo:
    line: int
    target: str
    iterable: str


@dataclass
class _WhileInfo:
    line: int
    condition: str


@dataclass
class _IfInfo:
    line: int
    condition: str  # e.g. "func(a, b)"


@dataclass
class _Analysis:
    name: str
    arrays: List[_ArrayInfo] = field(default_factory=list)
    csr_matrices: List[_CSRMatrixInfo] = field(default_factory=list)
    for_loops: List[_ForInfo] = field(default_factory=list)
    while_loops: List[_WhileInfo] = field(default_factory=list)
    if_stmts: List[_IfInfo] = field(default_factory=list)


# ── Static analysis ──────────────────────────────────────────────────────────

def _extract_array(line: int, var: str, call: ast.Call) -> _ArrayInfo:
    is_growing = _is_growing_array_call(call)
    param_names = _GROWING_ARRAY_PARAMS if is_growing else _ARRAY_PARAMS
    values = {p: "?" for p in param_names}
    if "is_single_version" in values:
        values["is_single_version"] = "False"  # optional, defaults to False
    for i, arg in enumerate(call.args):
        if i < len(param_names):
            values[param_names[i]] = _unparse(arg)
    for kw in call.keywords:
        if kw.arg in values:
            values[kw.arg] = _unparse(kw.value)
    return _ArrayInfo(
        line=line,
        var=var,
        n_dims=values["n_dims"],
        n_block_dims="0" if is_growing else values["n_block_dims"],
        array_size=values["array_size"],
        block_size="[]" if is_growing else values["block_size"],
        elem_type=values["type"],
        is_growing=is_growing,
        is_single_version=values.get("is_single_version") == "True",
    )


def _extract_csr_matrix(line: int, var: str, call: ast.Call) -> _CSRMatrixInfo:
    values = {p: "?" for p in _CSR_MATRIX_PARAMS}
    for i, arg in enumerate(call.args):
        if i < len(_CSR_MATRIX_PARAMS):
            values[_CSR_MATRIX_PARAMS[i]] = _unparse(arg)
    for kw in call.keywords:
        if kw.arg in values:
            values[kw.arg] = _unparse(kw.value)

    def _unpack_pair(node: ast.AST, label: str) -> Tuple[str, str]:
        if isinstance(node, ast.List) and len(node.elts) == 2:
            return _unparse(node.elts[0]), _unparse(node.elts[1])
        raise ValueError(
            f"CSRMatrix '{var}': {label} must be a 2-element list literal "
            f"[dim1, dim2] / [n_blocks_1, n_blocks_2], got '{_unparse(node)}'"
        )

    size_node = next((a for i, a in enumerate(call.args) if i == 0), None)
    n_blocks_node = next((a for i, a in enumerate(call.args) if i == 1), None)
    for kw in call.keywords:
        if kw.arg == "size":
            size_node = kw.value
        elif kw.arg == "n_blocks":
            n_blocks_node = kw.value

    if size_node is None or n_blocks_node is None:
        raise ValueError(
            f"CSRMatrix '{var}': expected CSRMatrix([dim1, dim2], "
            f"[n_blocks_1, n_blocks_2], nnz, ia, ja, a, type)"
        )
    dim1, dim2 = _unpack_pair(size_node, "size")
    n_blocks_1, n_blocks_2 = _unpack_pair(n_blocks_node, "n_blocks")

    return _CSRMatrixInfo(
        line=line,
        var=var,
        dim1=dim1,
        dim2=dim2,
        n_blocks_1=n_blocks_1,
        n_blocks_2=n_blocks_2,
        nnz=values["nnz"],
        ia=values["ia"],
        ja=values["ja"],
        a=values["a"],
        elem_type=values["type"],
    )


def _analyse(func: Callable) -> _Analysis:
    source_lines, start_line = inspect.getsourcelines(func)
    source = textwrap.dedent("".join(source_lines))
    tree = ast.parse(source)
    offset = start_line - 1  # convert 1-based relative → absolute file line

    analysis = _Analysis(name=func.__name__)

    for node in ast.walk(tree):
        abs_line = node.lineno + offset if hasattr(node, "lineno") else 0

        if isinstance(node, ast.Assign) and _is_array_call(node.value):
            for target in node.targets:
                analysis.arrays.append(_extract_array(abs_line, _target_name(target), node.value))

        elif isinstance(node, ast.AnnAssign) and node.value is not None and _is_array_call(node.value):
            analysis.arrays.append(_extract_array(abs_line, _target_name(node.target), node.value))

        elif isinstance(node, ast.Assign) and _is_csr_matrix_call(node.value):
            for target in node.targets:
                analysis.csr_matrices.append(
                    _extract_csr_matrix(abs_line, _target_name(target), node.value)
                )

        elif (isinstance(node, ast.AnnAssign) and node.value is not None
                and _is_csr_matrix_call(node.value)):
            analysis.csr_matrices.append(
                _extract_csr_matrix(abs_line, _target_name(node.target), node.value)
            )

        elif isinstance(node, ast.For):
            if not _is_range_for(node):
                raise NotSupportedLoop(
                    f"{func.__name__}: unsupported for-loop at line {abs_line}: "
                    f"'for {_target_name(node.target)} in {_unparse(node.iter)}' — "
                    f"only 'for <var> in range(...)' is allowed"
                )
            analysis.for_loops.append(_ForInfo(
                line=abs_line,
                target=_target_name(node.target),
                iterable=_unparse(node.iter),
            ))

        elif isinstance(node, ast.While):
            analysis.while_loops.append(_WhileInfo(
                line=abs_line,
                condition=_unparse(node.test),
            ))

        elif isinstance(node, ast.If):
            if not _is_valid_if(node):
                raise NotSupportedIfStatement(
                    f"{func.__name__}: unsupported if-statement at line {abs_line}: "
                    f"'if {_unparse(node.test)}' — "
                    f"only 'if expr(v1, ..., vn)' with variable arguments is allowed"
                )
            analysis.if_stmts.append(_IfInfo(
                line=abs_line,
                condition=_unparse(node.test),
            ))

    analysis.arrays.sort(key=lambda x: x.line)
    analysis.csr_matrices.sort(key=lambda x: x.line)
    analysis.for_loops.sort(key=lambda x: x.line)
    analysis.while_loops.sort(key=lambda x: x.line)
    analysis.if_stmts.sort(key=lambda x: x.line)
    return analysis


# ── Function registry ────────────────────────────────────────────────────────

_registry: Dict[str, Callable] = {}


# ── Struct types ──────────────────────────────────────────────────────────────

# Only these scalar types may appear as a @pysol_struct field.
_STRUCT_FIELD_TYPES = (Type.Int, Type.Float, Type.Double)

# struct class name -> [(field_name, Type), ...] in declaration order.
_struct_registry: Dict[str, List[Tuple[str, Type]]] = {}


def pysol_struct(cls: type) -> type:
    """
    Decorator for a plain class that declares a parsolgen struct type, used
    the same way as @dataclass: every class-level annotation must be one of
    Type.Int, Type.Float, or Type.Double.

    The decorated class becomes a real @dataclass (so it remains usable as a
    normal Python object) and is registered by name so that @pysol-decorated
    functions can reference it as the element type of an Array / GrowingArray,
    or as a variable's type annotation.
    """
    cls = dataclasses.dataclass(cls)

    fields: List[Tuple[str, Type]] = []
    for f in dataclasses.fields(cls):
        if f.type not in _STRUCT_FIELD_TYPES:
            raise UnsupportedStructFieldType(
                f"pysol_struct '{cls.__name__}': field '{f.name}' has unsupported "
                f"type {f.type!r}; supported types are Type.Int, Type.Float, Type.Double"
            )
        fields.append((f.name, f.type))

    if not fields:
        raise UnsupportedStructFieldType(
            f"pysol_struct '{cls.__name__}' declares no fields"
        )

    cls.__pysol_struct_fields__ = fields
    _struct_registry[cls.__name__] = fields
    return cls


# ── Public decorator ─────────────────────────────────────────────────────────

def pysol(func: Callable) -> Callable:
    """
    Decorator that validates and analyses the function source (ast + inspect)
    at decoration time and writes a <name>.sol file next to the source file.
    """
    analysis = _analyse(func)

    # Generate .sol file (lazy import avoids circular dependency)
    from pysol.sol_generator import generate_sol  # noqa: PLC0415
    sol_content = generate_sol(func, analysis)
    try:
        src_dir = os.path.dirname(os.path.abspath(inspect.getfile(func)))
    except (TypeError, OSError):
        src_dir = os.getcwd()
    sol_path = os.path.join(src_dir, f"{func.__name__}.sol")
    with open(sol_path, "w") as fh:
        fh.write(sol_content)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__pysol_config__ = get_config()
    wrapper.__pysol_sol_path__ = sol_path
    _registry[func.__name__] = wrapper
    return wrapper


# ── Compiler ──────────────────────────────────────────────────────────────────

def pysol_compile(
    name: str,
) -> Tuple[subprocess.CompletedProcess, subprocess.CompletedProcess]:
    """
    Compile the .sol file for the @pysol-decorated function called *name*.

    Step 1 — sol → cpp  (requires config key 'sgc'):
        <sgc> <name>.sol -o <name>.cpp

    Step 2 — sol → shared library  (requires config keys 'compiler' and
    optionally 'compile_options'):
        <compiler> <name>.sol [compile_options] -fPIC -shared -o <name>.so

    'compile_options' may be a list of strings or a single space-separated string.
    Both output files are placed next to the source .sol file.

    Returns a (result_sgc, result_compiler) tuple of CompletedProcess objects.

    Raises:
      KeyError          – function not registered, or 'sgc'/'compiler' missing from config.
      FileNotFoundError – binary not found (propagated from subprocess).
    """
    if name not in _registry:
        raise KeyError(f"No @pysol-decorated function named '{name}' registered")

    config = get_config()
    for required in ("sgc", "compiler"):
        if required not in config:
            raise KeyError(
                f"Config must contain a '{required}' key"
            )

    sol_path: str = _registry[name].__pysol_sol_path__
    sol_dir = os.path.dirname(sol_path)

    # Step 1: sgc → .cpp
    cpp_output = os.path.join(sol_dir, f"{name}.cpp")
    result_sgc = subprocess.run(
        [config["sgc"], sol_path, "-o", cpp_output, "--shared=true"],
        check=False,
    )

    # Step 2: compiler → .so
    raw_opts = config.get("compile_options", [])
    compile_opts: List[str] = (
        raw_opts if isinstance(raw_opts, list) else raw_opts.split()
    )
    so_output = os.path.join(sol_dir, f"{name}.so")
    result_compiler = subprocess.run(
        [config["compiler"], cpp_output] + compile_opts + ["-fPIC", "-shared", "-o", so_output],
        check=False,
    )

    if result_compiler.returncode != 0 or not os.path.exists(so_output):
        raise SolCompileError(
            f"C++ compiler returned exit code {result_compiler.returncode} "
            f"and '{so_output}' was not produced"
        )

    return result_sgc, result_compiler


# ── Runner ────────────────────────────────────────────────────────────────────

def pysol_run(name: str):
    """
    Load the shared library <name>.so produced by pysol_compile and invoke
    the exported C function of the same name.

    The .so is looked up next to the registered .sol file for *name*.

    Raises:
      KeyError                 – function not registered with @pysol.
      SolLibraryNotFoundError  – .so file absent or cannot be loaded by the OS.
      SolFunctionNotFoundError – symbol *name* not exported by the library.
    """
    if name not in _registry:
        raise KeyError(f"No @pysol-decorated function named '{name}' registered")

    sol_path: str = _registry[name].__pysol_sol_path__
    so_path = os.path.join(os.path.dirname(sol_path), f"{name}.so")

    if not os.path.exists(so_path):
        raise SolLibraryNotFoundError(
            f"Shared library not found: '{so_path}'"
        )

    try:
        # RTLD_GLOBAL, not ctypes' default RTLD_LOCAL: some compiled .sol
        # programs link against libraries (e.g. MKL) that themselves
        # dlopen plugins expecting to resolve symbols from their loading
        # process's global scope - with RTLD_LOCAL those plugin loads
        # fail at runtime with "undefined symbol" errors that never show
        # up when the same .so is instead the main executable (which the
        # OS always loads with globally-visible symbols).
        lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise SolLibraryNotFoundError(
            f"Failed to load shared library '{so_path}': {exc}"
        ) from exc

    try:
        init_fn = lib.init_pslib
    except AttributeError as exc:
        raise SolFunctionNotFoundError(
            f"Function 'init_pslib' not found in library '{so_path}'"
        ) from exc

    try:
        fini_fn = lib.fini_pslib
    except AttributeError as exc:
        raise SolFunctionNotFoundError(
            f"Function 'fini_pslib' not found in library '{so_path}'"
        ) from exc

    try:
        func = getattr(lib, name)
    except AttributeError as exc:
        raise SolFunctionNotFoundError(
            f"Function '{name}' not found in library '{so_path}'"
        ) from exc

    init_fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    init_fn.restype = ctypes.c_int
    fini_fn.restype = None

    argc = len(sys.argv)
    argv_arr = (ctypes.c_char_p * argc)(*[s.encode() for s in sys.argv])
    rc = init_fn(argc, argv_arr)
    if rc != 0:
        raise SolInitError(
            f"init_pslib returned error code {rc} for library '{so_path}'"
        )

    try:
        result = func()
    finally:
        fini_fn()
        handle = lib._handle
        del lib
        _ctypes.dlclose(handle)

    return result


# ── Pattern substitution ──────────────────────────────────────────────────────

def _extract_decl_name(decl: str) -> Optional[str]:
    """Return the declared identifier from a .sol declaration string.

    Handles Array<...> name[...] and primitive-type name forms.
    """
    m = re.search(r'>\s*(\w+)\s*\[', decl)
    if m:
        return m.group(1)
    m = re.search(r'(?:int|float|double|string|bool)\s+(\w+)', decl)
    return m.group(1) if m else None


def _pysol_apply_pattern(
    pattern_str: str,
    pattern_file: str,
    call_number: int = 1,
    array_types: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[str], List[str]]:
    """
    Read *pattern_file* and expand *pattern_str* using the template it defines.

    Pattern file format
    -------------------
    First line (header):
        replace "<call_name>(<param1>, <param2>, ...)":
    Remaining lines (the body):
        - ``$pN`` tokens: replaced by the corresponding positional argument
          from *pattern_str*.  Substitution is longest-first.
        - ``$$type(name)`` tokens: replaced by the ``.sol`` type keyword of
          array *name* (looked up from *array_types*).  *name* may itself be a
          ``$pN`` token — ``$param`` substitution runs first.  Unknown names
          are left unchanged.
        - ``$$declare: <decl>;`` lines: the declaration is extracted, its name
          is suffixed with ``_<call_number>``, and it is added to the returned
          *hoist* list rather than the body.  All occurrences of the original
          name in the body are also renamed.
        - ``$$import <import_string>;`` lines: an ``import <import_string>;``
          directive is collected in the returned *imports* list.  ``$param``
          and ``$$type()`` substitution apply before the line is collected.

    Returns
    -------
    Tuple[str, List[str], List[str]]
        ``(body, hoist, imports)`` where *body* is the expanded text,
        *hoist* is a list of renamed declarations, and *imports* is a list
        of ``import ...;`` strings to be added near the top of the .sol file.
    """
    with open(pattern_file) as fh:
        lines = fh.read().splitlines()

    if not lines:
        raise ValueError(f"Pattern file '{pattern_file}' is empty")

    # ── parse header ─────────────────────────────────────────────────────────
    # Expected: replace "<call_name>(<param1>, <param2>, ...)":
    header_match = re.match(
        r'\s*replace\s+"[^(]*\(([^)]*)\)"\s*:',
        lines[0],
    )
    if not header_match:
        raise ValueError(f"Invalid pattern file header: {lines[0]!r}")

    param_names: List[str] = [
        p.strip() for p in header_match.group(1).split(',') if p.strip()
    ]

    # ── parse pattern_str ────────────────────────────────────────────────────
    call_match = re.match(r'\s*\w+\(([^)]*)\)', pattern_str)
    if not call_match:
        raise ValueError(f"Invalid pattern string: {pattern_str!r}")

    actual_args = [a.strip() for a in call_match.group(1).split(',') if a.strip()]
    if len(actual_args) != len(param_names):
        raise ValueError(
            f"Pattern expects {len(param_names)} argument(s) but "
            f"pattern_str provides {len(actual_args)}"
        )

    # ── build $param substitution mapping (longest first to avoid prefix clash) ──
    param_mapping: Dict[str, str] = {
        f'${name}': arg for name, arg in zip(param_names, actual_args)
    }
    param_ordered = sorted(param_mapping.items(), key=lambda kv: -len(kv[0]))

    def _apply_params(s: str) -> str:
        for placeholder, replacement in param_ordered:
            s = s.replace(placeholder, replacement)
        return s

    def _apply_type(s: str) -> str:
        """Replace $$type(name) with the .sol type of array *name*."""
        if not array_types:
            return s
        def _sub(m: re.Match) -> str:  # type: ignore[type-arg]
            return array_types.get(m.group(1), m.group(0))
        return _DOLLAR_TYPE_RE.sub(_sub, s)

    def _process(s: str) -> str:
        """$param substitution → $$type resolution (in that order)."""
        return _apply_type(_apply_params(s))

    _DECLARE = '$$declare:'
    _IMPORT  = '$$import'
    body_lines = lines[1:]

    # ── pass 1: collect $$declare: names so we can rename them everywhere ──────
    decl_rename: Dict[str, str] = {}
    for line in body_lines:
        stripped = line.strip()
        if not stripped.startswith(_DECLARE):
            continue
        decl_str = _process(stripped[len(_DECLARE):].strip())
        name = _extract_decl_name(decl_str)
        if name is None:
            raise ValueError(f"Cannot extract name from declaration: {decl_str!r}")
        if name not in decl_rename:
            decl_rename[name] = f"{name}_{call_number}"

    # longest-first to avoid prefix collisions between declared names
    rename_ordered = sorted(decl_rename.items(), key=lambda kv: -len(kv[0]))

    def _apply_renames(s: str) -> str:
        for orig, renamed in rename_ordered:
            s = re.sub(r'\b' + re.escape(orig) + r'\b', renamed, s)
        return s

    # ── pass 2: produce body, hoist, and imports lists ────────────────────────
    hoist: List[str] = []
    imports: List[str] = []
    result: List[str] = []
    for line in body_lines:
        stripped = line.strip()
        if stripped.startswith(_DECLARE):
            decl_str = _process(stripped[len(_DECLARE):].strip())
            hoist.append(_apply_renames(decl_str))
        elif stripped.startswith(_IMPORT):
            import_body = _process(stripped[len(_IMPORT):].strip())
            imports.append(f"import {import_body}")
        else:
            result.append(_apply_renames(_process(line)))

    return '\n'.join(result), hoist, imports


# ── Array utilities ───────────────────────────────────────────────────────────

def pysol_update_array(
    source: Array,
    source_version: int,
    destination: Array,
    destination_version: int,
) -> None:
    pass


def pysol_update_array_3d_26_full_sum(
    source: Array,
    source_version: int,
    destination: Array,
    destination_version: int,
) -> None:
    """26-neighbor additive halo exchange for an n_dims=3 Array - see
    exchange_3d_26_full_sum.sol. Generalizes examples/pic3d_v2's
    exchange_density_additive_block: every block's data overlaps its
    neighbors' by a 2-cell-wide slab on each face/edge/corner, and the
    correct destination value there is the SUM of every overlapping
    block's contribution, not a copy from one of them."""
    pass
