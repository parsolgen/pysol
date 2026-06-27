from __future__ import annotations

import ast
import functools
import inspect
import textwrap
from dataclasses import dataclass, field
from typing import Callable, List


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
    if hasattr(ast, "Index") and isinstance(node, ast.Index):  # removed in 3.9
        return _unparse(node.value)  # type: ignore[attr-defined]
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


def _is_array_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "Array") or (
        isinstance(func, ast.Attribute) and func.attr == "Array"
    )


# ── Info containers ──────────────────────────────────────────────────────────

_ARRAY_PARAMS = ("n_dims", "n_block_dims", "array_size", "block_size", "type")


@dataclass
class _ArrayInfo:
    line: int
    var: str
    n_dims: str
    n_block_dims: str
    array_size: str
    block_size: str
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
class _Analysis:
    name: str
    arrays: List[_ArrayInfo] = field(default_factory=list)
    for_loops: List[_ForInfo] = field(default_factory=list)
    while_loops: List[_WhileInfo] = field(default_factory=list)


# ── Static analysis ──────────────────────────────────────────────────────────

def _extract_array(line: int, var: str, call: ast.Call) -> _ArrayInfo:
    values = {p: "?" for p in _ARRAY_PARAMS}
    for i, arg in enumerate(call.args):
        if i < len(_ARRAY_PARAMS):
            values[_ARRAY_PARAMS[i]] = _unparse(arg)
    for kw in call.keywords:
        if kw.arg in values:
            values[kw.arg] = _unparse(kw.value)
    return _ArrayInfo(
        line=line,
        var=var,
        n_dims=values["n_dims"],
        n_block_dims=values["n_block_dims"],
        array_size=values["array_size"],
        block_size=values["block_size"],
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

        elif isinstance(node, ast.For):
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

    analysis.arrays.sort(key=lambda x: x.line)
    analysis.for_loops.sort(key=lambda x: x.line)
    analysis.while_loops.sort(key=lambda x: x.line)
    return analysis


# ── Output ───────────────────────────────────────────────────────────────────

_SEP = "─" * 56


def _print_analysis(a: _Analysis) -> None:
    print(f"\n[pysol] {a.name}")
    print(_SEP)

    print(f"  Arrays ({len(a.arrays)}):")
    for info in a.arrays:
        print(f"    line {info.line:<4} {info.var} = Array(...)")
        print(f"           n_dims={info.n_dims}  n_block_dims={info.n_block_dims}")
        print(f"           array_size={info.array_size}  block_size={info.block_size}")
        print(f"           type={info.elem_type}")

    print(f"  For-loops ({len(a.for_loops)}):")
    for info in a.for_loops:
        print(f"    line {info.line:<4} for {info.target} in {info.iterable}")

    print(f"  While-loops ({len(a.while_loops)}):")
    for info in a.while_loops:
        print(f"    line {info.line:<4} while {info.condition}")

    print(_SEP)


# ── Public decorator ─────────────────────────────────────────────────────────

def pysol(func: Callable) -> Callable:
    """
    Decorator that on each call prints pysol.Array declarations, for-loops,
    and while-loops found in the decorated function's source code.
    Parsing is done once at decoration time via ast + inspect.
    """
    analysis = _analyse(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _print_analysis(analysis)
        return func(*args, **kwargs)

    return wrapper
