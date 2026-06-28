from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Dict, List, Optional, Tuple

from pysol.decorator import (
    NotImportedKernel,
    _Analysis,
    _ArrayInfo,
    _is_array_call,
    _target_name,
    _unparse,
)
from pysol.kernel import KernelArgument
from pysol.types import Type


# ── Type maps ────────────────────────────────────────────────────────────────

# AST attribute name → Type enum  (used when parsing KernelArgument calls)
_AST_TYPE_MAP: Dict[str, Type] = {
    "Int":    Type.Int,
    "Float":  Type.Float,
    "Double": Type.Double,
    "String": Type.String,
    "Bool":   Type.Bool,
}

# Type enum → .sol keyword  (used for code generation)
_TYPE_TO_SOL: Dict[Type, str] = {
    Type.Int:    "int",
    Type.Float:  "float",
    Type.Double: "double",
    Type.String: "string",
    Type.Bool:   "bool",
}

# Legacy string-keyed map kept for array-declaration generation (_ArrayInfo stores strings)
_SOL_TYPE: Dict[str, str] = {f"Type.{k}": v for k, v in
                              zip(_AST_TYPE_MAP, _TYPE_TO_SOL.values())}

_CMP_OPS = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=",
}
_BIN_OPS = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.Mod: "%", ast.FloorDiv: "/",
}


# ── AST helpers ───────────────────────────────────────────────────────────────

def _is_import_kernel_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "import_kernel") or (
        isinstance(func, ast.Attribute) and func.attr == "import_kernel"
    )


def _ast_to_kernel_arg(node: ast.Call) -> Optional[KernelArgument]:
    """
    Parse a KernelArgument(name, is_input, type[, n_dims]) AST call node into
    a real KernelArgument object.  Returns None if the structure is unrecognised.
    """
    args = node.args
    if len(args) < 3:
        return None

    # name — string constant
    if not isinstance(args[0], ast.Constant) or not isinstance(args[0].value, str):
        return None
    name = args[0].value

    # is_input — bool constant
    if not isinstance(args[1], ast.Constant) or not isinstance(args[1].value, bool):
        return None
    is_input = bool(args[1].value)

    # type — Attribute like Type.Int  or  pysol.Type.Int
    if not isinstance(args[2], ast.Attribute):
        return None
    type_name = args[2].attr
    if type_name not in _AST_TYPE_MAP:
        return None
    arg_type = _AST_TYPE_MAP[type_name]

    # n_dims — optional 4th positional arg or keyword
    n_dims = 0
    if len(args) >= 4 and isinstance(args[3], ast.Constant) and isinstance(args[3].value, int):
        n_dims = args[3].value
    for kw in node.keywords:
        if (kw.arg == "n_dims"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, int)):
            n_dims = kw.value.value

    try:
        return KernelArgument(name, is_input, arg_type, n_dims)
    except ValueError:
        return None


def _parse_size_list(size_str: str) -> List[str]:
    """'[30, 40, 50]' → ['30', '40', '50']"""
    s = size_str.strip().lstrip("[").rstrip("]")
    return [x.strip() for x in s.split(",") if x.strip()]


def _collect_kernel_imports(
    func_def: ast.FunctionDef,
) -> Dict[str, Tuple[str, List[KernelArgument]]]:
    """
    Return {var_name: (kernel_name, [KernelArgument, ...])}
    for every  var = import_kernel(...)  assignment in func_def.
    """
    result: Dict[str, Tuple[str, List[KernelArgument]]] = {}
    for node in ast.walk(func_def):
        if not (isinstance(node, ast.Assign) and _is_import_kernel_call(node.value)):
            continue
        var = _target_name(node.targets[0])
        call_args = node.value.args
        if not call_args:
            continue
        first = call_args[0]
        kernel_name = str(first.value if isinstance(first, ast.Constant) else _unparse(first))

        kernel_args: List[KernelArgument] = []
        if len(call_args) >= 2 and isinstance(call_args[1], ast.List):
            for elt in call_args[1].elts:
                if isinstance(elt, ast.Call):
                    karg = _ast_to_kernel_arg(elt)
                    if karg is not None:
                        kernel_args.append(karg)

        result[var] = (kernel_name, kernel_args)
    return result


# ── Import argument declaration ───────────────────────────────────────────────

def _gen_import_arg_decl(arg: KernelArgument) -> List[str]:
    """
    Generate the declaration token(s) for one KernelArgument in an import line.

    Scalar:
        KernelArgument("x", True,  Type.Int)          → ["in int x"]
        KernelArgument("y", False, Type.Double)        → ["out double y"]

    Array element  (n_dims=N):
        KernelArgument("a", False, Type.Float, n_dims=2)
            → ["in a_n1", "in a_n2", "out Array<float>a[a_n1][a_n2]"]
    """
    direction = "in" if arg.is_input else "out"
    sol_type = _TYPE_TO_SOL.get(arg.type, arg.type.value.lower())

    if not arg.is_array_element:
        return [f"{direction} {sol_type} {arg.name}"]

    size_params = [f"{arg.name}_n{i + 1}" for i in range(arg.n_dims)]
    parts: List[str] = [f"in {sp}" for sp in size_params]
    dims = "".join(f"[{sp}]" for sp in size_params)
    parts.append(f"{direction} Array<{sol_type}> {arg.name}{dims}")
    return parts


# ── Sol writer ────────────────────────────────────────────────────────────────

class _SolWriter:
    def __init__(
        self,
        func_name: str,
        func_def: ast.FunctionDef,
        imported_kernels: Dict[str, Tuple[str, List[KernelArgument]]],
        arrays: Dict[str, _ArrayInfo],
    ) -> None:
        self._name = func_name
        self._func_def = func_def
        # Split for independent lookup
        self._kernels: Dict[str, str] = {
            var: name for var, (name, _) in imported_kernels.items()
        }
        self._kernel_arg_lists: Dict[str, List[KernelArgument]] = {
            var: kargs for var, (_, kargs) in imported_kernels.items()
        }
        self._arrays = arrays

    # ── public ───────────────────────────────────────────────────────────────

    def render(self) -> str:
        lines: List[str] = []

        # 1. import declarations
        for var_name, kernel_name in self._kernels.items():
            kargs = self._kernel_arg_lists.get(var_name, [])
            decls: List[str] = []
            for karg in kargs:
                decls.extend(_gen_import_arg_decl(karg))
            lines.append(f"import {kernel_name}({', '.join(decls)});")
        if self._kernels:
            lines.append("")

        # 2. function declaration + body
        lines.append(f"export sub {self._name}()")
        lines.append("{")
        lines.extend(self._stmts(self._func_def.body, level=1))
        lines.append("}")

        return "\n".join(lines) + "\n"

    # ── statement list ────────────────────────────────────────────────────────

    def _stmts(self, stmts: List[ast.stmt], level: int) -> List[str]:
        out: List[str] = []
        for s in stmts:
            out.extend(self._stmt(s, level))
        return out

    def _stmt(self, node: ast.stmt, level: int) -> List[str]:
        ind = "    " * level

        if isinstance(node, (ast.Pass, ast.Import, ast.ImportFrom)):
            return []

        if isinstance(node, ast.Assign):
            if _is_import_kernel_call(node.value):
                return []
            if _is_array_call(node.value):
                var = _target_name(node.targets[0])
                info = self._arrays.get(var)
                return [ind + self._array_decl(info)] if info else []
            var = _target_name(node.targets[0])
            return [f"{ind}{var} = {self._expr(node.value)};"]

        if isinstance(node, ast.AnnAssign) and node.value is not None:
            if _is_array_call(node.value):
                var = _target_name(node.target)
                info = self._arrays.get(var)
                return [ind + self._array_decl(info)] if info else []
            return []

        if isinstance(node, ast.AugAssign):
            ops = {ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/="}
            op = ops.get(type(node.op), "?=")
            return [f"{ind}{_unparse(node.target)} {op} {self._expr(node.value)};"]

        if isinstance(node, ast.For):
            return self._for(node, level)

        if isinstance(node, ast.While):
            return self._while(node, level)

        if isinstance(node, ast.If):
            return self._if(node, level)

        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return [ind + self._call_stmt(node.value) + ";"]

        return []

    # ── compound statements ───────────────────────────────────────────────────

    def _for(self, node: ast.For, level: int) -> List[str]:
        ind = "    " * level
        var = _target_name(node.target)
        c_args = [self._expr(a) for a in node.iter.args]  # type: ignore[attr-defined]
        if len(c_args) == 1:
            init, cond, step = f"int {var} = 0", f"{var} < {c_args[0]}", f"{var}++"
        elif len(c_args) == 2:
            init = f"int {var} = {c_args[0]}"
            cond = f"{var} < {c_args[1]}"
            step = f"{var}++"
        else:
            init = f"int {var} = {c_args[0]}"
            cond = f"{var} < {c_args[1]}"
            step = f"{var} += {c_args[2]}"
        lines = [f"{ind}for ({init}; {cond}; {step}) {{"]
        lines.extend(self._stmts(node.body, level + 1))
        lines.append(f"{ind}}}")
        return lines

    def _while(self, node: ast.While, level: int) -> List[str]:
        ind = "    " * level
        lines = [f"{ind}while ({self._cond(node.test)}) {{"]
        lines.extend(self._stmts(node.body, level + 1))
        lines.append(f"{ind}}}")
        return lines

    def _if(self, node: ast.If, level: int) -> List[str]:
        ind = "    " * level
        lines = [f"{ind}if ({self._cond(node.test)}) {{"]
        lines.extend(self._stmts(node.body, level + 1))
        if node.orelse:
            lines.append(f"{ind}}} else {{")
            lines.extend(self._stmts(node.orelse, level + 1))
        lines.append(f"{ind}}}")
        return lines

    # ── call statements ───────────────────────────────────────────────────────

    def _call_stmt(self, call: ast.Call) -> str:
        func_str = _unparse(call.func)
        if func_str not in self._kernels:
            raise NotImportedKernel(f"'{func_str}' is not an imported kernel")
        kernel_name = self._kernels[func_str]
        c_args: List[str] = []
        for arg in call.args:
            c_args.extend(self._expand_arg(arg))
        return f"{kernel_name}({', '.join(c_args)})"

    def _expand_arg(self, node: ast.AST) -> List[str]:
        if isinstance(node, ast.Subscript):
            array_var, version, coords, block_idx = self._parse_subscript(node)
            if array_var and array_var in self._arrays:
                return self._subscript_expansion(array_var, version, coords, block_idx)

        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in self._arrays):
            return self._get_expansion(node)

        return [self._expr(node)]

    def _parse_subscript(
        self, node: ast.Subscript
    ) -> Tuple[Optional[str], Optional[str], List[str], List[str]]:
        block_idx: List[str] = []
        cur: ast.AST = node

        while isinstance(cur, ast.Subscript):
            sl = cur.slice
            if hasattr(ast, "Index") and isinstance(sl, ast.Index):
                sl = sl.value  # type: ignore[attr-defined]
            if isinstance(sl, ast.Tuple):
                break
            block_idx = [self._expr(sl)] + block_idx
            cur = cur.value

        if not isinstance(cur, ast.Subscript):
            return None, None, [], []

        inner_sl = cur.slice
        if hasattr(ast, "Index") and isinstance(inner_sl, ast.Index):
            inner_sl = inner_sl.value  # type: ignore[attr-defined]

        if not isinstance(cur.value, ast.Name):
            return None, None, [], []

        array_var = cur.value.id
        elts = (
            [self._expr(e) for e in inner_sl.elts]
            if isinstance(inner_sl, ast.Tuple)
            else [self._expr(inner_sl)]
        )
        version = elts[0] if elts else None
        coords = elts[1:] if elts else []
        return array_var, version, coords, block_idx

    def _subscript_expansion(
        self,
        array_var: str,
        version: Optional[str],
        coords: List[str],
        block_idx: List[str],
    ) -> List[str]:
        block_sizes = _parse_size_list(self._arrays[array_var].block_size)
        ref = f"{array_var}{{{version or '?'}}}"
        for c in coords:
            ref += f"[{c}]"
        for b in block_idx:
            ref += f"[{b}]"
        return block_sizes + [ref]

    def _get_expansion(self, call: ast.Call) -> List[str]:
        array_var: str = call.func.value.id  # type: ignore[attr-defined]
        block_sizes = _parse_size_list(self._arrays[array_var].block_size)
        version = self._expr(call.args[0])
        coords_node = call.args[1]
        coords = (
            [self._expr(e) for e in coords_node.elts]
            if isinstance(coords_node, ast.List)
            else [self._expr(coords_node)]
        )
        ref = f"{array_var}{{{version}}}"
        for c in coords:
            ref += f"[{c}]"
        return block_sizes + [ref]

    # ── expression generation ─────────────────────────────────────────────────

    def _cond(self, node: ast.AST) -> str:
        if isinstance(node, ast.Call):
            args = [_unparse(a) for a in node.args]
            return f"{_unparse(node.func)}({', '.join(args)})"
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = _CMP_OPS.get(type(node.ops[0]), "?")
            return f"{_unparse(node.left)} {op} {_unparse(node.comparators[0])}"
        if isinstance(node, ast.BoolOp):
            op_str = " && " if isinstance(node.op, ast.And) else " || "
            return op_str.join(_unparse(v) for v in node.values)
        return _unparse(node)

    def _expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            if isinstance(node.value, str):
                return f'"{node.value}"'
            return str(node.value)
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op), "?")
            return f"{self._expr(node.left)} {op} {self._expr(node.right)}"
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return f"!{self._expr(node.operand)}"
            ops = {ast.USub: "-", ast.UAdd: "+", ast.Invert: "~"}
            return f"{ops.get(type(node.op), '?')}{self._expr(node.operand)}"
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            op = _CMP_OPS.get(type(node.ops[0]), "?")
            return f"{self._expr(node.left)} {op} {self._expr(node.comparators[0])}"
        if isinstance(node, ast.BoolOp):
            op_str = " && " if isinstance(node.op, ast.And) else " || "
            return op_str.join(self._expr(v) for v in node.values)
        if isinstance(node, ast.Call):
            args = [self._expr(a) for a in node.args]
            return f"{_unparse(node.func)}({', '.join(args)})"
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if hasattr(ast, "Index") and isinstance(sl, ast.Index):
                sl = sl.value  # type: ignore[attr-defined]
            idx = (
                ", ".join(self._expr(e) for e in sl.elts)
                if isinstance(sl, ast.Tuple)
                else self._expr(sl)
            )
            return f"{self._expr(node.value)}[{idx}]"
        if isinstance(node, ast.Attribute):
            return f"{self._expr(node.value)}.{node.attr}"
        if isinstance(node, ast.Tuple):
            return ", ".join(self._expr(e) for e in node.elts)
        return _unparse(node)

    # ── array declaration ─────────────────────────────────────────────────────

    def _array_decl(self, info: _ArrayInfo) -> str:
        c_type = _SOL_TYPE.get(info.elem_type, info.elem_type)
        block_dims = "".join(f"[{s}]" for s in _parse_size_list(info.block_size))
        array_dims = "".join(f"[{s}]" for s in _parse_size_list(info.array_size))
        return f"Array<{c_type}{block_dims}> {info.var}{array_dims};"


# ── Public API ────────────────────────────────────────────────────────────────

def generate_sol(func, analysis: _Analysis) -> str:
    """Return the .sol file content for a @pysol-decorated function."""
    source_lines, _ = inspect.getsourcelines(func)
    source = textwrap.dedent("".join(source_lines))
    tree = ast.parse(source)

    func_def: Optional[ast.FunctionDef] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            func_def = node
            break

    if func_def is None:
        raise ValueError(f"Could not locate FunctionDef for '{func.__name__}'")

    imported_kernels = _collect_kernel_imports(func_def)
    arrays = {info.var: info for info in analysis.arrays}
    return _SolWriter(func.__name__, func_def, imported_kernels, arrays).render()
