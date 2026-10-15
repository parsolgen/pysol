from __future__ import annotations

import ast
import inspect
import os
import textwrap
from typing import Dict, List, Optional, Set, Tuple

_AST_INDEX = getattr(ast, "Index", None)

_EXCHANGE_PATTERN = os.path.join(os.path.dirname(__file__), "exchange.sol")
_EXCHANGE_PATTERN_3D = os.path.join(os.path.dirname(__file__), "exchange_3d.sol")

from pysol.decorator import (
    ArrayOperationError,
    NotImportedKernel,
    UnsupportedCallError,
    UnsupportedVariableType,
    _Analysis,
    _ArrayInfo,
    _is_array_call,
    _pysol_apply_pattern,
    _struct_registry,
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

# Type enum → .sol keyword  (used for import/array declarations)
_TYPE_TO_SOL: Dict[Type, str] = {
    Type.Int:    "int",
    Type.Float:  "float",
    Type.Double: "double",
    Type.String: "string",
    Type.Bool:   "bool",
}

# Variable declarations use a reduced type set: float collapses to double
_VAR_TYPE_TO_SOL: Dict[Type, str] = {
    Type.Int:    "int",
    Type.Float:  "float",
    Type.Double: "double",
    Type.String: "string",
    Type.Bool:   "bool",
}

_SUPPORTED_VAR_TYPES = ", ".join(f"Type.{t.name}" for t in _VAR_TYPE_TO_SOL)

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
# Precedence levels for binary operators (higher = tighter binding).
_OP_PREC: Dict[type, int] = {
    ast.Add: 1, ast.Sub: 1,
    ast.Mult: 2, ast.Div: 2, ast.Mod: 2, ast.FloorDiv: 2,
}
# Operators for which `a op (b op c)  !=  (a op b) op c` — right-side child
# with equal precedence must be parenthesised.
_NON_COMM_OPS = {ast.Sub, ast.Div, ast.Mod, ast.FloorDiv}


# ── AST helpers ───────────────────────────────────────────────────────────────

def _is_import_kernel_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "import_kernel") or (
        isinstance(func, ast.Attribute) and func.attr == "import_kernel"
    )


# Array-reduction operators: sum(A) / max(A) / min(A) → reduce(%<op>, A[:]...[:])
_REDUCE_OPS = {"sum", "max", "min"}


def _reduce_op_name(func: ast.AST) -> Optional[str]:
    if isinstance(func, ast.Name) and func.id in _REDUCE_OPS:
        return func.id
    if isinstance(func, ast.Attribute) and func.attr in _REDUCE_OPS:
        return func.attr
    return None


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

    # type — Attribute like Type.Int / pysol.Type.Int, or a bare Name
    # referencing a @pysol_struct-registered struct type.
    if isinstance(args[2], ast.Name) and args[2].id in _struct_registry:
        arg_type: object = args[2].id
    elif isinstance(args[2], ast.Attribute):
        type_name = args[2].attr
        if type_name not in _AST_TYPE_MAP:
            return None
        arg_type = _AST_TYPE_MAP[type_name]
    else:
        return None

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


def _annotation_to_sol_type(annotation: ast.AST) -> Optional[str]:
    """
    Map a Python type annotation to its .sol keyword.

    Handles all of:
      Type.Int / pysol.Type.Int / pysol.Int  →  "int"
      Type.Float / pysol.Float               →  "float"
      SomeStruct (a @pysol_struct class)     →  "SomeStruct"
      etc.

    Returns None for unrecognised annotations.
    """
    if isinstance(annotation, ast.Name) and annotation.id in _struct_registry:
        return annotation.id
    if not isinstance(annotation, ast.Attribute):
        return None
    attr = annotation.attr
    if attr not in _AST_TYPE_MAP:
        return None
    return _VAR_TYPE_TO_SOL[_AST_TYPE_MAP[attr]]


def _validate_array_operands(a: _ArrayInfo, b: _ArrayInfo) -> None:
    """Raise ArrayOperationError if A and B differ in any structural property."""
    mismatches: List[str] = []
    if a.n_dims != b.n_dims:
        mismatches.append(f"n_dims ({a.n_dims} vs {b.n_dims})")
    if a.n_block_dims != b.n_block_dims:
        mismatches.append(f"n_block_dims ({a.n_block_dims} vs {b.n_block_dims})")
    if a.array_size != b.array_size:
        mismatches.append(f"array_size ({a.array_size} vs {b.array_size})")
    if a.block_size != b.block_size:
        mismatches.append(f"block_size ({a.block_size} vs {b.block_size})")
    if a.elem_type != b.elem_type:
        mismatches.append(f"type ({a.elem_type} vs {b.elem_type})")
    if mismatches:
        raise ArrayOperationError(
            f"Arrays '{a.var}' and '{b.var}' are incompatible: "
            + ", ".join(mismatches)
        )


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


# ── Struct type declaration ───────────────────────────────────────────────────

def _render_struct_decl(name: str) -> str:
    """
    'type <Name> : struct { <f1> : <t1>, <f2> : <t2> };' for a @pysol_struct
    class registered under *name*.
    """
    fields = _struct_registry[name]
    field_strs = [f"{fname} : {_TYPE_TO_SOL[ftype]}" for fname, ftype in fields]
    return f"type {name} : struct {{ {', '.join(field_strs)} }};"


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

    A @pysol_struct class may be used in place of a Type.X enum member; its
    name (a plain string) is used verbatim as the .sol type keyword.
    """
    direction = "in" if arg.is_input else "out"
    if isinstance(arg.type, str):
        sol_type = arg.type
    else:
        sol_type = _TYPE_TO_SOL.get(arg.type, arg.type.value.lower())

    if not arg.is_array_element:
        return [f"{direction} {sol_type} {arg.name}"]

    size_params = [f"{arg.name}_n{i + 1}" for i in range(arg.n_dims)]
    parts: List[str] = [f"in int {sp}" for sp in size_params]
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
        self._declared_vars: Set[str] = set()
        self._update_array_call_count: int = 0
        # Each entry: (referenced_array_vars, hoisted_decl_strings) per call.
        self._pending_hoists: List[Tuple[List[str], List[str]]] = []
        self._hoist_imports: List[str] = []
        self._seen_imports: Set[str] = set()
        # @pysol_struct types referenced by name, in first-seen order.
        self._used_structs: List[str] = []
        self._seen_structs: Set[str] = set()
        for info in arrays.values():
            self._mark_struct_used(info.elem_type)
        for kargs in self._kernel_arg_lists.values():
            for karg in kargs:
                if isinstance(karg.type, str):
                    self._mark_struct_used(karg.type)

    def _mark_struct_used(self, name: str) -> None:
        if name in _struct_registry and name not in self._seen_structs:
            self._seen_structs.add(name)
            self._used_structs.append(name)

    # ── public ───────────────────────────────────────────────────────────────

    def render(self) -> str:
        lines: List[str] = []

        # Generate body first — populates _pending_hoists and _hoist_imports.
        body_lines = self._stmts(self._func_def.body, level=1)

        # Insert each call's hoisted declarations after the last declaration of any
        # array it references.  Scan original body_lines for insertion positions,
        # then insert in descending-position order (not call order: an array may
        # be the target of more than one pysol_update_array call - e.g. ping-ponged
        # once per timestep - so positions are not monotonic with call order, and
        # inserting in call order can insert at a stale index already shifted by
        # an earlier insertion that landed at or below it). Processing strictly by
        # descending position guarantees every insertion lands at an index greater
        # than all positions not yet processed, so their recorded positions stay valid.
        positions: List[int] = []
        for ref_vars, _ in self._pending_hoists:
            last_idx = -1
            for i, line in enumerate(body_lines):
                for var in ref_vars:
                    if "Array<" in line and (f" {var}[" in line or f" {var};" in line):
                        last_idx = i
                        break
            positions.append(last_idx)

        order = sorted(range(len(self._pending_hoists)), key=lambda k: positions[k], reverse=True)
        for k in order:
            _, hoisted_decls = self._pending_hoists[k]
            if not hoisted_decls:
                continue
            insert_at = positions[k] + 1  # 0 when no array decl was found
            for j, decl in enumerate(hoisted_decls):
                body_lines.insert(insert_at + j, "    " + decl)

        # 1. struct type declarations, for every @pysol_struct referenced by name
        for struct_name in self._used_structs:
            lines.append(_render_struct_decl(struct_name))
        if self._used_structs:
            lines.append("")

        # 2. import declarations (kernel imports + $$import-derived)
        for var_name, kernel_name in self._kernels.items():
            kargs = self._kernel_arg_lists.get(var_name, [])
            decls: List[str] = []
            for karg in kargs:
                decls.extend(_gen_import_arg_decl(karg))
            lines.append(f"import {kernel_name}({', '.join(decls)});")
        for imp in self._hoist_imports:
            lines.append(imp)
        if self._kernels or self._hoist_imports:
            lines.append("")

        # 3. function declaration + body
        lines.append(f"export sub {self._name}()")
        lines.append("{")
        lines.extend(body_lines)
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
            if self._is_array_binop(node.value):
                return self._array_binop_stmt(node.targets[0], node.value, ind)
            var = _target_name(node.targets[0])
            if var in self._declared_vars:
                return [f"{ind}{var} = {self._expr(node.value)};"]
            raise UnsupportedVariableType(
                f"Variable '{var}' has no type annotation; "
                f"supported types are: {_SUPPORTED_VAR_TYPES}"
            )

        if isinstance(node, ast.AnnAssign) and node.value is not None:
            if _is_array_call(node.value):
                var = _target_name(node.target)
                info = self._arrays.get(var)
                return [ind + self._array_decl(info)] if info else []
            if self._is_array_binop(node.value):
                return self._array_binop_stmt(node.target, node.value, ind)
            var = _target_name(node.target)
            if var in self._declared_vars:
                return [f"{ind}{var} = {self._expr(node.value)};"]
            sol_type = _annotation_to_sol_type(node.annotation)
            if sol_type is None:
                ann_str = _unparse(node.annotation)
                raise UnsupportedVariableType(
                    f"Variable '{var}' has unsupported type annotation '{ann_str}'; "
                    f"supported types are: {_SUPPORTED_VAR_TYPES}"
                )
            self._mark_struct_used(sol_type)
            self._declared_vars.add(var)
            return [f"{ind}{sol_type} {var} = {self._expr(node.value)};"]

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
            call = node.value
            if _unparse(call.func) in ("pysol_update_array", "pysol.pysol_update_array"):
                return self._update_array_stmt(call, ind)
            return [ind + self._call_stmt(call) + ";"]

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

    def _update_array_stmt(self, call: ast.Call, ind: str) -> List[str]:
        """Handle pysol_update_array(source, source_version, destination, destination_version)."""
        if len(call.args) != 4:
            raise UnsupportedCallError(
                "pysol_update_array expects 4 positional arguments: "
                "source, source_version, destination, destination_version"
            )
        src_arg, src_ver_arg, dst_arg, dst_ver_arg = call.args

        if not isinstance(src_arg, ast.Name) or src_arg.id not in self._arrays:
            raise UnsupportedCallError(
                "pysol_update_array: argument 1 (source) must be a known array variable"
            )
        if not isinstance(dst_arg, ast.Name) or dst_arg.id not in self._arrays:
            raise UnsupportedCallError(
                "pysol_update_array: argument 3 (destination) must be a known array variable"
            )

        src_info = self._arrays[src_arg.id]
        dst_info = self._arrays[dst_arg.id]

        if src_info.n_dims not in ("2", "3"):
            raise UnsupportedCallError(
                f"pysol_update_array: source array '{src_arg.id}' must have n_dims=2 "
                f"or n_dims=3, got n_dims={src_info.n_dims}"
            )
        if dst_info.n_dims not in ("2", "3"):
            raise UnsupportedCallError(
                f"pysol_update_array: destination array '{dst_arg.id}' must have n_dims=2 "
                f"or n_dims=3, got n_dims={dst_info.n_dims}"
            )
        if src_info.n_dims != dst_info.n_dims:
            raise UnsupportedCallError(
                f"pysol_update_array: source array '{src_arg.id}' (n_dims={src_info.n_dims}) "
                f"and destination array '{dst_arg.id}' (n_dims={dst_info.n_dims}) "
                "must have matching n_dims"
            )

        _validate_array_operands(src_info, dst_info)

        src_ver = self._expr(src_ver_arg)
        dst_ver = self._expr(dst_ver_arg)
        pattern_str = (
            f"pysol_update_array({src_arg.id}, {src_ver}, {dst_arg.id}, {dst_ver})"
        )
        array_types = {
            name: _SOL_TYPE.get(info.elem_type, info.elem_type)
            for name, info in self._arrays.items()
        }
        pattern_file = _EXCHANGE_PATTERN_3D if src_info.n_dims == "3" else _EXCHANGE_PATTERN
        self._update_array_call_count += 1
        expanded, hoisted, imports = _pysol_apply_pattern(
            pattern_str, pattern_file, self._update_array_call_count, array_types
        )
        self._pending_hoists.append(([src_arg.id, dst_arg.id], hoisted))
        for imp in imports:
            if imp not in self._seen_imports:
                self._seen_imports.add(imp)
                self._hoist_imports.append(imp)
        return [ind + line for line in expanded.splitlines()]

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
            # Plain subscript A[i][j] with no version tuple — peel to base name.
            cur: ast.AST = node
            while isinstance(cur, ast.Subscript):
                cur = cur.value
            if isinstance(cur, ast.Name) and cur.id in self._arrays:
                return self._block_n_args(cur.id) + [self._expr(node)]

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
            if _AST_INDEX is not None and isinstance(sl, _AST_INDEX):
                sl = sl.value  # type: ignore[arg-type]
            if isinstance(sl, ast.Tuple):
                break
            block_idx = [self._expr(sl)] + block_idx
            cur = cur.value

        if not isinstance(cur, ast.Subscript):
            return None, None, [], []

        inner_sl = cur.slice
        if _AST_INDEX is not None and isinstance(inner_sl, _AST_INDEX):
            inner_sl = inner_sl.value  # type: ignore[arg-type]

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

    def _block_n_args(self, array_var: str) -> List[str]:
        """Return [%block_n(array_var, 0), ..., %block_n(array_var, n_block_dims-1)]."""
        n = int(self._arrays[array_var].n_block_dims)
        return [f"%block_n({array_var}, {i})" for i in range(n)]

    def _subscript_expansion(
        self,
        array_var: str,
        version: Optional[str],
        coords: List[str],
        block_idx: List[str],
    ) -> List[str]:
        ref = f"{array_var}{{{version or '?'}}}"
        for c in coords:
            ref += f"[{c}]"
        for b in block_idx:
            ref += f"[{b}]"
        return self._block_n_args(array_var) + [ref]

    def _get_expansion(self, call: ast.Call) -> List[str]:
        """array.get(version, [e1, e2, ...]) → [%block_n(array, 0), ..., array{version}[e1][e2]...]"""
        array_var: str = call.func.value.id  # type: ignore[attr-defined]
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
        return self._block_n_args(array_var) + [ref]

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
            pp = _OP_PREC.get(type(node.op), 0)
            left_str = self._expr(node.left)
            if isinstance(node.left, ast.BinOp) and _OP_PREC.get(type(node.left.op), 0) < pp:
                left_str = f"({left_str})"
            right_str = self._expr(node.right)
            if isinstance(node.right, ast.BinOp):
                rp = _OP_PREC.get(type(node.right.op), 0)
                if rp < pp or (rp == pp and type(node.op) in _NON_COMM_OPS):
                    right_str = f"({right_str})"
            return f"{left_str} {op} {right_str}"
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
            op = _reduce_op_name(node.func)
            if op is not None:
                return self._reduce_expr(op, node)
            args = [self._expr(a) for a in node.args]
            return f"{_unparse(node.func)}({', '.join(args)})"
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if _AST_INDEX is not None and isinstance(sl, _AST_INDEX):
                sl = sl.value  # type: ignore[arg-type]
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

    def _reduce_expr(self, op: str, call: ast.Call) -> str:
        """sum(A) / max(A) / min(A) → reduce(%<op>, A[:]...[:]) with one [:] per array dim."""
        if (len(call.args) != 1 or call.keywords
                or not isinstance(call.args[0], ast.Name)
                or call.args[0].id not in self._arrays):
            raise UnsupportedCallError(
                f"{op}(): expects a single known array variable as its argument"
            )
        array_var = call.args[0].id
        info = self._arrays[array_var]
        try:
            ndims = int(info.n_dims)
        except (TypeError, ValueError):
            ndims = 0
        if ndims not in (1, 2, 3):
            raise UnsupportedCallError(
                f"{op}(): array '{array_var}' must have n_dims in {{1, 2, 3}}, "
                f"got n_dims={info.n_dims}"
            )
        return f"reduce(%{op}, {array_var}{'[:]' * ndims})"

    # ── array declaration ─────────────────────────────────────────────────────

    def _array_decl_prefix(self, info: _ArrayInfo) -> str:
        """Type + varname + dims, no trailing semicolon — used for plain decls and decl+assign."""
        c_type = _SOL_TYPE.get(info.elem_type, info.elem_type)
        array_dims = "".join(f"[{s}]" for s in _parse_size_list(info.array_size))
        if info.is_growing:
            return f"GrowingArray<{c_type}> {info.var}{array_dims}"
        block_dims = "".join(f"[{s}]" for s in _parse_size_list(info.block_size))
        return f"Array<{c_type}{block_dims}> {info.var}{array_dims}"

    def _array_decl(self, info: _ArrayInfo) -> str:
        return self._array_decl_prefix(info) + ";"

    # ── array binary-operation statement ─────────────────────────────────────

    def _is_array_binop(self, node: ast.AST) -> bool:
        """True when node is A+B, A*B, const*A, or A*const with at least one known array."""
        if not isinstance(node, ast.BinOp):
            return False
        if not isinstance(node.op, (ast.Add, ast.Mult)):
            return False
        left, right = node.left, node.right
        left_arr  = isinstance(left,  ast.Name) and left.id  in self._arrays
        right_arr = isinstance(right, ast.Name) and right.id in self._arrays
        left_const  = isinstance(left,  ast.Constant)
        right_const = isinstance(right, ast.Constant)
        return (left_arr and right_arr) or (left_const and right_arr) or (left_arr and right_const)

    def _array_binop_stmt(self, target: ast.AST, binop: ast.BinOp, ind: str) -> List[str]:
        var = _target_name(target)
        left, right = binop.left, binop.right
        op_str = "+" if isinstance(binop.op, ast.Add) else "*"

        left_name  = left.id  if isinstance(left,  ast.Name) and left.id  in self._arrays else None
        right_name = right.id if isinstance(right, ast.Name) and right.id in self._arrays else None

        if left_name and right_name:
            _validate_array_operands(self._arrays[left_name], self._arrays[right_name])

        ref_info = self._arrays[left_name or right_name]  # type: ignore[index]
        expr = f"{self._expr(left)} {op_str} {self._expr(right)}"

        if var in self._arrays:
            return [f"{ind}{var} = {expr};"]

        c_info = _ArrayInfo(
            line=ref_info.line,
            var=var,
            n_dims=ref_info.n_dims,
            n_block_dims=ref_info.n_block_dims,
            array_size=ref_info.array_size,
            block_size=ref_info.block_size,
            elem_type=ref_info.elem_type,
            is_growing=ref_info.is_growing,
        )
        self._arrays[var] = c_info
        return [f"{ind}{self._array_decl_prefix(c_info)} = {expr};"]


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
