from __future__ import annotations

import inspect
from typing import Callable, Dict, List

from pysol.array import ArrayElement, CSRBlock
from pysol.types import CSRBlockType, Type


_PY_TYPE: Dict[Type, type] = {
    Type.Int: int,
    Type.Float: float,
    Type.Double: float,
    Type.String: str,
    Type.Bool: bool,
}


class KernelArgument:
    """
    Describes one parameter of an imported kernel.

    Scalar argument:       KernelArgument(name, is_input, type)
    Array-element argument: KernelArgument(name, is_input, type, n_dims=N)  # N >= 1

    When n_dims >= 1 the argument is an ArrayElement; is_array_element is True
    and the Python annotation used in the generated signature is ArrayElement.
    The type field still carries the element's scalar type for documentation /
    code-generation purposes.
    """

    def __init__(self, name: str, is_input: bool, type: Type, n_dims: int = 0) -> None:
        if n_dims < 0:
            raise ValueError(f"n_dims must be >= 0, got {n_dims}")
        self.name = name
        self.is_input = is_input
        self.type = type
        self.n_dims = n_dims

    @property
    def is_array_element(self) -> bool:
        return self.n_dims > 0

    def _type_str(self) -> str:
        # self.type is a Type enum member, a struct name (plain str), or a
        # CSRBlockType (Type.CSRBlock(...)) — each needs its own rendering.
        t = self.type
        if isinstance(t, str):
            return t
        if hasattr(t, "value"):
            return t.value
        return repr(t)

    def __repr__(self) -> str:
        direction = "in" if self.is_input else "out"
        if self.is_array_element:
            return (
                f"KernelArgument({self.name!r}, {direction}, "
                f"{self._type_str()}, n_dims={self.n_dims})"
            )
        return f"KernelArgument({self.name!r}, {direction}, {self._type_str()})"


def import_kernel(name: str, args: List[KernelArgument]) -> Callable:
    """
    Dynamically create and return a callable named `name` whose parameters
    match `args`.  Python type annotations per argument:

        Scalar  Type.Int                 → int
        Scalar  Type.Float/Double        → float
        Scalar  Type.String              → str
        Scalar  Type.Bool                → bool
        Array element (n_dims>=1)        → ArrayElement
        Type.CSRBlock(...) (n_dims=0)    → CSRBlock

    The returned function validates argument types on every call and exposes
    the original KernelArgument metadata via __kernel_args__.
    """
    def _annotation(arg: KernelArgument) -> type:
        if arg.is_array_element:
            return ArrayElement
        if isinstance(arg.type, CSRBlockType):
            return CSRBlock
        return _PY_TYPE[arg.type]

    params = [
        inspect.Parameter(
            name=arg.name,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=_annotation(arg),
        )
        for arg in args
    ]
    sig = inspect.Signature(params)

    def kernel(*_a, **_kw):
        bound = sig.bind(*_a, **_kw)
        bound.apply_defaults()
        for param_name, value in bound.arguments.items():
            expected = sig.parameters[param_name].annotation
            if not isinstance(value, expected):
                raise TypeError(
                    f"{name}(): argument '{param_name}' expected "
                    f"{expected.__name__}, got {type(value).__name__}"
                )

    kernel.__name__ = name
    kernel.__qualname__ = name
    kernel.__signature__ = sig
    kernel.__kernel_args__: List[KernelArgument] = list(args)
    return kernel
