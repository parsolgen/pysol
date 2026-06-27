from __future__ import annotations

import inspect
from typing import Callable, Dict, List

from pysol.types import Type


_PY_TYPE: Dict[Type, type] = {
    Type.Int: int,
    Type.Float: float,
    Type.Double: float,
    Type.String: str,
}


class KernelArgument:
    def __init__(self, name: str, is_input: bool, type: Type) -> None:
        self.name = name
        self.is_input = is_input
        self.type = type

    def __repr__(self) -> str:
        direction = "in" if self.is_input else "out"
        return f"KernelArgument({self.name!r}, {direction}, {self.type.value})"


def import_kernel(name: str, args: List[KernelArgument]) -> Callable:
    """
    Dynamically create and return a callable named `name` whose parameters
    match `args` with Python types derived from each argument's pysol.Type:

        Type.Int           → int
        Type.Float/Double  → float
        Type.String        → str

    The returned function validates argument types on every call and exposes
    the original KernelArgument metadata via __kernel_args__.
    """
    params = [
        inspect.Parameter(
            name=arg.name,
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=_PY_TYPE[arg.type],
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
