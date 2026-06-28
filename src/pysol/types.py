from __future__ import annotations

from enum import Enum


class Type(Enum):
    Int = "Int"
    Float = "Float"
    Double = "Double"
    String = "String"
    Bool = "Bool"


class ArrayType:
    """Descriptor for the shape and element type of an Array."""

    def __init__(self, n_dims: int, n_block_dims: int, element_type: Type) -> None:
        if n_dims < 1:
            raise ValueError(f"n_dims must be >= 1, got {n_dims}")
        if n_block_dims < 1:
            raise ValueError(f"n_block_dims must be >= 1, got {n_block_dims}")
        self.n_dims = n_dims
        self.n_block_dims = n_block_dims
        self.element_type = element_type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayType):
            return NotImplemented
        return (
            self.n_dims == other.n_dims
            and self.n_block_dims == other.n_block_dims
            and self.element_type is other.element_type
        )

    def __hash__(self) -> int:
        return hash((self.n_dims, self.n_block_dims, self.element_type))

    def __repr__(self) -> str:
        return (
            f"ArrayType(n_dims={self.n_dims}, n_block_dims={self.n_block_dims}, "
            f"element_type={self.element_type})"
        )


class ArrayElementType:
    """Descriptor for the dimensionality and scalar type of an ArrayElement."""

    def __init__(self, n_dims: int, element_type: Type) -> None:
        if n_dims < 1:
            raise ValueError(f"n_dims must be >= 1, got {n_dims}")
        self.n_dims = n_dims
        self.element_type = element_type

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArrayElementType):
            return NotImplemented
        return self.n_dims == other.n_dims and self.element_type is other.element_type

    def __hash__(self) -> int:
        return hash((self.n_dims, self.element_type))

    def __repr__(self) -> str:
        return f"ArrayElementType(n_dims={self.n_dims}, element_type={self.element_type})"
