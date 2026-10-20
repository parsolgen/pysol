import inspect
import pytest
from pysol import Array, ArrayElement, CSRBlock, CSRMatrix, KernelArgument, Type, import_kernel


# ---------------------------------------------------------------------------
# KernelArgument
# ---------------------------------------------------------------------------

def test_kernel_argument_attributes():
    arg = KernelArgument("x", True, Type.Float)
    assert arg.name == "x"
    assert arg.is_input is True
    assert arg.type is Type.Float


def test_kernel_argument_output_flag():
    arg = KernelArgument("result", False, Type.Double)
    assert arg.is_input is False


def test_kernel_argument_repr():
    arg = KernelArgument("v", True, Type.Int)
    r = repr(arg)
    assert "v" in r
    assert "in" in r
    assert "Int" in r


# ---------------------------------------------------------------------------
# import_kernel — name and signature
# ---------------------------------------------------------------------------

def test_function_name():
    fn = import_kernel("my_kernel", [])
    assert fn.__name__ == "my_kernel"


def test_empty_args():
    fn = import_kernel("noop", [])
    sig = inspect.signature(fn)
    assert list(sig.parameters) == []
    fn()  # should not raise


def test_signature_param_names():
    args = [
        KernelArgument("a", True, Type.Int),
        KernelArgument("b", False, Type.Float),
        KernelArgument("c", True, Type.String),
    ]
    fn = import_kernel("k", args)
    sig = inspect.signature(fn)
    assert list(sig.parameters.keys()) == ["a", "b", "c"]


def test_signature_annotations_int():
    fn = import_kernel("k", [KernelArgument("n", True, Type.Int)])
    assert inspect.signature(fn).parameters["n"].annotation is int


def test_signature_annotations_float_and_double():
    args = [
        KernelArgument("f", True, Type.Float),
        KernelArgument("d", True, Type.Double),
    ]
    fn = import_kernel("k", args)
    sig = inspect.signature(fn)
    assert sig.parameters["f"].annotation is float
    assert sig.parameters["d"].annotation is float


def test_signature_annotations_string():
    fn = import_kernel("k", [KernelArgument("s", True, Type.String)])
    assert inspect.signature(fn).parameters["s"].annotation is str


# ---------------------------------------------------------------------------
# import_kernel — call-time type validation
# ---------------------------------------------------------------------------

def test_call_accepts_correct_types():
    fn = import_kernel("k", [
        KernelArgument("n", True, Type.Int),
        KernelArgument("x", True, Type.Double),
        KernelArgument("s", True, Type.String),
    ])
    fn(1, 2.0, "hello")  # should not raise


def test_call_rejects_wrong_type_int():
    fn = import_kernel("k", [KernelArgument("n", True, Type.Int)])
    with pytest.raises(TypeError, match="'n'"):
        fn(1.5)


def test_call_rejects_wrong_type_float():
    fn = import_kernel("k", [KernelArgument("x", True, Type.Float)])
    with pytest.raises(TypeError, match="'x'"):
        fn("oops")


def test_call_rejects_wrong_type_string():
    fn = import_kernel("k", [KernelArgument("s", True, Type.String)])
    with pytest.raises(TypeError, match="'s'"):
        fn(42)


def test_call_keyword_args():
    fn = import_kernel("k", [
        KernelArgument("a", True, Type.Int),
        KernelArgument("b", False, Type.Float),
    ])
    fn(b=3.14, a=1)  # should not raise


def test_call_missing_arg_raises():
    fn = import_kernel("k", [KernelArgument("x", True, Type.Int)])
    with pytest.raises(TypeError):
        fn()


def test_error_message_contains_kernel_name():
    fn = import_kernel("my_kernel", [KernelArgument("n", True, Type.Int)])
    with pytest.raises(TypeError, match="my_kernel"):
        fn("bad")


# ---------------------------------------------------------------------------
# import_kernel — metadata
# ---------------------------------------------------------------------------

def test_kernel_args_metadata():
    args = [
        KernelArgument("x", True, Type.Int),
        KernelArgument("y", False, Type.Float),
    ]
    fn = import_kernel("k", args)
    assert hasattr(fn, "__kernel_args__")
    assert len(fn.__kernel_args__) == 2
    assert fn.__kernel_args__[0].name == "x"
    assert fn.__kernel_args__[1].is_input is False


# ---------------------------------------------------------------------------
# KernelArgument — array element
# ---------------------------------------------------------------------------

def test_array_element_arg_attributes():
    arg = KernelArgument("blk", True, Type.Float, n_dims=3)
    assert arg.name == "blk"
    assert arg.is_input is True
    assert arg.type is Type.Float
    assert arg.n_dims == 3


def test_array_element_arg_is_array_element_true():
    arg = KernelArgument("blk", True, Type.Double, n_dims=2)
    assert arg.is_array_element is True


def test_scalar_arg_is_array_element_false():
    arg = KernelArgument("x", True, Type.Int)
    assert arg.is_array_element is False


def test_array_element_arg_repr():
    arg = KernelArgument("blk", False, Type.Int, n_dims=2)
    r = repr(arg)
    assert "blk" in r
    assert "out" in r
    assert "Int" in r
    assert "n_dims=2" in r


def test_scalar_arg_repr_has_no_n_dims():
    r = repr(KernelArgument("x", True, Type.Float))
    assert "n_dims" not in r


def test_array_element_arg_negative_n_dims_raises():
    with pytest.raises(ValueError):
        KernelArgument("blk", True, Type.Float, n_dims=-1)


# ---------------------------------------------------------------------------
# import_kernel — array element signature
# ---------------------------------------------------------------------------

def test_array_element_annotation_is_array_element():
    fn = import_kernel("k", [KernelArgument("blk", True, Type.Float, n_dims=2)])
    assert inspect.signature(fn).parameters["blk"].annotation is ArrayElement


def test_mixed_scalar_and_array_element_annotations():
    args = [
        KernelArgument("n", True, Type.Int),
        KernelArgument("blk", True, Type.Double, n_dims=3),
        KernelArgument("s", False, Type.String),
    ]
    fn = import_kernel("k", args)
    sig = inspect.signature(fn)
    assert sig.parameters["n"].annotation is int
    assert sig.parameters["blk"].annotation is ArrayElement
    assert sig.parameters["s"].annotation is str


def test_call_accepts_array_element():
    fn = import_kernel("k", [KernelArgument("blk", True, Type.Float, n_dims=2)])
    el = Array(1, 2, [4], [3, 3], Type.Float)[0, 0]
    fn(el)  # should not raise


def test_call_rejects_non_array_element():
    fn = import_kernel("k", [KernelArgument("blk", True, Type.Int, n_dims=1)])
    with pytest.raises(TypeError, match="'blk'"):
        fn(42)


def test_call_rejects_wrong_scalar_where_element_expected():
    fn = import_kernel("k", [KernelArgument("blk", True, Type.Double, n_dims=2)])
    with pytest.raises(TypeError, match="'blk'"):
        fn(3.14)


# ---------------------------------------------------------------------------
# KernelArgument / import_kernel — Type.CSRBlock(...)
# ---------------------------------------------------------------------------

def test_csr_block_kernel_argument_attributes():
    arg = KernelArgument("a", True, Type.CSRBlock(Type.Double))
    assert arg.name == "a"
    assert arg.n_dims == 0
    assert arg.is_array_element is False
    assert arg.type.element_type is Type.Double


def test_csr_block_kernel_argument_repr_does_not_crash():
    r = repr(KernelArgument("a", True, Type.CSRBlock(Type.Double)))
    assert "a" in r
    assert "in" in r
    assert "n_dims" not in r


def test_csr_block_annotation_is_csr_block():
    fn = import_kernel("foo", [KernelArgument("a", True, Type.CSRBlock(Type.Double))])
    assert inspect.signature(fn).parameters["a"].annotation is CSRBlock


def test_call_accepts_csr_block_argument():
    ia = Array(1, 1, [5], [1], Type.Int)
    ja = Array(1, 1, [6], [1], Type.Int)
    a = Array(1, 1, [6], [1], Type.Double)
    A = CSRMatrix([4, 4], [2, 2], 6, ia, ja, a, Type.Double)
    fn = import_kernel("foo", [KernelArgument("a", True, Type.CSRBlock(Type.Double))])
    fn(A[0, 0])  # should not raise


def test_call_rejects_non_csr_block_argument():
    fn = import_kernel("foo", [KernelArgument("a", True, Type.CSRBlock(Type.Double))])
    with pytest.raises(TypeError, match="'a'"):
        fn(42)


def test_struct_typed_argument_repr_does_not_crash():
    # A @pysol_struct class is passed as its plain string name, not a Type
    # enum member — repr() must not crash trying to access '.value' on it.
    r = repr(KernelArgument("s", True, "MyStruct"))
    assert "s" in r
    assert "MyStruct" in r
