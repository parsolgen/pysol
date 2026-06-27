import inspect
import pytest
from pysol import KernelArgument, Type, import_kernel


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
