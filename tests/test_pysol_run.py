"""Tests for pysol_run."""
import _ctypes
import ctypes
import os
import pytest

from pysol import pysol, pysol_run, SolLibraryNotFoundError, SolFunctionNotFoundError, SolInitError
from pysol.config import _reset_config
from pysol.decorator import _registry


@pytest.fixture(autouse=True)
def reset():
    _reset_config()
    yield
    _reset_config()


def _make_lib(init_rc=0, has_init=True, has_fini=True, has_fn=True, fn_result=0,
              calls=None):
    """Build a fake ctypes library with configurable behaviour.

    *calls* — if given, a list to which the names of called symbols are appended.
    """
    if calls is None:
        calls = []

    class _FakeLib:
        def __getattr__(self, attr_name):
            if attr_name == "init_pslib":
                if not has_init:
                    raise AttributeError("init_pslib not found")
                def _init(argc, argv):
                    calls.append("init_pslib")
                    return init_rc
                return _init

            if attr_name == "fini_pslib":
                if not has_fini:
                    raise AttributeError("fini_pslib not found")
                def _fini():
                    calls.append("fini_pslib")
                return _fini

            if not has_fn:
                raise AttributeError(f"function '{attr_name}' not found")

            def _fn():
                calls.append(attr_name)
                return fn_result
            return _fn

        # ctypes CDLL exposes _handle as an integer (OS library handle).
        _handle = 0

    return _FakeLib()


# ---------------------------------------------------------------------------
# Error: function not registered
# ---------------------------------------------------------------------------

def test_pysol_run_unregistered_raises():
    with pytest.raises(KeyError, match="registered"):
        pysol_run("ghost_function")


# ---------------------------------------------------------------------------
# Library loaded with RTLD_GLOBAL (needed for libraries, e.g. MKL, that
# dlopen their own plugins expecting globally-visible symbols - see
# examples/bicg, where RTLD_LOCAL caused an "undefined symbol" crash only
# reproducible via ctypes, never when running the equivalent program
# directly as a binary)
# ---------------------------------------------------------------------------

def test_pysol_run_uses_rtld_global(tmp_path, monkeypatch):
    @pysol
    def fn_global():
        pass

    received_mode = {}

    def _fake_cdll(_path, mode=None):
        received_mode["mode"] = mode
        return _make_lib()

    (tmp_path / "fn_global.so").write_bytes(b"")
    _registry["fn_global"].__pysol_sol_path__ = str(tmp_path / "fn_global.sol")
    monkeypatch.setattr(ctypes, "CDLL", _fake_cdll)
    monkeypatch.setattr(_ctypes, "dlclose", lambda _h: None)

    pysol_run("fn_global")

    assert received_mode["mode"] == ctypes.RTLD_GLOBAL

    del fn_global


# ---------------------------------------------------------------------------
# Error: .so file missing
# ---------------------------------------------------------------------------

def test_pysol_run_missing_so_raises():
    @pysol
    def fn_no_so():
        pass

    with pytest.raises(SolLibraryNotFoundError, match="fn_no_so.so"):
        pysol_run("fn_no_so")

    del fn_no_so


# ---------------------------------------------------------------------------
# Error: init_pslib symbol absent
# ---------------------------------------------------------------------------

def test_pysol_run_missing_init_raises(tmp_path, monkeypatch):
    @pysol
    def fn_no_sgc():
        pass

    (tmp_path / "fn_no_sgc.so").write_bytes(b"")
    _registry["fn_no_sgc"].__pysol_sol_path__ = str(tmp_path / "fn_no_sgc.sol")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, **_kw: _make_lib(has_init=False))

    with pytest.raises(SolFunctionNotFoundError, match="init_pslib"):
        pysol_run("fn_no_sgc")

    del fn_no_sgc


# ---------------------------------------------------------------------------
# Error: fini_pslib symbol absent
# ---------------------------------------------------------------------------

def test_pysol_run_missing_fini_raises(tmp_path, monkeypatch):
    @pysol
    def fn_no_so():
        pass

    (tmp_path / "fn_no_so.so").write_bytes(b"")
    _registry["fn_no_so"].__pysol_sol_path__ = str(tmp_path / "fn_no_so.sol")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, **_kw: _make_lib(has_fini=False))

    with pytest.raises(SolFunctionNotFoundError, match="fini_pslib"):
        pysol_run("fn_no_so")

    del fn_no_so


# ---------------------------------------------------------------------------
# Error: init_pslib returns non-zero
# ---------------------------------------------------------------------------

def test_pysol_run_init_nonzero_raises(tmp_path, monkeypatch):
    @pysol
    def fn_sgc():
        pass

    (tmp_path / "fn_sgc.so").write_bytes(b"")
    _registry["fn_sgc"].__pysol_sol_path__ = str(tmp_path / "fn_sgc.sol")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, **_kw: _make_lib(init_rc=1))

    with pytest.raises(SolInitError, match="1"):
        pysol_run("fn_sgc")

    del fn_sgc


def test_pysol_run_init_nonzero_code_in_message(tmp_path, monkeypatch):
    @pysol
    def fn_compiler():
        pass

    (tmp_path / "fn_compiler.so").write_bytes(b"")
    _registry["fn_compiler"].__pysol_sol_path__ = str(tmp_path / "fn_compiler.sol")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, **_kw: _make_lib(init_rc=42))

    with pytest.raises(SolInitError, match="42"):
        pysol_run("fn_compiler")

    del fn_compiler


# ---------------------------------------------------------------------------
# Error: main function symbol absent
# ---------------------------------------------------------------------------

def test_pysol_run_missing_symbol_raises(tmp_path, monkeypatch):
    @pysol
    def fn_bad_sym():
        pass

    (tmp_path / "fn_bad_sym.so").write_bytes(b"")
    _registry["fn_bad_sym"].__pysol_sol_path__ = str(tmp_path / "fn_bad_sym.sol")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, **_kw: _make_lib(has_fn=False))

    with pytest.raises(SolFunctionNotFoundError, match="fn_bad_sym"):
        pysol_run("fn_bad_sym")

    del fn_bad_sym


# ---------------------------------------------------------------------------
# Happy path — call order and return value
# ---------------------------------------------------------------------------

def test_pysol_run_calls_function(tmp_path, monkeypatch):
    @pysol
    def fn_ok():
        pass

    call_log = []
    (tmp_path / "fn_ok.so").write_bytes(b"")
    _registry["fn_ok"].__pysol_sol_path__ = str(tmp_path / "fn_ok.sol")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, **_kw: _make_lib(calls=call_log))
    monkeypatch.setattr(_ctypes, "dlclose", lambda _h: None)

    pysol_run("fn_ok")
    assert "fn_ok" in call_log

    del fn_ok


def test_pysol_run_returns_function_result(tmp_path, monkeypatch):
    @pysol
    def fn_retval():
        pass

    (tmp_path / "fn_retval.so").write_bytes(b"")
    _registry["fn_retval"].__pysol_sol_path__ = str(tmp_path / "fn_retval.sol")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, **_kw: _make_lib(fn_result=42))
    monkeypatch.setattr(_ctypes, "dlclose", lambda _h: None)

    result = pysol_run("fn_retval")
    assert result == 42

    del fn_retval


def test_pysol_run_call_order(tmp_path, monkeypatch):
    """init_pslib → main fn → fini_pslib, in that order."""
    @pysol
    def fn_paths():
        pass

    call_log = []
    (tmp_path / "fn_paths.so").write_bytes(b"")
    _registry["fn_paths"].__pysol_sol_path__ = str(tmp_path / "fn_paths.sol")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, **_kw: _make_lib(calls=call_log))
    monkeypatch.setattr(_ctypes, "dlclose", lambda _h: None)

    pysol_run("fn_paths")
    assert call_log == ["init_pslib", "fn_paths", "fini_pslib"]

    del fn_paths


# ---------------------------------------------------------------------------
# fini_pslib and dlclose are called even when the main function raises
# ---------------------------------------------------------------------------

def test_pysol_run_fini_called_on_main_exception(tmp_path, monkeypatch):
    @pysol
    def fn_two_steps():
        pass

    call_log = []
    closed = []

    def _bad_main():
        raise RuntimeError("main failed")

    class _BrokenLib:
        _handle = 0
        def __getattr__(self, attr_name):
            if attr_name == "init_pslib":
                def _init(argc, argv): return 0
                return _init
            if attr_name == "fini_pslib":
                def _fini(): call_log.append("fini_pslib")
                return _fini
            return _bad_main

    (tmp_path / "fn_two_steps.so").write_bytes(b"")
    _registry["fn_two_steps"].__pysol_sol_path__ = str(tmp_path / "fn_two_steps.sol")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, **_kw: _BrokenLib())
    monkeypatch.setattr(_ctypes, "dlclose", lambda _h: closed.append(True))

    with pytest.raises(RuntimeError, match="main failed"):
        pysol_run("fn_two_steps")

    assert "fini_pslib" in call_log
    assert closed == [True]

    del fn_two_steps


# ---------------------------------------------------------------------------
# Library is unloaded (dlclose called)
# ---------------------------------------------------------------------------

def test_pysol_run_unloads_library(tmp_path, monkeypatch):
    @pysol
    def fn_ok():
        pass

    closed_handles = []
    (tmp_path / "fn_ok.so").write_bytes(b"")
    _registry["fn_ok"].__pysol_sol_path__ = str(tmp_path / "fn_ok.sol")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, **_kw: _make_lib())
    monkeypatch.setattr(_ctypes, "dlclose", lambda h: closed_handles.append(h))

    pysol_run("fn_ok")
    assert len(closed_handles) == 1

    del fn_ok


# ---------------------------------------------------------------------------
# argv forwarding
# ---------------------------------------------------------------------------

def test_pysol_run_passes_argv(tmp_path, monkeypatch):
    """init_pslib receives argc/argv derived from sys.argv."""
    import sys

    @pysol
    def fn_paths():
        pass

    received = {}

    class _ArgvLib:
        _handle = 0
        def __getattr__(self, attr_name):
            if attr_name == "init_pslib":
                def _init(argc, argv):
                    received["argc"] = argc
                    received["argv0"] = argv[0]
                    return 0
                return _init
            if attr_name == "fini_pslib":
                return lambda: None
            return lambda: 0

    (tmp_path / "fn_paths.so").write_bytes(b"")
    _registry["fn_paths"].__pysol_sol_path__ = str(tmp_path / "fn_paths.sol")
    monkeypatch.setattr(ctypes, "CDLL", lambda _path, **_kw: _ArgvLib())
    monkeypatch.setattr(_ctypes, "dlclose", lambda _h: None)
    monkeypatch.setattr(sys, "argv", ["myprogram", "--flag"])

    pysol_run("fn_paths")

    assert received["argc"] == 2
    assert received["argv0"] == b"myprogram"

    del fn_paths
