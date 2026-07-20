"""Tests for pysol_compile."""
import json
import os
import subprocess
import pytest

from pysol import load_config, pysol_compile, pysol, SolCompileError
from pysol.config import _reset_config
from pysol.decorator import _registry


@pytest.fixture(autouse=True)
def reset():
    _reset_config()
    yield
    _reset_config()


def _fake_run(calls):
    """Return a subprocess.run replacement that records calls and creates the -o output file."""
    def run(cmd, **_):
        calls.append(cmd)
        if "-o" in cmd:
            idx = cmd.index("-o")
            if idx + 1 < len(cmd):
                open(cmd[idx + 1], "w").close()
        return subprocess.CompletedProcess(cmd, returncode=0)
    return run


def _make_config(tmp_path, extra=None):
    data = {"sgc": "/usr/bin/sgc", "compiler": "gcc"}
    if extra:
        data.update(extra)
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(data))
    load_config(str(cfg))


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------

def test_pysol_registers_function():
    @pysol
    def my_fn():
        pass

    assert "my_fn" in _registry
    assert my_fn is _registry["my_fn"]


def test_pysol_stores_sol_path():
    @pysol
    def fn_with_sol():
        pass

    assert fn_with_sol.__pysol_sol_path__.endswith("fn_with_sol.sol")


# ---------------------------------------------------------------------------
# pysol_compile — error cases
# ---------------------------------------------------------------------------

def test_pysol_compile_unknown_function_raises():
    with pytest.raises(KeyError, match="registered"):
        pysol_compile("definitely_not_registered")


def test_pysol_compile_missing_sgc_raises():
    @pysol
    def fn_no_sgc():
        pass

    del fn_no_sgc
    with pytest.raises(KeyError, match="sgc"):
        pysol_compile("fn_no_sgc")


def test_pysol_compile_missing_compiler_raises(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"sgc": "/usr/bin/sgc"}))   # no 'compiler' key
    load_config(str(cfg))

    @pysol
    def fn_no_compiler():
        pass

    del fn_no_compiler
    with pytest.raises(KeyError, match="compiler"):
        pysol_compile("fn_no_compiler")


# ---------------------------------------------------------------------------
# pysol_compile — two-step invocation
# ---------------------------------------------------------------------------

def test_pysol_compile_runs_two_commands(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    _make_config(tmp_path)

    @pysol
    def fn_two_steps():
        pass

    del fn_two_steps
    r1, r2 = pysol_compile("fn_two_steps")

    assert r1.returncode == 0
    assert r2.returncode == 0
    assert len(calls) == 2


def test_pysol_compile_step1_sgc_command(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    _make_config(tmp_path)

    @pysol
    def fn_sgc():
        pass

    del fn_sgc
    pysol_compile("fn_sgc")

    sgc_cmd = calls[0]
    assert sgc_cmd[0] == "/usr/bin/sgc"
    assert sgc_cmd[1].endswith("fn_sgc.sol")
    assert sgc_cmd[2] == "-o"
    assert sgc_cmd[3].endswith("fn_sgc.cpp")


def test_pysol_compile_step2_compiler_command(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    _make_config(tmp_path)

    @pysol
    def fn_compiler():
        pass

    del fn_compiler
    pysol_compile("fn_compiler")

    cc_cmd = calls[1]
    assert cc_cmd[0] == "gcc"
    assert cc_cmd[1].endswith("fn_compiler.cpp")
    assert "-fPIC" in cc_cmd
    assert "-shared" in cc_cmd
    assert "-o" in cc_cmd
    assert cc_cmd[-1].endswith("fn_compiler.so")


def test_pysol_compile_options_list(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    _make_config(tmp_path, {"compile_options": ["-O2", "-march=native"]})

    @pysol
    def fn_opts_list():
        pass

    del fn_opts_list
    pysol_compile("fn_opts_list")

    cc_cmd = calls[1]
    assert "-O2" in cc_cmd
    assert "-march=native" in cc_cmd


def test_pysol_compile_options_string(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    _make_config(tmp_path, {"compile_options": "-O2 -march=native"})

    @pysol
    def fn_opts_str():
        pass

    del fn_opts_str
    pysol_compile("fn_opts_str")

    cc_cmd = calls[1]
    assert "-O2" in cc_cmd
    assert "-march=native" in cc_cmd


def test_pysol_compile_no_options(tmp_path, monkeypatch):
    """compile_options defaults to empty — -fPIC -shared -o are still present."""
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    _make_config(tmp_path)   # no compile_options key

    @pysol
    def fn_no_opts():
        pass

    del fn_no_opts
    pysol_compile("fn_no_opts")

    cc_cmd = calls[1]
    assert "-fPIC" in cc_cmd
    assert "-shared" in cc_cmd


def test_pysol_compile_outputs_next_to_sol(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    _make_config(tmp_path)

    @pysol
    def fn_paths():
        pass

    sol_dir = os.path.dirname(fn_paths.__pysol_sol_path__)
    del fn_paths
    pysol_compile("fn_paths")

    assert calls[0][3] == os.path.join(sol_dir, "fn_paths.cpp")   # step 1 output
    assert calls[1][-1] == os.path.join(sol_dir, "fn_paths.so")   # step 2 output


# ---------------------------------------------------------------------------
# pysol_compile — SolCompileError on compiler failure
# ---------------------------------------------------------------------------

def _failing_run(calls):
    """subprocess.run stub: step 1 succeeds, step 2 returns non-zero (no .so created)."""
    call_count = [0]
    def run(cmd, **_):
        call_count[0] += 1
        rc = 0 if call_count[0] == 1 else 1
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=rc)
    return run


def test_compiler_failure_raises_sol_compile_error(tmp_path, monkeypatch):
    """Non-zero compiler exit code with no .so produced raises SolCompileError."""
    calls = []
    monkeypatch.setattr(subprocess, "run", _failing_run(calls))
    _make_config(tmp_path)

    @pysol
    def fn_ok():
        pass

    del fn_ok
    with pytest.raises(SolCompileError, match="exit code 1"):
        pysol_compile("fn_ok")


def test_compiler_failure_message_includes_so_path(tmp_path, monkeypatch):
    """SolCompileError message names the expected .so output path."""
    calls = []
    monkeypatch.setattr(subprocess, "run", _failing_run(calls))
    _make_config(tmp_path)

    @pysol
    def fn_no_so():
        pass

    sol_dir = os.path.dirname(fn_no_so.__pysol_sol_path__)
    del fn_no_so
    with pytest.raises(SolCompileError, match="fn_no_so.so"):
        pysol_compile("fn_no_so")


def test_compiler_success_does_not_raise(tmp_path, monkeypatch):
    """Zero exit code + .so produced by the compiler does not raise."""
    calls = []
    monkeypatch.setattr(subprocess, "run", _fake_run(calls))
    _make_config(tmp_path)

    @pysol
    def fn_bad_sym():
        pass

    del fn_bad_sym
    # _fake_run returns 0 and creates the -o output file — no exception expected
    r1, r2 = pysol_compile("fn_bad_sym")
    assert r2.returncode == 0


def test_compiler_nonzero_always_raises_regardless_of_so(tmp_path, monkeypatch):
    """Non-zero compiler exit code raises SolCompileError even when .so was produced."""
    def run_nonzero_with_so(cmd, **_):
        so = next((a for a in cmd if a.endswith(".so")), None)
        if so:
            open(so, "w").close()
        return subprocess.CompletedProcess(cmd, returncode=1)

    monkeypatch.setattr(subprocess, "run", run_nonzero_with_so)
    _make_config(tmp_path)

    @pysol
    def fn_retval():
        pass

    del fn_retval
    with pytest.raises(SolCompileError, match="exit code 1"):
        pysol_compile("fn_retval")
