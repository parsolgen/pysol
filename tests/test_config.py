import json
import pytest
from pysol import load_config, get_config, pysol
from pysol.config import _reset_config


@pytest.fixture(autouse=True)
def reset():
    """Ensure each test starts with an empty config store."""
    _reset_config()
    yield
    _reset_config()


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_reads_values(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"key": "value", "num": 42}))
    load_config(str(cfg))
    assert get_config()["key"] == "value"
    assert get_config()["num"] == 42


def test_load_config_replaces_previous(tmp_path):
    cfg1 = tmp_path / "a.json"
    cfg1.write_text(json.dumps({"x": 1}))
    cfg2 = tmp_path / "b.json"
    cfg2.write_text(json.dumps({"y": 2}))
    load_config(str(cfg1))
    load_config(str(cfg2))
    config = get_config()
    assert "x" not in config
    assert config["y"] == 2


def test_load_config_nested(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"section": {"a": 1, "b": [1, 2, 3]}}))
    load_config(str(cfg))
    assert get_config()["section"]["b"] == [1, 2, 3]


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(OSError):
        load_config(str(tmp_path / "nonexistent.json"))


def test_load_config_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json {{{")
    with pytest.raises(json.JSONDecodeError):
        load_config(str(bad))


def test_load_config_non_object_raises(tmp_path):
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="JSON object"):
        load_config(str(arr))


# ---------------------------------------------------------------------------
# get_config
# ---------------------------------------------------------------------------

def test_get_config_empty_before_load():
    assert get_config() == {}


def test_get_config_returns_same_dict_reference():
    assert get_config() is get_config()


# ---------------------------------------------------------------------------
# Decorator access via __pysol_config__
# ---------------------------------------------------------------------------

def test_decorator_exposes_config(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"backend": "sol"}))
    load_config(str(cfg))

    @pysol
    def fn():
        pass

    assert fn.__pysol_config__["backend"] == "sol"


def test_decorator_config_empty_when_not_loaded():
    @pysol
    def fn():
        pass

    assert fn.__pysol_config__ == {}
