from __future__ import annotations

import json
from typing import Any, Dict


_store: Dict[str, Any] = {}


def load_config(path: str) -> None:
    """
    Read the JSON file at *path* and replace the global config store.
    The top-level JSON value must be an object (dict).
    Raises ValueError for non-object JSON and propagates OSError / json.JSONDecodeError
    for missing or malformed files.
    """
    global _store
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file must contain a JSON object, got {type(data).__name__}"
        )
    _store = data


def get_config() -> Dict[str, Any]:
    """Return the currently loaded config (empty dict if load_config has not been called)."""
    return _store


def _reset_config() -> None:
    """Reset the global store to empty.  Intended for use in tests only."""
    global _store
    _store = {}
