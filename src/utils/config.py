"""Small, deterministic YAML configuration loader with inheritance support."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``child`` over ``parent`` without mutating either input."""
    result = copy.deepcopy(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml_mapping(path: str | Path, _ancestors: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML mapping, optionally merging one or more relative ``extends`` files.

    Child mappings override parent mappings recursively.  ``extends`` is a loader
    directive and is not included in the returned effective configuration.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    if resolved in _ancestors:
        chain = " -> ".join(str(item) for item in (*_ancestors, resolved))
        raise ValueError(f"Circular YAML configuration inheritance: {chain}")

    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected a YAML mapping in {resolved}")

    parents = loaded.pop("extends", None)
    if parents is None:
        return loaded
    if isinstance(parents, (str, Path)):
        parents = [parents]
    if not isinstance(parents, list) or not all(isinstance(item, str) for item in parents):
        raise TypeError(f"'extends' in {resolved} must be a path or a list of paths")

    effective: dict[str, Any] = {}
    for parent in parents:
        effective = _merge(
            effective,
            load_yaml_mapping(resolved.parent / parent, (*_ancestors, resolved)),
        )
    return _merge(effective, loaded)
