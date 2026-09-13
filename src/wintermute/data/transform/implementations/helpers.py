from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def require_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise RuntimeError(f"Unable to process field '{field}': expected str, got {type(value).__name__}")
    return value


def lookup_path(record: Mapping[str, Any], path: str) -> tuple[bool, Any]:
    value: Any = record
    for raw_token in path.split("."):
        token = raw_token.strip()
        if not token or not isinstance(value, Mapping) or token not in value:
            return False, None
        value = value[token]
    return True, value


def ensure_allowed_params(params: Mapping[str, Any], allowed_keys: set[str], *, kind: str) -> None:
    unexpected_keys = sorted(set(params.keys()).difference(allowed_keys))
    if unexpected_keys:
        raise ValueError(f"{kind} does not support parameters: {', '.join(unexpected_keys)}")
