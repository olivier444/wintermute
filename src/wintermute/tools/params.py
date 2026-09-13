from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Tuple


def get_int(params: Mapping[str, Any], key: str, default: int) -> int:
    value = params.get(key, default)
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"Expected int-compatible value for '{key}', got {type(value)!r}.")
    return int(value)


def get_optional_int(params: Mapping[str, Any], key: str) -> int | None:
    if key not in params or params[key] is None:
        return None
    value = params[key]
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"Expected int-compatible value for '{key}', got {type(value)!r}.")
    return int(value)


def get_float(params: Mapping[str, Any], key: str, default: float) -> float:
    value = params.get(key, default)
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"Expected float-compatible value for '{key}', got {type(value)!r}.")
    return float(value)


def get_optional_float(params: Mapping[str, Any], key: str) -> float | None:
    if key not in params or params[key] is None:
        return None
    value = params[key]
    if not isinstance(value, (int, float, str)):
        raise TypeError(f"Expected float-compatible value for '{key}', got {type(value)!r}.")
    return float(value)


def get_bool(params: Mapping[str, Any], key: str, default: bool) -> bool:
    value = params.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def get_optional_bool(params: Mapping[str, Any], key: str) -> bool | None:
    if key not in params or params[key] is None:
        return None
    return get_bool(params, key, False)


def get_str(params: Mapping[str, Any], key: str, default: str) -> str:
    value = params.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"Expected string value for '{key}', got {type(value)!r}.")
    return value


def get_optional_str(params: Mapping[str, Any], key: str) -> str | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Expected string value for '{key}', got {type(value)!r}.")
    return value


def get_str_list(params: Mapping[str, Any], key: str, default: Sequence[str]) -> List[str]:
    value = params.get(key)
    if value is None:
        return [str(item) for item in default]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Expected a sequence for '{key}', got {type(value)!r}.")
    return [str(item) for item in value]


def get_optional_mapping(params: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected a mapping for '{key}', got {type(value)!r}.")
    return value


def get_optional_list(params: Mapping[str, Any], key: str) -> list[Any] | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"Expected a list for '{key}', got {type(value)!r}.")
    return value


def parse_optional_positive_int(value: Any, *, field_name: str) -> Optional[int]:
    if value is None:
        return None

    result = int(value)
    if result <= 0:
        raise ValueError(f"Invalid {field_name}={result}.")
    return result


def get_optional_positive_int(params: Mapping[str, Any], key: str, *, field_name: str) -> Optional[int]:
    return parse_optional_positive_int(params.get(key), field_name=field_name)


def normalize_kind_params_payload(
    payload: str | Mapping[str, Any],
    *,
    context: str,
    kind_field: str = "kind",
    params_field: str = "params",
) -> Tuple[str, Dict[str, Any]]:
    if isinstance(payload, str):
        kind = payload.strip()
        if not kind:
            raise ValueError(f"{context}.{kind_field} must be a non-empty string.")
        return kind, {}

    data = dict(payload)
    kind = get_str(data, kind_field, "").strip()
    if not kind:
        raise ValueError(f"{context}.{kind_field} must be a non-empty string.")

    raw_params = data.get(params_field, {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, Mapping):
        raise TypeError(f"{context}.{params_field} must be a mapping when provided.")

    data.pop(kind_field, None)
    data.pop(params_field, None)
    params = dict(raw_params)
    params.update(data)
    return kind, params
