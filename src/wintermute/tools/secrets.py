from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SecretEntry:
    id: str
    value: str


def load_secret_entries(path: str | Path) -> dict[str, SecretEntry]:
    secret_path = Path(path).expanduser()
    return _load_secret_entries_cached(str(secret_path))


@lru_cache(maxsize=None)
def _load_secret_entries_cached(secret_path_str: str) -> dict[str, SecretEntry]:
    secret_path = Path(secret_path_str)
    raw_payload = _decrypt_secret_file(secret_path)
    payload = json.loads(raw_payload)
    if not isinstance(payload, list):
        raise TypeError(f"Secret file '{secret_path}' must contain a JSON array.")

    entries: dict[str, SecretEntry] = {}
    for index, item in enumerate(payload):
        context = f"{secret_path}[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{context} must be an object.")

        entry_id = item.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise ValueError(f"{context}.id must be a non-empty string.")

        value = _coerce_secret_value(item, context=context)
        key = entry_id.strip()
        if key in entries:
            raise ValueError(f"Duplicate secret id '{key}' in '{secret_path}'.")
        entries[key] = SecretEntry(id=key, value=value)

    return entries


def resolve_secret_value(secret_id: str, *, path: str | Path) -> str:
    clean_secret_id = secret_id.strip()
    if not clean_secret_id:
        raise ValueError("Secret id must be a non-empty string.")

    entries = load_secret_entries(path)
    try:
        return entries[clean_secret_id].value
    except KeyError as exc:
        raise ValueError(f"No secret entry found for id '{clean_secret_id}'.") from exc


def _decrypt_secret_file(path: Path) -> str:
    proc = subprocess.run(
        ["gpg", "--quiet", "--batch", "--decrypt", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or "unknown gpg error"
        raise RuntimeError(f"Unable to decrypt secret file '{path}': {message}")
    return proc.stdout


def _coerce_secret_value(item: dict[str, Any], *, context: str) -> str:
    for field_name in ("value", "password"):
        raw_value = item.get(field_name)
        if raw_value is None:
            continue
        if not isinstance(raw_value, str):
            raise TypeError(f"{context}.{field_name} must be a string.")
        clean_value = raw_value.strip()
        if not clean_value:
            raise ValueError(f"{context}.{field_name} must be non-empty.")
        return clean_value

    raise ValueError(f"{context} must define either 'value' or 'password'.")
