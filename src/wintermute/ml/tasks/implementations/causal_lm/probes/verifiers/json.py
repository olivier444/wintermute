from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping

from wintermute.ml.tasks.implementations.causal_lm.probes.verifiers.base import Verifier, extract_fenced_code


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


@dataclass(frozen=True)
class JsonFields(Verifier):
    expected_fields: Mapping[str, tuple[object, ...]]
    allow_comments: bool = True

    def __post_init__(self) -> None:
        if not self.expected_fields:
            raise ValueError("JsonFields requires at least one expected field")
        if any(not accepted_values for accepted_values in self.expected_fields.values()):
            raise ValueError("JsonFields expected fields require accepted values")
        normalized_fields = {field.casefold() for field in self.expected_fields}
        if len(normalized_fields) != len(self.expected_fields):
            raise ValueError("JsonFields expected fields must be unique case-insensitively")

    def score(self, completion: str) -> float:
        try:
            value = _parse_json(completion)
        except (json.JSONDecodeError, ValueError):
            if not self.allow_comments:
                return 0.0
            source = extract_fenced_code(completion, language="json")
            if source is None:
                return 0.0
            try:
                value = _parse_json(source)
            except (json.JSONDecodeError, ValueError):
                return 0.0

        if not isinstance(value, dict):
            return 0.0
        actual_fields = _casefold_json_object(value)
        if actual_fields is None:
            return 0.0
        valid_fields = sum(
            field.casefold() in actual_fields
            and any(
                _json_values_equal(
                    actual_fields[field.casefold()],
                    accepted_value,
                )
                for accepted_value in accepted_values
            )
            for field, accepted_values in self.expected_fields.items()
        )
        return valid_fields / len(self.expected_fields)


def _parse_json(source: str) -> object:
    return json.loads(
        source,
        parse_constant=_reject_nonstandard_json_constant,
    )


def _json_values_equal(actual: object, expected: object) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.casefold() == expected.casefold()
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return actual == expected
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict) and isinstance(expected, dict):
        actual_fields = _casefold_json_object(actual)
        expected_fields = _casefold_json_object(expected)
        return (
            actual_fields is not None
            and expected_fields is not None
            and actual_fields.keys() == expected_fields.keys()
            and all(
                _json_values_equal(actual_fields[key], expected_fields[key])
                for key in actual_fields
            )
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_values_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return actual == expected


def _casefold_json_object(value: dict[object, object]) -> dict[str, object] | None:
    if any(not isinstance(key, str) for key in value):
        return None
    normalized = {
        key.casefold(): nested_value
        for key, nested_value in value.items()
        if isinstance(key, str)
    }
    return normalized if len(normalized) == len(value) else None
