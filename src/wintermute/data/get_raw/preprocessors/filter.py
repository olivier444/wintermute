from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Dict, Iterator

from wintermute.data.get_raw.preprocessors.base import AbstractPreProcessor, Item, ItemsIterable
from wintermute.tools.logging import console_log


_PREDICATE_KEYS = {"equals", "in", "gt", "gte", "lt", "lte", "exists"}
_RUNTIME_PARAMS = {"seed", "language", "language_field"}


class FilterPreProcessor(AbstractPreProcessor):
    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        params = dict(params or {})
        unexpected = sorted(set(params).difference({"field", *_PREDICATE_KEYS, *_RUNTIME_PARAMS}))
        if unexpected:
            raise ValueError(f"filter does not support parameters: {', '.join(unexpected)}")

        field = params.get("field")
        if not isinstance(field, str) or not field.strip():
            raise ValueError("filter requires a non-empty 'field'")
        self.field = field.strip()

        predicate_keys = [key for key in _PREDICATE_KEYS if key in params]
        if len(predicate_keys) != 1:
            raise ValueError("filter requires exactly one predicate: equals, in, gt, gte, lt, lte, or exists")
        self.predicate = predicate_keys[0]
        self.expected = params[self.predicate]
        self._validate_predicate()

    def preprocess(self, items: ItemsIterable) -> ItemsIterable:
        def _iter_filtered() -> Iterator[Item]:
            seen = 0
            kept = 0
            for item in items:
                seen += 1
                if not self._matches(item):
                    continue
                kept += 1
                yield item
            excluded = seen - kept
            console_log(
                "filter-preproc",
                f"received {seen:_} records; included {kept:_}; excluded {excluded:_} "
                f"({self.field} {self.predicate})",
            )

        return _iter_filtered()

    def _validate_predicate(self) -> None:
        if self.predicate == "exists":
            if not isinstance(self.expected, bool):
                raise TypeError("filter.exists must be a boolean")
            return
        if self.predicate == "in":
            if not isinstance(self.expected, Sequence) or isinstance(self.expected, (str, bytes, bytearray)):
                raise TypeError("filter.in must be a sequence")
            return
        if self.predicate in {"gt", "gte", "lt", "lte"}:
            if not _is_number(self.expected):
                raise TypeError(f"filter.{self.predicate} must be a number")

    def _matches(self, item: Item) -> bool:
        found, value = _resolve_path(item, self.field)
        if self.predicate == "exists":
            return found is bool(self.expected)
        if not found:
            return False
        if self.predicate == "equals":
            return value == self.expected
        if self.predicate == "in":
            return value in self.expected
        if not _is_number(value):
            return False
        if self.predicate == "gt":
            return value > self.expected
        if self.predicate == "gte":
            return value >= self.expected
        if self.predicate == "lt":
            return value < self.expected
        if self.predicate == "lte":
            return value <= self.expected
        raise RuntimeError(f"Unsupported filter predicate: {self.predicate}")


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _resolve_path(item: Mapping[str, Any], field: str) -> tuple[bool, Any]:
    current: Any = item
    for token in field.split("."):
        normalized = token.strip()
        if not normalized:
            raise ValueError(f"filter field contains an empty path token: {field!r}")
        if not isinstance(current, Mapping) or normalized not in current:
            return False, None
        current = current[normalized]
    return True, current
