from __future__ import annotations

from typing import Any, Callable, Dict

from wintermute.data.get_raw.preprocessors.base import AbstractPreProcessor
from wintermute.data.get_raw.preprocessors.filter import FilterPreProcessor
from wintermute.data.get_raw.preprocessors.tree import TreePreProcessor


PreProcessorFactory = Callable[[Dict[str, Any]], AbstractPreProcessor]


_PREPROCESSOR_FACTORIES: Dict[str, PreProcessorFactory] = {
    "tree": lambda params: TreePreProcessor(params=params),
    "filter": lambda params: FilterPreProcessor(params=params),
}


def build_preprocessor(
    kind: str,
    params: Dict[str, Any] | None = None,
) -> AbstractPreProcessor:
    normalized_kind = kind.strip().lower()
    factory = _PREPROCESSOR_FACTORIES.get(normalized_kind)
    if factory is None:
        supported_kinds = ", ".join(sorted(_PREPROCESSOR_FACTORIES))
        raise ValueError(
            f"unknown preprocessor kind: {kind!r}. Supported kinds: {supported_kinds}"
        )
    return factory(dict(params or {}))
