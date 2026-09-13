from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


Item = Dict[str, Any]
ItemsIterable = Iterable[Item]


class AbstractPreProcessor(ABC):
    @abstractmethod
    def preprocess(self, items: ItemsIterable) -> ItemsIterable:
        raise NotImplementedError
