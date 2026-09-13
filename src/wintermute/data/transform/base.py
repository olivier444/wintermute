from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from wintermute.data.record import DataRecord
from wintermute.data.transform.context import TransformContext


class RecordTransform(ABC):
    @abstractmethod
    def transform(
        self,
        record: DataRecord,
        *,
        context: TransformContext,
    ) -> Iterable[DataRecord]: ...
