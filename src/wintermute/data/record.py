from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DataRecord:
    record_id: str
    fields: dict[str, Any]

    def copy(self) -> DataRecord:
        return DataRecord(self.record_id, self.fields.copy())
