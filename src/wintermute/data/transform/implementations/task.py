from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from wintermute.data.constants import TASK_LABELS
from wintermute.data.transform.implementations.helpers import ensure_allowed_params, lookup_path


@dataclass(frozen=True)
class TaskSpec:
    constant: str | None = None
    field: str | None = None
    mapping: Mapping[str, str | None] | None = None

    @classmethod
    def from_config(cls, raw_task: Any) -> TaskSpec | None:
        if raw_task is None:
            return None
        if isinstance(raw_task, str):
            if not raw_task.strip():
                raise ValueError("task must be a non-empty label")
            label = raw_task.strip()
            if label not in TASK_LABELS:
                raise ValueError(f"unknown task label '{label}'")
            return cls(constant=label)
        if not isinstance(raw_task, Mapping):
            raise TypeError("task must be a label string or a metadata mapping")

        ensure_allowed_params(raw_task, {"field", "mapping"}, kind="task")
        raw_field = raw_task.get("field")
        if not isinstance(raw_field, str) or not raw_field.strip():
            raise ValueError("task requires a non-empty 'field'")
        raw_mapping = raw_task.get("mapping")
        if raw_mapping is not None and not isinstance(raw_mapping, Mapping):
            raise TypeError("task.mapping must be a mapping")
        return cls(field=raw_field.strip(), mapping=cast(Mapping[str, str | None] | None, raw_mapping))

    def resolve(self, scope: Mapping[str, Any]) -> str | None:
        if self.constant is not None:
            return self.constant
        assert self.field is not None
        _, raw_value = lookup_path(scope, self.field)
        if raw_value is None:
            return None
        if not isinstance(raw_value, str):
            raise RuntimeError(
                f"Unable to resolve task from field '{self.field}': expected str, got {type(raw_value).__name__}"
            )
        source_value = raw_value.strip()
        if not source_value:
            return None
        if self.mapping is not None:
            if source_value not in self.mapping:
                raise RuntimeError(
                    f"Unable to resolve task from field '{self.field}': "
                    f"value '{source_value}' is missing from task.mapping"
                )
            label = self.mapping[source_value]
            if label is None:
                return None
            if not isinstance(label, str) or not label.strip():
                raise RuntimeError(f"Task mapping for '{source_value}' must resolve to a non-empty string")
            source_value = label.strip()
        if source_value not in TASK_LABELS:
            raise RuntimeError(f"unknown task label '{source_value}' resolved from field '{self.field}'")
        return source_value
