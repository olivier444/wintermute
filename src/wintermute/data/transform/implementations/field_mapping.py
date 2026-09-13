from __future__ import annotations

from typing import Any, Dict, List

from wintermute.data.constants import FLD_GENERIC_TASK
from wintermute.data.record import DataRecord
from wintermute.data.transform.base import RecordTransform
from wintermute.data.transform.context import TransformContext
from wintermute.data.transform.implementations.helpers import require_string
from wintermute.data.transform.implementations.task import TaskSpec


class FieldMappingRecordTransform(RecordTransform):
    """Project selected source fields into a new, compact record schema.

    Each configured output field joins its non-empty source fields with blank
    lines.  An optional task specification adds the canonical task label;
    fields absent from the mapping are deliberately discarded.
    """

    KIND = "field_mapping"

    def __init__(self, fields: Dict[str, List[str]], task: Any = None):
        self.fields = fields
        self.task_spec = TaskSpec.from_config(task)

    def transform(self, record: DataRecord, *, context: TransformContext) -> List[DataRecord]:
        result: Dict[str, str] = {}
        for output_field, input_fields in self.fields.items():
            values = [
                value
                for input_field in input_fields
                if (value := require_string(record.fields, input_field))
            ]
            result[output_field] = "\n\n".join(values)
        if self.task_spec is not None:
            task = self.task_spec.resolve(record.fields)
            if task is not None:
                result[FLD_GENERIC_TASK] = task
        return [DataRecord(record_id=record.record_id, fields=result)]
