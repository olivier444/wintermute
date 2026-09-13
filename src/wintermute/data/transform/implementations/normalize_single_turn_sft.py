from __future__ import annotations

from collections.abc import Iterable

from wintermute.data.constants import (
    FLD_GENERIC_COMPLETION,
    FLD_GENERIC_PROMPT,
    FLD_GENERIC_SCRATCHPAD,
    FLD_GENERIC_TASK,
)
from wintermute.data.record import DataRecord
from wintermute.data.transform.base import RecordTransform
from wintermute.data.transform.context import TransformContext


class NormalizeSingleTurnSftRecord(RecordTransform):
    """Normalize an unindexed one-turn SFT record to the multi-turn schema.

    It renames ``prompt``, ``completion``, and optional task or scratchpad
    fields to their ``.0`` forms.  Already multi-turn records and non-SFT
    records pass through unchanged.
    """

    def transform(
        self,
        record: DataRecord,
        *,
        context: TransformContext,
    ) -> Iterable[DataRecord]:
        if FLD_GENERIC_PROMPT not in record.fields or FLD_GENERIC_COMPLETION not in record.fields:
            yield record
            return

        fields = record.fields.copy()
        fields[f"{FLD_GENERIC_PROMPT}.0"] = fields.pop(FLD_GENERIC_PROMPT)
        fields[f"{FLD_GENERIC_COMPLETION}.0"] = fields.pop(FLD_GENERIC_COMPLETION)
        for field_name in (FLD_GENERIC_TASK, FLD_GENERIC_SCRATCHPAD):
            if field_name in fields:
                fields[f"{field_name}.0"] = fields.pop(field_name)

        yield DataRecord(record_id=record.record_id, fields=fields)
