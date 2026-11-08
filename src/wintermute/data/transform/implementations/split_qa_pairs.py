from __future__ import annotations

import re
from typing import Any

from wintermute.data.constants import (
    FLD_GENERIC_COMPLETION,
    FLD_GENERIC_PROMPT,
    FLD_GENERIC_TASK,
)
from wintermute.data.record import DataRecord
from wintermute.data.transform.base import RecordTransform
from wintermute.data.transform.context import TransformContext
from wintermute.data.transform.implementations.helpers import (
    ensure_allowed_params,
    require_string,
)
from wintermute.data.transform.implementations.task import TaskSpec


class SplitQaPairsRecordTransform(RecordTransform):
    """Split a document followed by several Q&A pairs into SFT records."""

    KIND = "split_qa_pairs"
    ALLOWED_PARAMS = {"input_field", "task"}
    _QUESTION_MARKER = re.compile(r"^[ \t]*Question:[ \t]*", re.MULTILINE)
    _ANSWER_MARKER = re.compile(r"(?<!\S)Answer:[ \t]*")

    def __init__(self, params: dict[str, Any]) -> None:
        ensure_allowed_params(params, self.ALLOWED_PARAMS, kind=self.KIND)
        self.input_field = str(params.get("input_field", "content"))
        self.task_spec = TaskSpec.from_config(params.get("task"))

    def transform(
        self,
        record: DataRecord,
        *,
        context: TransformContext,
    ) -> list[DataRecord]:
        content = require_string(record.fields, self.input_field)
        question_markers = list(self._QUESTION_MARKER.finditer(content))
        if not question_markers:
            return []

        passage = content[: question_markers[0].start()].strip()
        if not passage:
            return []
        task = self.task_spec.resolve(record.fields) if self.task_spec is not None else None

        outputs: list[DataRecord] = []
        for pair_index, question_marker in enumerate(question_markers):
            pair_end = (
                question_markers[pair_index + 1].start()
                if pair_index + 1 < len(question_markers)
                else len(content)
            )
            pair = content[question_marker.end() : pair_end]
            answer_marker = self._ANSWER_MARKER.search(pair)
            if answer_marker is None:
                continue

            question = pair[: answer_marker.start()].strip()
            answer = pair[answer_marker.end() :].strip()
            if not question or not answer:
                continue

            fields = {
                FLD_GENERIC_PROMPT: f"{passage}\n\nQuestion: {question}",
                FLD_GENERIC_COMPLETION: answer,
            }
            if task is not None:
                fields[FLD_GENERIC_TASK] = task
            outputs.append(
                DataRecord(
                    record_id=f"{record.record_id}.qa-{pair_index + 1}",
                    fields=fields,
                )
            )
        return outputs
