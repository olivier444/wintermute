from __future__ import annotations

from wintermute.data.constants import (
    FLD_GENERIC_COMPLETION,
    FLD_GENERIC_PROMPT,
    TASK_KNOWLEDGE_QA,
)
from wintermute.data.transform import RecordTransformConfig
from wintermute.data.transform.config_builders import field_mapping


def xp3_en_knowledge_qa_transform() -> RecordTransformConfig:
    return field_mapping(
        fields={
            FLD_GENERIC_PROMPT: ["inputs"],
            FLD_GENERIC_COMPLETION: ["targets"],
        },
        task=TASK_KNOWLEDGE_QA,
    )
