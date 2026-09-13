from __future__ import annotations

from wintermute.data.constants import TASK_PROBLEM_SOLVING
from wintermute.data.transform import RecordTransformConfig
from wintermute.data.transform.config_builders import chat_to_sft, prompt_completion, to_sft


def kaist_cot_collection_transform() -> RecordTransformConfig:
    return to_sft(
        prompt=[{"field": "source"}],
        scratchpad=[{"field": "rationale"}],
        completion=[{"field": "target"}],
    )


def open_thoughts_3_transform() -> RecordTransformConfig:
    return chat_to_sft(
        input_field="conversations",
        role_field="from",
        content_field="value",
        user_role="human",
        assistant_role="gpt",
        assistant_content_regex=r"\s*<think>\s*(.*?)\s*</think>\s*(.*)\s*\Z",
        task=TASK_PROBLEM_SOLVING,
    )


def xp3_en_problem_solving_transform() -> RecordTransformConfig:
    return prompt_completion(
        prompt_fields=["inputs"],
        completion_fields=["targets"],
        task=TASK_PROBLEM_SOLVING,
    )
