from __future__ import annotations

from wintermute.data.constants import (
    TASK_CLASSIFICATION,
    TASK_GENERATION,
    TASK_INFORMATION_EXTRACTION,
    TASK_REWRITING,
    TASK_SUMMARIZATION,
)
from wintermute.data.transform import RecordTransformConfig
from wintermute.data.transform.config_builders import prompt_completion, to_sft

from .prompts.text_processing import (
    _EN_SUMMARIZATION_INSTRUCTIONS,
    _FR_HEADLINE_INSTRUCTIONS,
    _FR_SUMMARIZATION_INSTRUCTIONS,
)


def xp3_en_summary_transform() -> RecordTransformConfig:
    return prompt_completion(
        prompt_fields=["inputs"],
        completion_fields=["targets"],
        task=TASK_SUMMARIZATION,
    )


def xp3_en_qa_transform() -> RecordTransformConfig:
    return prompt_completion(
        prompt_fields=["inputs"],
        completion_fields=["targets"],
        task=TASK_INFORMATION_EXTRACTION,
    )


def xp3_en_paws_rewrite_transform() -> RecordTransformConfig:
    return prompt_completion(
        prompt_fields=["inputs"],
        completion_fields=["targets"],
        task=TASK_REWRITING,
    )


def xp3_en_generate_transform() -> RecordTransformConfig:
    return prompt_completion(
        prompt_fields=["inputs"],
        completion_fields=["targets"],
        task=TASK_GENERATION,
    )


def xp3_fr_summary_transform() -> RecordTransformConfig:
    return prompt_completion(
        prompt_fields=["inputs"],
        completion_fields=["targets"],
        task=TASK_SUMMARIZATION,
    )


def xp3_fr_generate_transform() -> RecordTransformConfig:
    return prompt_completion(
        prompt_fields=["inputs"],
        completion_fields=["targets"],
        task=TASK_GENERATION,
    )


def xsum_transform() -> RecordTransformConfig:
    return to_sft(
        prompt=[
            {"random_text": _EN_SUMMARIZATION_INSTRUCTIONS},
            {"field": "document"},
        ],
        completion=[{"field": "summary"}],
        task=TASK_SUMMARIZATION,
    )


def orange_sum_abstract_transform() -> RecordTransformConfig:
    return to_sft(
        prompt=[
            {"random_text": _FR_SUMMARIZATION_INSTRUCTIONS},
            {"field": "text"},
        ],
        completion=[{"field": "summary"}],
        task=TASK_SUMMARIZATION,
    )


def orange_sum_title_transform() -> RecordTransformConfig:
    return to_sft(
        prompt=[
            {"random_text": _FR_HEADLINE_INSTRUCTIONS},
            {"field": "text"},
        ],
        completion=[{"field": "summary"}],
        task=TASK_GENERATION,
    )


def xp3_en_general_classification_transform() -> RecordTransformConfig:
    return prompt_completion(
        prompt_fields=["inputs"],
        completion_fields=["targets"],
        task=TASK_CLASSIFICATION,
    )


def xp3_en_general_generation_transform() -> RecordTransformConfig:
    return prompt_completion(
        prompt_fields=["inputs"],
        completion_fields=["targets"],
        task=TASK_GENERATION,
    )


def xp3_en_information_extraction_transform() -> RecordTransformConfig:
    return prompt_completion(
        prompt_fields=["inputs"],
        completion_fields=["targets"],
        task=TASK_INFORMATION_EXTRACTION,
    )
