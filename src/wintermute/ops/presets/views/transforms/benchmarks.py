from __future__ import annotations

from wintermute.data.constants import (
    TASK_INFORMATION_EXTRACTION,
    TASK_PROBLEM_SOLVING,
)
from wintermute.data.transform import RecordTransformConfig
from wintermute.data.transform.config_builders import (
    arithmetic_template_augmentation,
    equation_answer_arithmetic_augmentation,
    sequential,
    to_sft,
)

from .prompts.benchmarks import (
    _MATH_REASONING_INSTRUCTIONS,
    _NUMERIC_ANSWER_INSTRUCTIONS,
    _SHORT_FACT_INSTRUCTIONS,
)
from .prompts.common import _QUESTION_HEADERS


def babi_qa_transform() -> RecordTransformConfig:
    return to_sft(
        items_field="story",
        context=[{"field": "text"}],
        context_filter={"field": "type", "equals": 0},
        prompt=[
            {"random_text": _QUESTION_HEADERS},
            {"field": "text"},
            {"random_text": _SHORT_FACT_INSTRUCTIONS},
        ],
        completion=[{"field": "answer"}],
        emit_filter={"field": "type", "equals": 1},
        context_joiner="\n",
        task=TASK_INFORMATION_EXTRACTION,
    )


def gsm8k_transform() -> RecordTransformConfig:
    return to_sft(
        prompt=[
            {"random_text": _MATH_REASONING_INSTRUCTIONS},
            {"field": "question"},
        ],
        completion=[{"field": "answer", "transform": "math_reasoning"}],
        task=TASK_PROBLEM_SOLVING,
    )


def asdiv_a_svamp_transform() -> RecordTransformConfig:
    return sequential(
        arithmetic_template_augmentation(
            variants_per_record=100,
            keep_original=True,
            relative_delta=3,
            absolute_delta=15,
            max_attempts_per_variant=1024,
            min_value=1,
        ),
        to_sft(
            prompt=[
                {"random_text": _NUMERIC_ANSWER_INSTRUCTIONS},
                {
                    "field": "Question",
                    "transform": "placeholder_substitute",
                    "values_field": "Numbers",
                },
            ],
            completion=[{"field": "Answer", "transform": "normalize_number"}],
            task=TASK_PROBLEM_SOLVING,
        ),
    )


def mawps_transform() -> RecordTransformConfig:
    return sequential(
        equation_answer_arithmetic_augmentation(
            variants_per_record=100,
            keep_original=True,
            relative_delta=3,
            absolute_delta=15,
            max_attempts_per_variant=1024,
            min_value=1,
        ),
        to_sft(
            prompt=[
                {"random_text": _MATH_REASONING_INSTRUCTIONS},
                {"field": "question"},
            ],
            completion=[{"field": "answer", "transform": "math_reasoning"}],
            task=TASK_PROBLEM_SOLVING,
        ),
    )
