from __future__ import annotations

from collections.abc import Mapping

from wintermute.data.constants import (
    TASK_CLASSIFICATION,
    TASK_INFORMATION_EXTRACTION,
    TASK_KNOWLEDGE_QA,
    TASK_PROBLEM_SOLVING,
)
from wintermute.data.transform import RecordTransformConfig
from wintermute.data.transform.config_builders import fan_out, to_sft

from .prompts.common import (
    _CHOICE_HEADERS,
    _CHOICE_NUMBER_INSTRUCTIONS,
    _CHOICE_QA_INSTRUCTIONS,
    _GOAL_HEADERS,
    _PASSAGE_HEADERS,
    _QUESTION_HEADERS,
    _SENTENCE_HEADERS,
)
from .prompts.multiple_choice import (
    _BOOLQ_INSTRUCTIONS,
    _EN_BINARY_INSTRUCTIONS,
    _EN_FIRST_TEXT_HEADERS,
    _EN_PAWS_QUESTIONS,
    _EN_SECOND_TEXT_HEADERS,
    _EN_TARGET_WORD_HEADERS,
    _EN_WIC_QUESTIONS,
    _FR_BINARY_INSTRUCTIONS,
    _FR_CHOICE_LABEL_INSTRUCTIONS,
    _FR_CHOICE_NUMBER_INSTRUCTIONS,
    _FR_FIRST_TEXT_HEADERS,
    _FR_SECOND_TEXT_HEADERS,
    _FR_TARGET_WORD_HEADERS,
    _FR_WIC_QUESTIONS,
)


_CHOICE_RESPONSE_INSTRUCTIONS = {
    "option_number": _CHOICE_NUMBER_INSTRUCTIONS,
    "option_label": _CHOICE_QA_INSTRUCTIONS,
}

_CHOICE_QA_PROMPT = [
    {"random_text": _QUESTION_HEADERS},
    {"field": "question"},
    {"random_text": _CHOICE_HEADERS},
    {"field": "choices", "transform": "numbered_choices"},
]
_CHOICE_QA_COMPLETION = {
    "field": "answerKey",
    "transform": "choice_by_label",
    "choices_field": "choices",
}


def _multiple_choice_transform(
    *,
    prompt: list[dict[str, object]],
    completion: Mapping[str, object],
    task: str,
) -> RecordTransformConfig:
    return fan_out(
        *(
            to_sft(
                prompt=[*prompt, {"random_text": instructions}],
                completion=[{**completion, "response_format": response_format}],
                task=task,
            )
            for response_format, instructions in _CHOICE_RESPONSE_INSTRUCTIONS.items()
        )
    )


def _binary_classification_transform(
    *,
    prompt: list[dict[str, object]],
    label_field: str,
    true_label: str,
    false_label: str,
    natural_instructions: list[str],
    number_instructions: list[str],
    label_instructions: list[str],
) -> RecordTransformConfig:
    boolean_labels = {
        "true_label": true_label,
        "false_label": false_label,
    }
    variants = (
        (False, "option_label", natural_instructions),
        (True, "option_number", number_instructions),
        (True, "option_label", label_instructions),
    )
    return fan_out(
        *(
            to_sft(
                prompt=[
                    *prompt,
                    *([{"boolean_choices": boolean_labels}] if multiple_choice else []),
                    {"random_text": instructions},
                ],
                completion=[
                    {
                        "field": label_field,
                        "transform": "bool_yes_no",
                        **boolean_labels,
                        "response_format": response_format,
                    }
                ],
                task=TASK_CLASSIFICATION,
            )
            for multiple_choice, response_format, instructions in variants
        )
    )


def commonsense_qa_transform() -> RecordTransformConfig:
    return _multiple_choice_transform(
        prompt=_CHOICE_QA_PROMPT,
        completion=_CHOICE_QA_COMPLETION,
        task=TASK_KNOWLEDGE_QA,
    )


def winogrande_transform() -> RecordTransformConfig:
    return _multiple_choice_transform(
        prompt=[
            {"random_text": _SENTENCE_HEADERS},
            {"field": "sentence"},
            {"random_text": _CHOICE_HEADERS},
            {"fields": ["option1", "option2"], "transform": "numbered_choices"},
        ],
        completion={
            "field": "answer",
            "transform": "choice_by_index",
            "choices_fields": ["option1", "option2"],
            "one_based": True,
        },
        task=TASK_PROBLEM_SOLVING,
    )


def boolq_transform() -> RecordTransformConfig:
    base_prompt = [
        {"random_text": _PASSAGE_HEADERS},
        {"field": "passage"},
        {"random_text": _QUESTION_HEADERS},
        {"field": "question"},
    ]
    return fan_out(
        *(
            to_sft(
                prompt=[
                    *base_prompt,
                    *(
                        [
                            {"random_text": _CHOICE_HEADERS},
                            {"boolean_choices": True},
                        ]
                        if question_format == "multiple_choice"
                        else []
                    ),
                    {"random_text": instructions},
                ],
                completion=[
                    {
                        "field": "answer",
                        "transform": "bool_yes_no",
                        "response_format": response_format,
                    }
                ],
                task=TASK_INFORMATION_EXTRACTION,
            )
            for question_format, response_format, instructions in (
                ("natural", "option_label", _BOOLQ_INSTRUCTIONS),
                ("multiple_choice", "option_number", _CHOICE_NUMBER_INSTRUCTIONS),
                ("multiple_choice", "option_label", _CHOICE_QA_INSTRUCTIONS),
            )
        )
    )


def ai2_arc_transform() -> RecordTransformConfig:
    return _multiple_choice_transform(
        prompt=_CHOICE_QA_PROMPT,
        completion=_CHOICE_QA_COMPLETION,
        task=TASK_KNOWLEDGE_QA,
    )


def piqa_transform() -> RecordTransformConfig:
    return _multiple_choice_transform(
        prompt=[
            {"random_text": _GOAL_HEADERS},
            {"field": "goal"},
            {"random_text": _CHOICE_HEADERS},
            {"fields": ["sol1", "sol2"], "transform": "numbered_choices"},
        ],
        completion={
            "field": "label",
            "transform": "choice_by_index",
            "choices_fields": ["sol1", "sol2"],
            "one_based": False,
        },
        task=TASK_KNOWLEDGE_QA,
    )


def ag_news_transform() -> RecordTransformConfig:
    choices = [
        "World",
        "Sports",
        "Business",
        "Science and technology",
    ]
    prompt = [
        {"random_text": _PASSAGE_HEADERS},
        {"field": "text"},
        {"random_text": _CHOICE_HEADERS},
        {"choices": choices, "shuffle": True},
    ]
    return fan_out(
        *(
            to_sft(
                prompt=[*prompt, {"random_text": instructions}],
                completion=[
                    {
                        "field": "label",
                        "transform": "choice_by_index",
                        "choices": choices,
                        "one_based": False,
                        "shuffle": True,
                        "response_format": response_format,
                    }
                ],
                task=TASK_CLASSIFICATION,
            )
            for response_format, instructions in (
                ("option_number", _CHOICE_NUMBER_INSTRUCTIONS),
                ("option_label", _CHOICE_QA_INSTRUCTIONS),
            )
        )
    )


def paws_transform() -> RecordTransformConfig:
    return _binary_classification_transform(
        prompt=[
            {"random_text": _EN_FIRST_TEXT_HEADERS},
            {"field": "sentence1"},
            {"random_text": _EN_SECOND_TEXT_HEADERS},
            {"field": "sentence2"},
            {"random_text": _EN_PAWS_QUESTIONS},
        ],
        label_field="label",
        true_label="yes",
        false_label="no",
        natural_instructions=_EN_BINARY_INSTRUCTIONS,
        number_instructions=_CHOICE_NUMBER_INSTRUCTIONS,
        label_instructions=_CHOICE_QA_INSTRUCTIONS,
    )


def wic_transform() -> RecordTransformConfig:
    return _binary_classification_transform(
        prompt=[
            {"random_text": _EN_TARGET_WORD_HEADERS},
            {"field": "word"},
            {"random_text": _EN_FIRST_TEXT_HEADERS},
            {"field": "sentence1"},
            {"random_text": _EN_SECOND_TEXT_HEADERS},
            {"field": "sentence2"},
            {"random_text": _EN_WIC_QUESTIONS},
        ],
        label_field="label",
        true_label="yes",
        false_label="no",
        natural_instructions=_EN_BINARY_INSTRUCTIONS,
        number_instructions=_CHOICE_NUMBER_INSTRUCTIONS,
        label_instructions=_CHOICE_QA_INSTRUCTIONS,
    )


def xlwic_fr_transform() -> RecordTransformConfig:
    return _binary_classification_transform(
        prompt=[
            {"random_text": _FR_TARGET_WORD_HEADERS},
            {"field": "target_word"},
            {"random_text": _FR_FIRST_TEXT_HEADERS},
            {"field": "context_1"},
            {"random_text": _FR_SECOND_TEXT_HEADERS},
            {"field": "context_2"},
            {"random_text": _FR_WIC_QUESTIONS},
        ],
        label_field="label",
        true_label="oui",
        false_label="non",
        natural_instructions=_FR_BINARY_INSTRUCTIONS,
        number_instructions=_FR_CHOICE_NUMBER_INSTRUCTIONS,
        label_instructions=_FR_CHOICE_LABEL_INSTRUCTIONS,
    )
