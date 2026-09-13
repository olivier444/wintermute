from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from wintermute.data.constants import (
    FLD_GENERIC_COMPLETION,
    FLD_GENERIC_PROMPT,
    FLD_GENERIC_SYSTEM_PROMPT,
)
from wintermute.data.transform.config import RecordTransformConfig


def fan_out(*transforms: RecordTransformConfig) -> RecordTransformConfig:
    return RecordTransformConfig(kind="fan_out", children=tuple(transforms))


def sequential(*transforms: RecordTransformConfig) -> RecordTransformConfig:
    return RecordTransformConfig(kind="sequential", children=tuple(transforms))


def field_mapping(
    *,
    fields: Mapping[str, list[str]],
    task: str | Mapping[str, Any] | None = None,
) -> RecordTransformConfig:
    params: dict[str, Any] = {
        "fields": {field: list(source_fields) for field, source_fields in fields.items()}
    }
    if task is not None:
        params["task"] = dict(task) if isinstance(task, Mapping) else task
    return RecordTransformConfig(kind="field_mapping", params=params)


def prompt_completion(
    *,
    prompt_fields: list[str],
    completion_fields: list[str],
    system_prompt_fields: list[str] | None = None,
    task: str | Mapping[str, Any] | None = None,
) -> RecordTransformConfig:
    fields = {
        FLD_GENERIC_PROMPT: list(prompt_fields),
        FLD_GENERIC_COMPLETION: list(completion_fields),
    }
    if system_prompt_fields is not None:
        fields[FLD_GENERIC_SYSTEM_PROMPT] = list(system_prompt_fields)
    return field_mapping(fields=fields, task=task)


def to_sft(
    *,
    prompt: list[Any],
    completion: list[Any],
    task: str | Mapping[str, Any] | None = None,
    scratchpad: list[Any] | None = None,
    system_prompt: list[Any] | None = None,
    context: list[Any] | None = None,
    items_field: str | None = None,
    emit_filter: Mapping[str, Any] | None = None,
    context_filter: Mapping[str, Any] | None = None,
    joiner: str | None = None,
    context_joiner: str | None = None,
) -> RecordTransformConfig:
    params: dict[str, Any] = {
        "prompt": list(prompt),
        "completion": list(completion),
    }
    if system_prompt is not None:
        params["system_prompt"] = list(system_prompt)
    if task is not None:
        params["task"] = dict(task) if isinstance(task, Mapping) else task
    if scratchpad is not None:
        params["scratchpad"] = list(scratchpad)
    if context is not None:
        params["context"] = list(context)
    if items_field is not None:
        params["items_field"] = items_field
    if emit_filter is not None:
        params["emit_filter"] = dict(emit_filter)
    if context_filter is not None:
        params["context_filter"] = dict(context_filter)
    if joiner is not None:
        params["joiner"] = joiner
    if context_joiner is not None:
        params["context_joiner"] = context_joiner
    return RecordTransformConfig(kind="to_sft", params=params)


def _chat_params(
    *,
    input_field: str,
    role_field: str | None = None,
    content_field: str | None = None,
    user_role: str | None = None,
    assistant_role: str | None = None,
    system_role: str | None = None,
    assistant_content_regex: str | None = None,
    assistant_scratchpad_group: int | None = None,
    assistant_final_group: int | None = None,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "input_field": input_field,
            "role_field": role_field,
            "content_field": content_field,
            "user_role": user_role,
            "assistant_role": assistant_role,
            "system_role": system_role,
            "assistant_content_regex": assistant_content_regex,
            "assistant_scratchpad_group": assistant_scratchpad_group,
            "assistant_final_group": assistant_final_group,
        }.items()
        if value is not None
    }


def chat_to_sft(
    *,
    input_field: str,
    role_field: str | None = None,
    content_field: str | None = None,
    user_role: str | None = None,
    assistant_role: str | None = None,
    system_role: str | None = None,
    assistant_content_regex: str | None = None,
    assistant_scratchpad_group: int | None = None,
    assistant_final_group: int | None = None,
    max_required_turn_chars: int | None = None,
    task: str | Mapping[str, Any] | None = None,
) -> RecordTransformConfig:
    params = _chat_params(
        input_field=input_field,
        role_field=role_field,
        content_field=content_field,
        user_role=user_role,
        assistant_role=assistant_role,
        system_role=system_role,
        assistant_content_regex=assistant_content_regex,
        assistant_scratchpad_group=assistant_scratchpad_group,
        assistant_final_group=assistant_final_group,
    )
    if max_required_turn_chars is not None:
        params["max_required_turn_chars"] = max_required_turn_chars
    if task is not None:
        params["task"] = dict(task) if isinstance(task, Mapping) else task
    return RecordTransformConfig(kind="chat_to_sft", params=params)


def arithmetic_template_augmentation(
    *,
    variants_per_record: int | None = None,
    keep_original: bool | None = None,
    relative_delta: float | None = None,
    absolute_delta: int | None = None,
    max_attempts_per_variant: int | None = None,
    min_value: int | None = None,
    numbers_field: str | None = None,
    equation_field: str | None = None,
    answer_field: str | None = None,
) -> RecordTransformConfig:
    params = {
        key: value
        for key, value in {
            "variants_per_record": variants_per_record,
            "keep_original": keep_original,
            "relative_delta": relative_delta,
            "absolute_delta": absolute_delta,
            "max_attempts_per_variant": max_attempts_per_variant,
            "min_value": min_value,
            "numbers_field": numbers_field,
            "equation_field": equation_field,
            "answer_field": answer_field,
        }.items()
        if value is not None
    }
    return RecordTransformConfig(kind="arithmetic_template_augmentation", params=params)


def equation_answer_arithmetic_augmentation(
    *,
    variants_per_record: int | None = None,
    keep_original: bool | None = None,
    relative_delta: float | None = None,
    absolute_delta: int | None = None,
    max_attempts_per_variant: int | None = None,
    min_value: int | None = None,
    question_field: str | None = None,
    answer_field: str | None = None,
) -> RecordTransformConfig:
    params = {
        key: value
        for key, value in {
            "variants_per_record": variants_per_record,
            "keep_original": keep_original,
            "relative_delta": relative_delta,
            "absolute_delta": absolute_delta,
            "max_attempts_per_variant": max_attempts_per_variant,
            "min_value": min_value,
            "question_field": question_field,
            "answer_field": answer_field,
        }.items()
        if value is not None
    }
    return RecordTransformConfig(kind="equation_answer_arithmetic_augmentation", params=params)


def split_qa_pairs(
    *,
    input_field: str = "content",
) -> RecordTransformConfig:
    return RecordTransformConfig(
        kind="split_qa_pairs",
        params={"input_field": input_field},
    )


def lookup_retrieval(*, min_items: int = 4, max_items: int = 10) -> RecordTransformConfig:
    return RecordTransformConfig(
        kind="lookup_retrieval",
        params={"min_items": min_items, "max_items": max_items},
    )
