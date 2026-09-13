from __future__ import annotations

from typing import Any, List, Mapping, Optional

from wintermute.data.constants import FLD_GENERIC_TEXT
from wintermute.data.snapshot.config import SnapshotSourceConfig
from wintermute.data.transform import RecordTransformConfig
from wintermute.data.transform import config_builders as transform_configs

MAX_RECORD_SIZE_CHARS = int(1e9)


def src(
    weight: float,
    *,
    record_transform: RecordTransformConfig | None,
    oversampling: int = 1,
    min_chars: int = 200,
    max_chars: int = MAX_RECORD_SIZE_CHARS,
    excluded_snapshot_ids: Optional[List[str]] = None,
) -> SnapshotSourceConfig:
    return SnapshotSourceConfig(
        weight=weight,
        oversampling=oversampling,
        min_record_size_char=min_chars,
        max_record_size_char=max_chars,
        excluded_snapshot_ids=list(excluded_snapshot_ids or []),
        record_transform=record_transform,
    )


def text_src(
    weight: float,
    *,
    field: str = "text",
    oversampling: int = 1,
    min_chars: int = 200,
    max_chars: int = MAX_RECORD_SIZE_CHARS,
    excluded_snapshot_ids: Optional[List[str]] = None,
) -> SnapshotSourceConfig:
    return field_mapping_src(
        weight,
        fields={FLD_GENERIC_TEXT: [field]},
        oversampling=oversampling,
        min_chars=min_chars,
        max_chars=max_chars,
        excluded_snapshot_ids=excluded_snapshot_ids,
    )


def qa_pairs_sft_src(
    weight: float,
    *,
    field: str = "content",
    oversampling: int = 1,
    min_chars: int = 200,
    max_chars: int = MAX_RECORD_SIZE_CHARS,
    excluded_snapshot_ids: Optional[List[str]] = None,
) -> SnapshotSourceConfig:
    return src(
        weight,
        oversampling=oversampling,
        min_chars=min_chars,
        max_chars=max_chars,
        excluded_snapshot_ids=excluded_snapshot_ids,
        record_transform=transform_configs.split_qa_pairs(input_field=field),
    )


def field_mapping_src(
    weight: float,
    *,
    fields: dict[str, list[str]],
    task: str | Mapping[str, Any] | None = None,
    oversampling: int = 1,
    min_chars: int = 200,
    max_chars: int = MAX_RECORD_SIZE_CHARS,
    excluded_snapshot_ids: Optional[List[str]] = None,
) -> SnapshotSourceConfig:
    return src(
        weight,
        oversampling=oversampling,
        min_chars=min_chars,
        max_chars=max_chars,
        excluded_snapshot_ids=excluded_snapshot_ids,
        record_transform=transform_configs.field_mapping(fields=fields, task=task),
    )


def prompt_completion_src(
    weight: float,
    *,
    prompt_fields: list[str],
    completion_fields: list[str],
    system_prompt_fields: list[str] | None = None,
    task: str | Mapping[str, Any] | None = None,
    oversampling: int = 1,
    min_chars: int = 1,
    max_chars: int = MAX_RECORD_SIZE_CHARS,
    excluded_snapshot_ids: Optional[List[str]] = None,
) -> SnapshotSourceConfig:
    return src(
        weight,
        oversampling=oversampling,
        min_chars=min_chars,
        max_chars=max_chars,
        excluded_snapshot_ids=excluded_snapshot_ids,
        record_transform=transform_configs.prompt_completion(
            prompt_fields=prompt_fields,
            completion_fields=completion_fields,
            system_prompt_fields=system_prompt_fields,
            task=task,
        ),
    )


def to_sft_src(
    weight: float,
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
    oversampling: int = 1,
    min_chars: int = 1,
    max_chars: int = MAX_RECORD_SIZE_CHARS,
    excluded_snapshot_ids: Optional[List[str]] = None,
) -> SnapshotSourceConfig:
    return src(
        weight,
        oversampling=oversampling,
        min_chars=min_chars,
        max_chars=max_chars,
        excluded_snapshot_ids=excluded_snapshot_ids,
        record_transform=transform_configs.to_sft(
            prompt=prompt,
            completion=completion,
            task=task,
            scratchpad=scratchpad,
            system_prompt=system_prompt,
            context=context,
            items_field=items_field,
            emit_filter=emit_filter,
            context_filter=context_filter,
            joiner=joiner,
            context_joiner=context_joiner,
        ),
    )


def chat_to_sft_src(
    weight: float,
    *,
    input_field: str,
    oversampling: int = 1,
    min_chars: int = 1,
    max_chars: int = MAX_RECORD_SIZE_CHARS,
    excluded_snapshot_ids: Optional[List[str]] = None,
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
) -> SnapshotSourceConfig:
    return src(
        weight,
        oversampling=oversampling,
        min_chars=min_chars,
        max_chars=max_chars,
        excluded_snapshot_ids=excluded_snapshot_ids,
        record_transform=transform_configs.chat_to_sft(
            input_field=input_field,
            role_field=role_field,
            content_field=content_field,
            user_role=user_role,
            assistant_role=assistant_role,
            system_role=system_role,
            assistant_content_regex=assistant_content_regex,
            assistant_scratchpad_group=assistant_scratchpad_group,
            assistant_final_group=assistant_final_group,
            max_required_turn_chars=max_required_turn_chars,
            task=task,
        ),
    )
