from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Dict, List, cast

from wintermute.data.constants import (
    FLD_GENERIC_COMPLETION,
    FLD_GENERIC_PROMPT,
    FLD_GENERIC_SCRATCHPAD,
    FLD_GENERIC_SYSTEM_PROMPT,
    FLD_GENERIC_TASK,
)
from wintermute.data.record import DataRecord
from wintermute.data.transform.base import RecordTransform
from wintermute.data.transform.context import TransformContext
from wintermute.data.transform.implementations.helpers import ensure_allowed_params
from wintermute.data.transform.implementations.task import TaskSpec
from wintermute.tools.logging import console_log
from wintermute.tools.params import get_int, get_optional_positive_int, get_optional_str, get_str


_CHAT_PARAM_DEFAULTS = {
    "role_field": "role",
    "content_field": "content",
    "user_role": "user",
    "assistant_role": "assistant",
    "system_role": "system",
}
_CHAT_ALLOWED_PARAMS = {
    "input_field",
    "assistant_content_regex",
    "assistant_scratchpad_group",
    "assistant_final_group",
} | set(_CHAT_PARAM_DEFAULTS)


class ChatToSftRecordTransform(RecordTransform):
    """Convert one role-based conversation into canonical multi-turn SFT data.

    System messages become the system prompt; each user/assistant exchange
    becomes a numbered prompt/completion pair.  An optional regex separates an
    assistant scratchpad from its final completion.
    """

    KIND = "chat_to_sft"
    ALLOWED_PARAMS = _CHAT_ALLOWED_PARAMS | {"max_required_turn_chars", "task"}

    def __init__(self, params: Mapping[str, Any]):
        ensure_allowed_params(params, self.ALLOWED_PARAMS, kind=self.KIND)
        input_field = get_optional_str(params, "input_field")
        if input_field is None:
            raise ValueError(f"{self.KIND} requires 'input_field'")
        self.input_field = input_field
        self.role_field = get_str(params, "role_field", _CHAT_PARAM_DEFAULTS["role_field"])
        self.content_field = get_str(params, "content_field", _CHAT_PARAM_DEFAULTS["content_field"])
        self.user_role = get_str(params, "user_role", _CHAT_PARAM_DEFAULTS["user_role"])
        self.assistant_role = get_str(params, "assistant_role", _CHAT_PARAM_DEFAULTS["assistant_role"])
        self.system_role = get_str(params, "system_role", _CHAT_PARAM_DEFAULTS["system_role"])
        self.assistant_content_regex = get_optional_str(params, "assistant_content_regex")
        self.assistant_scratchpad_group = get_int(params, "assistant_scratchpad_group", 1)
        self.assistant_final_group = get_int(params, "assistant_final_group", 2)
        self.max_required_turn_chars = get_optional_positive_int(
            params,
            "max_required_turn_chars",
            field_name="chat_to_sft.max_required_turn_chars",
        )
        self.task_spec = TaskSpec.from_config(params.get("task"))

    def transform(self, record: DataRecord, *, context: TransformContext) -> List[DataRecord]:
        uid = record.record_id
        task = self.task_spec.resolve(record.fields) if self.task_spec is not None else None
        conversation = record.fields.get(self.input_field)
        if not isinstance(conversation, list):
            raise RuntimeError(
                f"Unable to process chat field '{self.input_field}': expected list, got {type(conversation).__name__}"
            )

        output_fields: Dict[str, str] = {}
        system_parts: List[str] = []
        prompt_parts: List[str] = []
        scratchpad_parts: List[str] = []
        completion_parts: List[str] = []
        current_index = 0
        result: List[DataRecord] = []

        def _flush(index: int, *, last: bool) -> int:
            next_index = index
            prompt = "\n\n".join(prompt_parts)
            if completion_parts and prompt.strip():
                completion = "\n\n".join(completion_parts)
                required_turn_chars = len(prompt) + len(completion)
                if (
                    self.max_required_turn_chars is None
                    or required_turn_chars <= self.max_required_turn_chars
                ):
                    if not output_fields and system_parts:
                        system_prompt = "\n\n".join(system_parts)
                        if system_prompt.strip():
                            output_fields[FLD_GENERIC_SYSTEM_PROMPT] = system_prompt
                    output_fields[f"{FLD_GENERIC_PROMPT}.{index}"] = prompt
                    if task is not None:
                        output_fields[f"{FLD_GENERIC_TASK}.{index}"] = task
                    if scratchpad_parts:
                        output_fields[f"{FLD_GENERIC_SCRATCHPAD}.{index}"] = "\n\n".join(scratchpad_parts)
                    output_fields[f"{FLD_GENERIC_COMPLETION}.{index}"] = completion
                    next_index += 1
            if last and output_fields:
                result.append(DataRecord(record_id=uid, fields=output_fields))
            prompt_parts.clear()
            scratchpad_parts.clear()
            completion_parts.clear()
            return next_index

        for turn in conversation:
            if not isinstance(turn, Mapping):
                raise RuntimeError(f"error: invalid chat turn in {conversation}")
            role = cast(str, turn.get(self.role_field))
            content = cast(str, turn.get(self.content_field))
            if role == self.system_role:
                if current_index > 0:
                    raise RuntimeError(f"error: system prompt at {current_index} in {conversation[:300]}")
                system_parts.append(content)
            elif role == self.user_role:
                current_index = _flush(current_index, last=False)
                prompt_parts.append(content)
            elif role == self.assistant_role:
                if not prompt_parts:
                    console_log(
                        "chat-to-sft",
                        f"**** error ****: assistant response without user prompt in {uid} at {current_index} in {conversation[:150]}",
                    )
                    continue
                scratchpad, final = self._split_assistant_content(content)
                if scratchpad:
                    scratchpad_parts.append(scratchpad)
                if final:
                    completion_parts.append(final)
            else:
                raise RuntimeError(f"error: unsupported role: [{role}] at {current_index} in {conversation[:300]}")

        _flush(current_index, last=True)
        return result

    def _split_assistant_content(self, content: str) -> tuple[str | None, str]:
        if not self.assistant_content_regex:
            return (None, content)
        match = re.match(self.assistant_content_regex, content, flags=re.DOTALL)
        if match is None:
            return (None, content)
        scratchpad = match.group(self.assistant_scratchpad_group).strip()
        final = match.group(self.assistant_final_group).strip()
        return (scratchpad or None, final)
