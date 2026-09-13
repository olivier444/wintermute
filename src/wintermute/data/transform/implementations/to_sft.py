from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from random import Random
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
from wintermute.data.transform.implementations.helpers import ensure_allowed_params, lookup_path
from wintermute.data.transform.implementations.task import TaskSpec
from wintermute.tools.params import get_bool, get_optional_str, get_str


ALLOWED_PARAMS = {
    "items_field", "prompt", "completion", "task", "scratchpad", "system_prompt", "context", "emit_filter", "context_filter", "joiner", "context_joiner",
}

CHOICE_RESPONSE_FORMATS = {
    "option_number",
    "option_label",
    "option_number_and_label",
}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _normalize_parts(raw_parts: Any, *, field_name: str) -> List[Mapping[str, Any]]:
    if raw_parts is None:
        return []
    if isinstance(raw_parts, Mapping):
        return [raw_parts]
    if isinstance(raw_parts, (str, bytes, bytearray)) or not isinstance(raw_parts, Sequence):
        raise TypeError(f"{field_name} must be a part mapping or sequence of part mappings")
    result: List[Mapping[str, Any]] = []
    for part in raw_parts:
        if not isinstance(part, Mapping):
            raise TypeError(f"{field_name} parts must be mappings; use {{'field': '...'}} for source fields")
        result.append(part)
    return result


def _validate_random_text(parts: Sequence[Mapping[str, Any]], *, field_name: str) -> None:
    for part in parts:
        if "random_text" not in part:
            continue
        choices = part["random_text"]
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes, bytearray)):
            raise TypeError(f"{field_name}.random_text must be a sequence of strings")
        if not choices:
            raise ValueError(f"{field_name}.random_text cannot be empty")
        if any(not isinstance(choice, str) or not choice.strip() for choice in choices):
            raise ValueError(f"{field_name}.random_text must contain only non-empty strings")


def _normalize_filter(raw_filter: Any, *, field_name: str) -> Mapping[str, Any] | None:
    if raw_filter is None:
        return None
    if not isinstance(raw_filter, Mapping):
        raise TypeError(f"{field_name} must be a mapping when provided")
    if "field" not in raw_filter or "equals" not in raw_filter:
        raise ValueError(f"{field_name} must define 'field' and 'equals'")
    return raw_filter


def _as_string_list(value: Any, *, field_name: str) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RuntimeError(f"Unable to process field '{field_name}': expected sequence, got {type(value).__name__}")
    return [_coerce_text(item) for item in value]


def _render_choices(value: Any, *, field_name: str) -> str:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Unable to process field '{field_name}': expected mapping, got {type(value).__name__}")
    labels = _as_string_list(value.get("label"), field_name=f"{field_name}.label")
    texts = _as_string_list(value.get("text"), field_name=f"{field_name}.text")
    if len(labels) != len(texts):
        raise RuntimeError(f"Invalid choice payload in '{field_name}': labels/text lengths differ")
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))


def _normalize_number(value: Any) -> str:
    return str(int(value)) if isinstance(value, float) and value.is_integer() else _coerce_text(value)


def _normalize_math_reasoning(value: Any) -> str:
    text = _coerce_text(value)
    text = re.sub(r"<<[^>]+>>", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if "####" not in text:
        return text
    reasoning, final_answer = text.split("####", 1)
    reasoning, final_answer = reasoning.strip(), final_answer.strip()
    if reasoning and final_answer:
        return f"{reasoning}\nFinal answer: {final_answer}"
    return final_answer or reasoning


def _render_choice_response(index: int, option_label: str, part: Mapping[str, Any]) -> str:
    response_format = str(part.get("response_format", "option_label")).strip()
    if response_format not in CHOICE_RESPONSE_FORMATS:
        expected = ", ".join(sorted(CHOICE_RESPONSE_FORMATS))
        raise ValueError(f"Invalid choice response_format '{response_format}'; expected one of: {expected}")

    option_number = str(index + 1)
    if response_format == "option_number":
        return option_number
    if response_format == "option_number_and_label":
        return f"{option_number}. {option_label}"
    return option_label


def _boolean_labels(params: Mapping[str, Any]) -> tuple[str, str]:
    true_label = _coerce_text(params.get("true_label", "yes")).strip()
    false_label = _coerce_text(params.get("false_label", "no")).strip()
    if not true_label or not false_label:
        raise ValueError("boolean choice labels must be non-empty")
    if true_label == false_label:
        raise ValueError("boolean choice labels must be distinct")
    return true_label, false_label


def _choice_order(
    choices: Sequence[str],
    *,
    shuffle: bool,
    rng: Random,
    cache: Dict[tuple[str, tuple[str, ...]], tuple[int, ...]],
    namespace: str,
) -> tuple[int, ...]:
    order = tuple(range(len(choices)))
    if not shuffle:
        return order
    key = (namespace, tuple(choices))
    cached = cache.get(key)
    if cached is not None:
        return cached
    shuffled = list(order)
    rng.shuffle(shuffled)
    result = tuple(shuffled)
    cache[key] = result
    return result


def _boolean_choice_labels(
    *,
    true_label: str,
    false_label: str,
    rng: Random,
    choice_orders: Dict[tuple[str, tuple[str, ...]], tuple[int, ...]],
) -> tuple[str, str]:
    labels = (true_label, false_label)
    order = _choice_order(
        labels,
        shuffle=True,
        rng=rng,
        cache=choice_orders,
        namespace="boolean",
    )
    return labels[order[0]], labels[order[1]]


def _render_boolean_choices(
    *,
    true_label: str,
    false_label: str,
    rng: Random,
    choice_orders: Dict[tuple[str, tuple[str, ...]], tuple[int, ...]],
) -> str:
    return "\n".join(
        f"{index}. {label}"
        for index, label in enumerate(
            _boolean_choice_labels(
                true_label=true_label,
                false_label=false_label,
                rng=rng,
                choice_orders=choice_orders,
            ),
            start=1,
        )
    )


def _static_choices(part: Mapping[str, Any]) -> List[str]:
    raw_choices = part.get("choices")
    if not isinstance(raw_choices, Sequence) or isinstance(raw_choices, (str, bytes, bytearray)):
        raise TypeError("to_sft 'choices' must be a sequence")
    choices = [_coerce_text(choice).strip() for choice in raw_choices]
    if not choices or any(not choice for choice in choices):
        raise ValueError("to_sft 'choices' must contain non-empty values")
    return choices


def _static_choice_order(
    choices: Sequence[str],
    *,
    shuffle: bool,
    rng: Random,
    choice_orders: Dict[tuple[str, tuple[str, ...]], tuple[int, ...]],
) -> tuple[int, ...]:
    return _choice_order(
        choices,
        shuffle=shuffle,
        rng=rng,
        cache=choice_orders,
        namespace="static",
    )


class ToSftRecordTransform(RecordTransform):
    """Render source fields into canonical SFT prompts and completions.

    Configured parts can format fields, choices, random text, context, a
    scratchpad, and a system prompt.  With ``items_field``, one source record
    can emit multiple turns while retaining the same record identifier.
    """

    KIND = "to_sft"

    def __init__(self, params: Mapping[str, Any]):
        ensure_allowed_params(params, ALLOWED_PARAMS, kind=self.KIND)
        self.joiner = get_str(params, "joiner", "\n\n")
        self.context_joiner = get_str(params, "context_joiner", "\n")
        self.items_field = get_optional_str(params, "items_field")
        self.prompt_parts = _normalize_parts(params.get("prompt"), field_name="prompt")
        self.completion_parts = _normalize_parts(params.get("completion"), field_name="completion")
        self.task_spec = TaskSpec.from_config(params.get("task"))
        self.scratchpad_parts = _normalize_parts(params.get("scratchpad"), field_name="scratchpad")
        self.system_prompt_parts = _normalize_parts(params.get("system_prompt"), field_name="system_prompt")
        self.context_parts = _normalize_parts(params.get("context"), field_name="context")
        for name, parts in (("prompt", self.prompt_parts), ("completion", self.completion_parts), ("scratchpad", self.scratchpad_parts), ("system_prompt", self.system_prompt_parts), ("context", self.context_parts)):
            _validate_random_text(parts, field_name=name)
        self.emit_filter = _normalize_filter(params.get("emit_filter"), field_name="emit_filter")
        self.context_filter = _normalize_filter(params.get("context_filter"), field_name="context_filter")
        if not self.prompt_parts:
            raise ValueError("to_sft requires a non-empty 'prompt' specification")
        if not self.completion_parts:
            raise ValueError("to_sft requires a non-empty 'completion' specification")
        if self.items_field is None and self.context_filter is not None:
            raise ValueError("to_sft.context_filter requires 'items_field'")
        if self.items_field is None and self.context_parts:
            raise ValueError("to_sft.context requires 'items_field'")
        if self.items_field is None and self.emit_filter is not None:
            raise ValueError("to_sft.emit_filter requires 'items_field'")

    def transform(self, record: DataRecord, *, context: TransformContext) -> List[DataRecord]:
        turns: List[tuple[str, str | None, str, str]] = []
        context_chunks: List[str] = []
        choice_orders: Dict[tuple[str, tuple[str, ...]], tuple[int, ...]] = {}

        for scope in self._iter_scopes(record.fields):
            if self.context_parts and self._matches_filter(self.context_filter, scope):
                context_chunk = self._render_parts(
                    self.context_parts,
                    scope,
                    context=context,
                    choice_orders=choice_orders,
                )
                if context_chunk:
                    context_chunks.append(context_chunk)
            if not self._matches_filter(self.emit_filter, scope):
                continue
            prompt = self._render_parts(
                self.prompt_parts,
                scope,
                context=context,
                choice_orders=choice_orders,
            )
            completion = self._render_parts(
                self.completion_parts,
                scope,
                context=context,
                choice_orders=choice_orders,
            )
            task = self.task_spec.resolve(scope) if self.task_spec is not None else None
            scratchpad = self._render_parts(
                self.scratchpad_parts,
                scope,
                context=context,
                choice_orders=choice_orders,
            )
            if context_chunks:
                context_payload = self.context_joiner.join(context_chunks).strip()
                if context_payload:
                    prompt = self.joiner.join(part for part in (context_payload, prompt) if part)
            if prompt.strip() and completion.strip():
                turns.append((prompt.strip(), task, scratchpad.strip(), completion.strip()))

        if not turns:
            return []

        fields: Dict[str, str] = {}
        system_prompt = self._render_parts(
            self.system_prompt_parts,
            record.fields,
            context=context,
            choice_orders=choice_orders,
        ).strip()

        if system_prompt:
            fields[FLD_GENERIC_SYSTEM_PROMPT] = system_prompt

        if len(turns) == 1:
            prompt, task, scratchpad, completion = turns[0]
            fields[FLD_GENERIC_PROMPT] = prompt
            fields[FLD_GENERIC_COMPLETION] = completion
            if task:
                fields[FLD_GENERIC_TASK] = task
            if scratchpad:
                fields[FLD_GENERIC_SCRATCHPAD] = scratchpad
            return [DataRecord(record_id=record.record_id, fields=fields)]

        for index, (prompt, task, scratchpad, completion) in enumerate(turns):
            fields[f"{FLD_GENERIC_PROMPT}.{index}"] = prompt
            fields[f"{FLD_GENERIC_COMPLETION}.{index}"] = completion
            if task:
                fields[f"{FLD_GENERIC_TASK}.{index}"] = task
            if scratchpad:
                fields[f"{FLD_GENERIC_SCRATCHPAD}.{index}"] = scratchpad

        return [DataRecord(record_id=record.record_id, fields=fields)]

    def _iter_scopes(self, input_record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        if self.items_field is None:
            return [input_record]
        found, items_value = lookup_path(input_record, self.items_field)
        if not found:
            raise RuntimeError(f"Unable to process items_field '{self.items_field}': field not found")
        if isinstance(items_value, list):
            return [item if isinstance(item, Mapping) else {"value": item} for item in items_value]
        if isinstance(items_value, Mapping):
            return self._iter_columnar_items(items_value)
        raise RuntimeError(
            f"Unable to process items_field '{self.items_field}': expected list or mapping, got {type(items_value).__name__}"
        )

    def _iter_columnar_items(self, value: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        keys = list(value.keys())
        if not keys:
            return []
        columns: Dict[str, Sequence[Any]] = {}
        row_count: int | None = None
        for key in keys:
            column = value[key]
            if not isinstance(column, Sequence) or isinstance(column, (str, bytes, bytearray)):
                raise RuntimeError(
                    f"Unable to iterate items in '{self.items_field}': field '{key}' is not a sequence "
                    f"(got {type(column).__name__})"
                )
            columns[str(key)] = column
            if row_count is None:
                row_count = len(column)
            elif len(column) != row_count:
                raise RuntimeError(f"Unable to iterate items in '{self.items_field}': inconsistent sequence lengths")
        assert row_count is not None
        return [{key: columns[key][index] for key in columns} for index in range(row_count)]

    def _matches_filter(
        self,
        filter_spec: Mapping[str, Any] | None,
        scope: Mapping[str, Any],
    ) -> bool:
        if filter_spec is None:
            return True
        found, value = lookup_path(scope, str(filter_spec["field"]))
        if not found:
            return False
        expected = filter_spec["equals"]
        if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes, bytearray)):
            return value in expected
        return value == expected

    def _render_parts(
        self,
        parts: Sequence[Mapping[str, Any]],
        scope: Mapping[str, Any],
        *,
        context: TransformContext,
        choice_orders: Dict[tuple[str, tuple[str, ...]], tuple[int, ...]],
    ) -> str:
        rendered_parts = [
            self._render_part(
                part,
                scope,
                context=context,
                choice_orders=choice_orders,
            ).strip()
            for part in parts
        ]
        return self.joiner.join(part for part in rendered_parts if part)

    def _render_part(
        self,
        part: Mapping[str, Any],
        scope: Mapping[str, Any],
        *,
        context: TransformContext,
        choice_orders: Dict[tuple[str, tuple[str, ...]], tuple[int, ...]],
    ) -> str:
        if "text" in part:
            return _coerce_text(part["text"])
        if "random_text" in part:
            choices = cast(Sequence[str], part["random_text"])
            return context.rng.choice(choices)
        if "boolean_choices" in part:
            raw_boolean_choices = part["boolean_choices"]
            if raw_boolean_choices is True:
                labels: Mapping[str, Any] = {}
            elif isinstance(raw_boolean_choices, Mapping):
                labels = raw_boolean_choices
            else:
                raise TypeError("to_sft 'boolean_choices' must be true or a label mapping")
            true_label, false_label = _boolean_labels(labels)
            return _render_boolean_choices(
                true_label=true_label,
                false_label=false_label,
                rng=context.rng,
                choice_orders=choice_orders,
            )
        if "choices" in part and "field" not in part and "fields" not in part:
            choices = _static_choices(part)
            order = _static_choice_order(
                choices,
                shuffle=get_bool(part, "shuffle", False),
                rng=context.rng,
                choice_orders=choice_orders,
            )
            return "\n".join(
                f"{display_index}. {choices[source_index]}"
                for display_index, source_index in enumerate(order, start=1)
            )
        transform = str(part.get("transform", "identity")).strip() or "identity"
        if "field" in part:
            field_name = str(part["field"])
            found, value = lookup_path(scope, field_name)
            if not found:
                raise RuntimeError(f"Unable to process field '{field_name}': field not found")
            return self._apply_transform(
                transform,
                value,
                part,
                scope,
                context=context,
                choice_orders=choice_orders,
                field_name=field_name,
            )
        if "fields" in part:
            raw_fields = part["fields"]
            if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes, bytearray)):
                raise TypeError("to_sft 'fields' transform input must be a sequence")
            values: List[Any] = []
            field_names: List[str] = []
            for raw_field in raw_fields:
                field_name = str(raw_field)
                found, value = lookup_path(scope, field_name)
                if not found:
                    raise RuntimeError(f"Unable to process field '{field_name}': field not found")
                values.append(value)
                field_names.append(field_name)
            return self._apply_transform(
                transform,
                values,
                part,
                scope,
                context=context,
                choice_orders=choice_orders,
                field_name=", ".join(field_names),
            )
        raise ValueError(
            "to_sft part spec must define either 'field', 'fields', 'text', "
            "'random_text', 'boolean_choices', or 'choices'"
        )

    def _apply_transform(
        self,
        transform: str,
        value: Any,
        part: Mapping[str, Any],
        scope: Mapping[str, Any],
        *,
        context: TransformContext,
        choice_orders: Dict[tuple[str, tuple[str, ...]], tuple[int, ...]],
        field_name: str,
    ) -> str:
        if transform == "identity":
            return _coerce_text(value)
        if transform == "bool_yes_no":
            true_label, false_label = _boolean_labels(part)
            label = true_label if bool(value) else false_label
            labels = _boolean_choice_labels(
                true_label=true_label,
                false_label=false_label,
                rng=context.rng,
                choice_orders=choice_orders,
            )
            return _render_choice_response(labels.index(label), label, part)
        if transform == "choices":
            return _render_choices(value, field_name=field_name)
        if transform == "numbered_choices":
            if isinstance(value, Mapping):
                choices = _as_string_list(value.get("text"), field_name=f"{field_name}.text")
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                choices = [_coerce_text(choice) for choice in value]
            else:
                raise RuntimeError(f"Unable to process field '{field_name}': expected choices, got {type(value).__name__}")
            return "\n".join(f"{index}. {choice}" for index, choice in enumerate(choices, start=1))
        if transform == "normalize_number":
            return _normalize_number(value)
        if transform == "math_reasoning":
            return _normalize_math_reasoning(value)
        if transform in {"choice_by_label", "choice_text_by_label"}:
            choices_field = str(part.get("choices_field", "")).strip()
            if not choices_field:
                raise ValueError(f"{transform} requires 'choices_field'")
            found, choices_value = lookup_path(scope, choices_field)
            if not found:
                raise RuntimeError(f"Unable to process field '{choices_field}': field not found")
            return self._choice_by_label(value, choices_value, part, field_name, choices_field)
        if transform in {"choice_by_index", "choice_text_by_index"}:
            raw_choice_fields = part.get("choices_fields")
            static_choices = part.get("choices")
            if static_choices is not None:
                choices: List[Any] = _static_choices(part)
            else:
                if not isinstance(raw_choice_fields, Sequence) or isinstance(raw_choice_fields, (str, bytes, bytearray)):
                    raise ValueError(f"{transform} requires 'choices_fields' or 'choices'")
                choices = []
                for raw_choice_field in raw_choice_fields:
                    choice_field = str(raw_choice_field)
                    found, choice = lookup_path(scope, choice_field)
                    if not found:
                        raise RuntimeError(f"Unable to process field '{choice_field}': field not found")
                    choices.append(choice)
            index_token = _coerce_text(value).strip()
            try:
                index = int(index_token)
            except ValueError as exc:
                raise RuntimeError(f"Unable to process field '{field_name}': invalid index '{index_token}'") from exc
            if get_bool(part, "one_based", True):
                index -= 1
            if index < 0 or index >= len(choices):
                raise RuntimeError(
                    f"Unable to process field '{field_name}': index {index_token} out of range for {len(choices)} choices"
                )
            display_index = index
            if static_choices is not None:
                order = _static_choice_order(
                    [_coerce_text(choice) for choice in choices],
                    shuffle=get_bool(part, "shuffle", False),
                    rng=context.rng,
                    choice_orders=choice_orders,
                )
                display_index = order.index(index)
            return _render_choice_response(display_index, _coerce_text(choices[index]), part)
        if transform == "placeholder_substitute":
            values_field = str(part.get("values_field", "")).strip()
            if not values_field:
                raise ValueError("placeholder_substitute requires 'values_field'")
            found, values = lookup_path(scope, values_field)
            if not found:
                raise RuntimeError(f"Unable to process field '{values_field}': field not found")
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
                raise RuntimeError(f"Unable to process field '{values_field}': expected sequence, got {type(values).__name__}")
            rendered = _coerce_text(value)
            for index, item in enumerate(values):
                rendered = rendered.replace(f"number{index}", _normalize_number(item))
            return rendered
        raise ValueError(f"Unknown to_sft transform: {transform}")

    def _choice_by_label(
        self,
        label_value: Any,
        choices_value: Any,
        part: Mapping[str, Any],
        label_field: str,
        choices_field: str,
    ) -> str:
        label = _coerce_text(label_value).strip()
        if not label:
            raise RuntimeError(f"Unable to process field '{label_field}': empty choice label")
        if not isinstance(choices_value, Mapping):
            raise RuntimeError(f"Unable to process field '{choices_field}': expected mapping, got {type(choices_value).__name__}")
        labels = _as_string_list(choices_value.get("label"), field_name=f"{choices_field}.label")
        texts = _as_string_list(choices_value.get("text"), field_name=f"{choices_field}.text")
        for index, (current_label, current_text) in enumerate(zip(labels, texts)):
            if current_label == label:
                return _render_choice_response(index, current_text, part)
        raise RuntimeError(f"Unable to process field '{label_field}': label '{label}' not found in '{choices_field}'")
