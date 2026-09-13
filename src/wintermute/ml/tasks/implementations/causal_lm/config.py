from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict

from wintermute.data.iterate.completion_format import (
    SUPPORTED_FORMAT_MATCHERS,
    CompletionFormatRule,
)
from wintermute.data.iterate.dataclasses import PackingPolicy, WindowingPolicy
from wintermute.tools.params import (
    get_bool,
    get_float,
    get_int,
    get_optional_list,
    get_optional_mapping,
    get_str,
)


_ALLOWED_ASSISTANT_FORMATS = {"legacy", "normalized"}


@dataclass(frozen=True)
class CompletionPositionWeighting:
    boundaries: tuple[int, ...]
    weights: tuple[float, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompletionPositionWeighting":
        boundaries = _parse_position_boundaries(
            data.get("boundaries", []),
            "completion_position_weighting.boundaries",
        )
        raw_weights = data.get("weights")
        if not isinstance(raw_weights, list):
            raise TypeError("completion_position_weighting.weights must be a list")

        weights = tuple(float(value) for value in raw_weights)
        if len(weights) != len(boundaries) + 1:
            raise ValueError(
                "completion_position_weighting.weights must contain one weight per position segment"
            )
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError(
                "completion_position_weighting.weights must be finite and non-negative"
            )
        return cls(boundaries=boundaries, weights=weights)


@dataclass(frozen=True)
class CompletionPrefixMasking:
    probability: float
    max_tokens: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CompletionPrefixMasking":
        probability = get_float(data, "probability", float("nan"))
        max_tokens = get_int(data, "max_tokens", 0)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("completion_prefix_masking.probability must be in [0, 1]")
        if max_tokens <= 0:
            raise ValueError("completion_prefix_masking.max_tokens must be a positive integer")
        return cls(probability=probability, max_tokens=max_tokens)


@dataclass(frozen=True)
class CausalLmTaskConfig:
    windowing_policy: WindowingPolicy
    packing_policy: PackingPolicy
    assistant_format: str = "normalized"
    tokenizer_id: str | None = None
    hf_tokenizer_name: str | None = None
    tokenizer_file: str | None = None
    eval_position_boundaries: tuple[int, ...] = (2, 7)
    completion_position_weighting: CompletionPositionWeighting | None = None
    completion_prefix_masking: CompletionPrefixMasking | None = None
    use_task_labels: bool = False
    tag_loss_weight: float = 1.0
    verbosity_labels: tuple[tuple[str, int | None], ...] | None = None
    format_labels: tuple[CompletionFormatRule, ...] | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalLmTaskConfig":
        d = dict(data)

        if "windowing_policy" not in d or d["windowing_policy"] is None:
            raise ValueError("CausalLmTaskWrapper requires 'windowing_policy' in task params")
        if "assistant_format" not in d:
            raise ValueError("CausalLmTaskWrapper requires 'assistant_format' in task params")

        d["windowing_policy"] = WindowingPolicy.from_dict(d["windowing_policy"])
        if "packing_policy" in d and d["packing_policy"] is not None:
            d["packing_policy"] = PackingPolicy.from_dict(d["packing_policy"])
        else:
            d["packing_policy"] = PackingPolicy()
        d["assistant_format"] = get_str(d, "assistant_format", "").strip().lower()
        if d["assistant_format"] not in _ALLOWED_ASSISTANT_FORMATS:
            allowed = ", ".join(sorted(_ALLOWED_ASSISTANT_FORMATS))
            raise ValueError(f"assistant_format must be one of: {allowed}")
        d["eval_position_boundaries"] = _parse_position_boundaries(
            d.get("eval_position_boundaries", [2, 7]),
            "eval_position_boundaries",
        )
        raw_position_weighting = d.get("completion_position_weighting")
        if raw_position_weighting is not None:
            if not isinstance(raw_position_weighting, Mapping):
                raise TypeError("completion_position_weighting must be a mapping")
            d["completion_position_weighting"] = CompletionPositionWeighting.from_dict(
                raw_position_weighting
            )
        raw_prefix_masking = d.get("completion_prefix_masking")
        if raw_prefix_masking is not None:
            if not isinstance(raw_prefix_masking, Mapping):
                raise TypeError("completion_prefix_masking must be a mapping")
            d["completion_prefix_masking"] = CompletionPrefixMasking.from_dict(
                raw_prefix_masking
            )
        d["use_task_labels"] = get_bool(d, "use_task_labels", False)
        d["tag_loss_weight"] = get_float(d, "tag_loss_weight", 1.0)
        if d["tag_loss_weight"] < 0:
            raise ValueError("tag_loss_weight must be non-negative")
        d["verbosity_labels"] = _parse_verbosity_labels(
            get_optional_mapping(d, "verbosity_labels")
        )
        d["format_labels"] = _parse_format_labels(
            get_optional_list(d, "format_labels")
        )

        tokenizer_id = d.get("tokenizer_id")
        hf_tokenizer_name = d.get("hf_tokenizer_name")
        tokenizer_file = d.get("tokenizer_file")
        tokenizer_source_count = sum(
            value is not None
            for value in (tokenizer_id, hf_tokenizer_name, tokenizer_file)
        )
        if tokenizer_source_count != 1:
            raise ValueError(
                "CausalLmTaskWrapper requires exactly one of 'tokenizer_id', "
                "'hf_tokenizer_name', or 'tokenizer_file' in task params"
            )

        return cls(**d)


def _parse_verbosity_labels(
    raw_labels: Mapping[str, Any] | None,
) -> tuple[tuple[str, int | None], ...] | None:
    if raw_labels is None:
        return None
    if not raw_labels:
        raise ValueError("verbosity_labels must not be empty")

    parsed_labels: list[tuple[str, int | None]] = []
    for label, max_tokens in raw_labels.items():
        if not isinstance(label, str) or not label:
            raise ValueError("verbosity_labels keys must be non-empty strings")
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
            raise ValueError(
                "verbosity_labels values must be positive integers or one null fallback"
            )
        parsed_labels.append((label, max_tokens))

    if sum(max_tokens is None for _, max_tokens in parsed_labels) != 1:
        raise ValueError("verbosity_labels must contain exactly one null fallback")
    finite_maxima = [
        max_tokens
        for _, max_tokens in parsed_labels
        if max_tokens is not None
    ]
    if len(set(finite_maxima)) != len(finite_maxima):
        raise ValueError("verbosity_labels max_tokens values must be unique")
    return tuple(
        sorted(
            parsed_labels,
            key=lambda item: item[1] if item[1] is not None else float("inf"),
        )
    )


def _parse_format_labels(
    raw_labels: list[Any] | None,
) -> tuple[CompletionFormatRule, ...] | None:
    if raw_labels is None:
        return None
    if not raw_labels:
        raise ValueError("format_labels must not be empty")

    parsed_labels: list[CompletionFormatRule] = []
    for index, raw_label in enumerate(raw_labels):
        context = f"format_labels[{index}]"
        if not isinstance(raw_label, Mapping):
            raise TypeError(f"{context} must be a mapping")

        label = raw_label.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{context}.label must be a non-empty string")

        pattern = raw_label.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"{context}.pattern must be a non-empty string")
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"{context}.pattern is not a valid regular expression") from error
        matcher = raw_label.get("matcher")
        if matcher is not None:
            if not isinstance(matcher, str) or not matcher:
                raise ValueError(f"{context}.matcher must be a non-empty string")
            if matcher not in SUPPORTED_FORMAT_MATCHERS:
                supported = ", ".join(sorted(SUPPORTED_FORMAT_MATCHERS))
                raise ValueError(f"{context}.matcher must be one of: {supported}")
        if pattern is not None and matcher is not None:
            raise ValueError(f"{context} may define either pattern or matcher, not both")
        parsed_labels.append(
            CompletionFormatRule(label=label, pattern=pattern, matcher=matcher)
        )

    if len({rule.label for rule in parsed_labels}) != len(parsed_labels):
        raise ValueError("format_labels labels must be unique")
    fallback_indexes = [
        index
        for index, rule in enumerate(parsed_labels)
        if rule.pattern is None and rule.matcher is None
    ]
    if fallback_indexes not in ([], [len(parsed_labels) - 1]):
        raise ValueError("format_labels may end with at most one pattern-less fallback")
    return tuple(parsed_labels)


def _parse_position_boundaries(value: Any, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    if any(type(boundary) is not int or boundary <= 0 for boundary in value):
        raise ValueError(f"{field_name} must contain positive integers")
    if any(left >= right for left, right in zip(value, value[1:])):
        raise ValueError(f"{field_name} must be strictly increasing")
    return tuple(value)
