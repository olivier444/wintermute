from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from lm_eval.api.model import LM  # type: ignore[import-not-found]


SAMPLES_REPORT_FILENAME = "samples.md"


@dataclass(frozen=True)
class LmEvalTrace:
    request_type: str
    arguments: tuple[Any, ...]
    response: Any


@dataclass
class LmEvalTraceCollector:
    sample_rate: float
    traces: list[LmEvalTrace] = field(default_factory=list)

    def record(self, request_type: str, arguments: Iterable[Any], response: Any) -> None:
        normalized_arguments = tuple(arguments)
        if not self._is_selected(request_type, normalized_arguments):
            return
        self.traces.append(
            LmEvalTrace(
                request_type=request_type,
                arguments=normalized_arguments,
                response=response,
            )
        )

    def write_report(self, path: Path) -> None:
        report = [
            "# lm-eval sampled traces",
            "",
            f"Sample rate: {self.sample_rate:.2%}",
            f"Captured requests: {len(self.traces)}",
        ]
        for index, trace in enumerate(self.traces, start=1):
            report.extend(_render_trace(index, trace))
        path.write_text("\n".join(report) + "\n", encoding="utf-8")

    def _is_selected(self, request_type: str, arguments: Sequence[Any]) -> bool:
        if self.sample_rate >= 1.0:
            return True
        prompt = str(arguments[0]) if arguments else ""
        digest = hashlib.sha256(
            f"{request_type}\0{prompt}".encode("utf-8")
        ).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        return value < self.sample_rate


class TracedLmEvalAdapter(LM):
    """Delegate an lm-eval adapter while retaining a deterministic request sample."""

    def __init__(self, adapter: Any, collector: LmEvalTraceCollector) -> None:
        super().__init__()
        self._adapter = adapter
        self._collector = collector

    @property
    def device(self) -> Any:
        return self._adapter.device

    @property
    def rank(self) -> int:
        return self._adapter.rank

    @property
    def world_size(self) -> int:
        return self._adapter.world_size

    @property
    def tokenizer_name(self) -> str:
        return self._adapter.tokenizer_name

    def set_cache_hook(self, cache_hook: Any) -> None:
        super().set_cache_hook(cache_hook)
        self._adapter.set_cache_hook(cache_hook)

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        responses = self._adapter.loglikelihood(requests)
        self._record("loglikelihood", requests, responses)
        return responses

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        responses = self._adapter.loglikelihood_rolling(requests)
        self._record("loglikelihood_rolling", requests, responses)
        return responses

    def generate_until(self, requests: list[Any]) -> list[str]:
        responses = self._adapter.generate_until(requests)
        self._record("generate_until", requests, responses)
        return responses

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def _record(self, request_type: str, requests: Sequence[Any], responses: Sequence[Any]) -> None:
        for request, response in zip(requests, responses, strict=True):
            self._collector.record(request_type, request.args, response)


def _render_trace(index: int, trace: LmEvalTrace) -> list[str]:
    prompt = str(trace.arguments[0]) if trace.arguments else ""
    sections = [
        "",
        f"## {index}. {trace.request_type}",
        "",
        "### Prompt",
        "```text",
        prompt,
        "```",
    ]
    if len(trace.arguments) > 1:
        sections.extend(
            [
                "",
                "### Request options",
                "```json",
                _render_value(trace.arguments[1:]),
                "```",
            ]
        )
    sections.extend(
        [
            "",
            "### Model result",
            "```text",
            _render_value(trace.response),
            "```",
        ]
    )
    return sections


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
