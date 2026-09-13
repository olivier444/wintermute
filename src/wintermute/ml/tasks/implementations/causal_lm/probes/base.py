from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

from wintermute.ml.tasks.implementations.causal_lm.probes.verifiers import Verifier
from wintermute.ml.tokenization.chat_format import (
    AssistantCompletion,
    AssistantPromptFormat,
)

if TYPE_CHECKING:
    from wintermute.ml.tasks.implementations.causal_lm.probes.metrics.base import ProbeMetric


def validate_score(score: float) -> None:
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("probe scores must be finite and in [0, 1]")


@dataclass(frozen=True)
class ProbeSampling:
    sample_count: int = 1
    temperature: float = 0.7
    top_p: float = 0.85
    top_k: int | None = 50
    greedy: bool = False

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("ProbeSampling.sample_count must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("ProbeSampling.temperature must be finite and positive")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("ProbeSampling.top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("ProbeSampling.top_k must be positive when provided")

    def distribution_key(self) -> tuple[object, ...]:
        if self.greedy:
            return ("greedy",)
        return ("sample", self.temperature, self.top_p, self.top_k)


@dataclass(frozen=True)
class Probe:
    id: str
    prompt: str
    assistant_prompt_format: AssistantPromptFormat | None = None
    verifier: Verifier | None = None
    max_new_tokens: int = 80

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Probe.id must be a non-empty string")
        if not self.prompt:
            raise ValueError("Probe.prompt must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("Probe.max_new_tokens must be positive")


@dataclass(frozen=True)
class ProbeSuite:
    name: str
    probes: tuple[Probe, ...]
    observable_sampling: ProbeSampling
    metrics: tuple[ProbeMetric, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ProbeSuite.name must be a non-empty string")
        if not self.probes:
            raise ValueError("ProbeSuite.probes must not be empty")
        probe_ids = [probe.id for probe in self.probes]
        if len(set(probe_ids)) != len(probe_ids):
            raise ValueError("ProbeSuite probe ids must be unique")
        metric_names = [metric.name for metric in self.metrics]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("ProbeSuite metric names must be unique")


@dataclass(frozen=True)
class ProbeCandidate:
    completion: AssistantCompletion
    score: float | None = None

    def __post_init__(self) -> None:
        if self.score is not None:
            validate_score(self.score)


@dataclass(frozen=True)
class ProbeResult:
    probe_id: str
    candidates: tuple[ProbeCandidate, ...]
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class ProbeReport:
    suite_name: str
    results: tuple[ProbeResult, ...]
    metrics: Mapping[str, float]
