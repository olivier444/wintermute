from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from wintermute.ml.tasks.implementations.causal_lm.probes.base import ProbeSampling, validate_score
from wintermute.ml.tasks.implementations.causal_lm.probes.metrics.base import ProbeMetric


@dataclass(frozen=True)
class MeanScore(ProbeMetric):
    sampling: ProbeSampling
    name: str = "mean_score"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("metric name must be a non-empty string")

    def compute(self, scores: Sequence[float]) -> float | None:
        if not scores:
            return None
        for score in scores:
            validate_score(score)
        return sum(scores) / len(scores)
