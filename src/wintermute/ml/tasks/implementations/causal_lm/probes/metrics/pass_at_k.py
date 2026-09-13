from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from wintermute.ml.tasks.implementations.causal_lm.probes.base import ProbeSampling, validate_score
from wintermute.ml.tasks.implementations.causal_lm.probes.metrics.base import ProbeMetric


@dataclass(frozen=True)
class PassAtK(ProbeMetric):
    k: int
    threshold: float
    sampling: ProbeSampling
    name: str = ""

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("PassAtK.k must be positive")
        if self.sampling.sample_count < self.k:
            raise ValueError("PassAtK sampling must provide at least k candidates")
        validate_score(self.threshold)
        if not self.name:
            object.__setattr__(self, "name", f"pass@{self.k}")
        elif not self.name.strip():
            raise ValueError("metric name must be a non-empty string")

    def compute(self, scores: Sequence[float]) -> float | None:
        candidate_count = len(scores)
        if candidate_count == 0 or self.k > candidate_count:
            return None

        for score in scores:
            validate_score(score)
        passing_count = sum(score >= self.threshold for score in scores)
        failing_count = candidate_count - passing_count
        if failing_count < self.k:
            return 1.0

        return 1.0 - (
            math.comb(failing_count, self.k)
            / math.comb(candidate_count, self.k)
        )
