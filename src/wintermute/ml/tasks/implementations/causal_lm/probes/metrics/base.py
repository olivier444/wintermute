from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from wintermute.ml.tasks.implementations.causal_lm.probes.base import ProbeSampling


class ProbeMetric(ABC):
    sampling: ProbeSampling
    name: str

    @abstractmethod
    def compute(self, scores: Sequence[float]) -> float | None:
        """Compute the metric for one probe's completion scores."""

def compute_probe_metrics(
    metrics: Sequence[ProbeMetric],
    scores: Sequence[float],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in metrics:
        value = metric.compute(scores)
        if value is not None:
            values[metric.name] = value
    return values
