from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping


CompositeSourceExhaustionObserver = Callable[[int, int, int, int], bool]


@dataclass(frozen=True)
class TrainingViewIterationReport:
    summary: str
    metrics: Mapping[str, float]
