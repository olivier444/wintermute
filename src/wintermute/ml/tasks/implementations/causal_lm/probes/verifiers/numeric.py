from __future__ import annotations

import math
import re
from dataclasses import dataclass

from wintermute.ml.tasks.implementations.causal_lm.probes.verifiers.base import Verifier


_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:[.,]\d+)?|[.,]\d+)")


def _parse_number(value: str) -> float:
    return float(value.replace(",", "."))


@dataclass(frozen=True)
class Numeric(Verifier):
    expected: float
    tolerance: float = 0.0
    embedded_score: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.expected):
            raise ValueError("Numeric.expected must be finite")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("Numeric.tolerance must be finite and non-negative")
        if not math.isfinite(self.embedded_score) or not 0 <= self.embedded_score <= 1:
            raise ValueError("Numeric.embedded_score must be in [0, 1]")

    def score(self, completion: str) -> float:
        stripped = completion.strip()
        if _NUMBER_PATTERN.fullmatch(stripped):
            return 1.0 if self._matches(_parse_number(stripped)) else 0.0

        if self.embedded_score > 0 and any(
            self._matches(_parse_number(match.group(0)))
            for match in _NUMBER_PATTERN.finditer(completion)
        ):
            return self.embedded_score

        return 0.0

    def _matches(self, value: float) -> bool:
        return math.isclose(
            value,
            self.expected,
            rel_tol=0.0,
            abs_tol=self.tolerance,
        )

