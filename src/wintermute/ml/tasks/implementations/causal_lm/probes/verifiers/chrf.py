from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from wintermute.ml.tasks.implementations.causal_lm.probes.verifiers.base import Verifier


def _character_ngrams(value: str, order: int) -> Counter[str]:
    return Counter(
        value[index : index + order]
        for index in range(len(value) - order + 1)
    )


@dataclass(frozen=True)
class ChrF(Verifier):
    references: tuple[str, ...]
    char_order: int = 6
    beta: float = 2.0
    ignore_whitespace: bool = True

    def __post_init__(self) -> None:
        if not self.references or any(not reference.strip() for reference in self.references):
            raise ValueError("ChrF requires at least one non-empty reference")
        if self.char_order <= 0:
            raise ValueError("ChrF.char_order must be positive")
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("ChrF.beta must be finite and positive")

    def score(self, completion: str) -> float:
        hypothesis = self._normalize(completion)
        if not hypothesis:
            return 0.0
        return max(
            self._score_reference(hypothesis, self._normalize(reference))
            for reference in self.references
        )

    def _normalize(self, value: str) -> str:
        value = value.casefold()
        if self.ignore_whitespace:
            return "".join(value.split())
        return value.strip()

    def _score_reference(self, hypothesis: str, reference: str) -> float:
        precision_sum = 0.0
        recall_sum = 0.0
        effective_orders = 0

        for order in range(1, self.char_order + 1):
            hypothesis_ngrams = _character_ngrams(hypothesis, order)
            reference_ngrams = _character_ngrams(reference, order)
            if not hypothesis_ngrams or not reference_ngrams:
                continue

            overlap = sum((hypothesis_ngrams & reference_ngrams).values())
            precision_sum += overlap / hypothesis_ngrams.total()
            recall_sum += overlap / reference_ngrams.total()
            effective_orders += 1

        if effective_orders == 0:
            return 0.0

        precision = precision_sum / effective_orders
        recall = recall_sum / effective_orders
        beta_squared = self.beta**2
        denominator = beta_squared * precision + recall
        if denominator == 0:
            return 0.0
        return (1 + beta_squared) * precision * recall / denominator
