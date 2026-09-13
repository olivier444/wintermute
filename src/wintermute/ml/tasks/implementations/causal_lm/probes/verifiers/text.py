from __future__ import annotations

import math
import re
from dataclasses import dataclass

from wintermute.ml.tasks.implementations.causal_lm.probes.verifiers.base import Verifier


def _normalize_text(value: str, *, strip: bool) -> str:
    if strip:
        value = value.strip()
    return value.casefold()


@dataclass(frozen=True)
class Exact(Verifier):
    expected: str
    strip: bool = True
    embedded_score: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.embedded_score) or not 0 <= self.embedded_score <= 1:
            raise ValueError("Exact.embedded_score must be in [0, 1]")

    def score(self, completion: str) -> float:
        actual = _normalize_text(
            completion,
            strip=self.strip,
        )
        expected = _normalize_text(
            self.expected,
            strip=self.strip,
        )
        if actual == expected:
            return 1.0
        if (
            self.embedded_score > 0
            and expected
            and re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", actual)
        ):
            return self.embedded_score
        return 0.0


@dataclass(frozen=True)
class OneOf(Verifier):
    expected: tuple[str, ...]
    strip: bool = True

    def __post_init__(self) -> None:
        if not self.expected:
            raise ValueError("OneOf requires at least one expected completion")

    def score(self, completion: str) -> float:
        actual = _normalize_text(
            completion,
            strip=self.strip,
        )
        accepted = {
            _normalize_text(value, strip=self.strip)
            for value in self.expected
        }
        return 1.0 if actual in accepted else 0.0


@dataclass(frozen=True)
class ContainsAny(Verifier):
    expected: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.expected or any(not value.strip() for value in self.expected):
            raise ValueError("ContainsAny requires at least one non-empty term")

    def score(self, completion: str) -> float:
        actual = completion.casefold()
        return 1.0 if any(
            re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", actual)
            for term in self.expected
        ) else 0.0


@dataclass(frozen=True)
class StartsWith(Verifier):
    expected: str
    strip: bool = True

    def __post_init__(self) -> None:
        if not _normalize_text(self.expected, strip=self.strip):
            raise ValueError("StartsWith requires a non-empty expected prefix")

    def score(self, completion: str) -> float:
        actual = _normalize_text(completion, strip=self.strip)
        expected = _normalize_text(self.expected, strip=self.strip)
        return 1.0 if actual.startswith(expected) else 0.0

