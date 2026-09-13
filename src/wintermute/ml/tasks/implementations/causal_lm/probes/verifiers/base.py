from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Verifier(ABC):
    @abstractmethod
    def score(self, completion: str) -> float:
        """Score a model completion on the [0, 1] interval."""


def extract_fenced_code(completion: str, *, language: str) -> str | None:
    pattern = re.compile(
        rf"^[ \t]*```[ \t]*{re.escape(language)}[ \t]*\r?\n"
        r"(?P<source>.*?)"
        r"(?:\r?\n)?^[ \t]*```[ \t]*$",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    matches = tuple(pattern.finditer(completion))
    if len(matches) != 1:
        return None
    return matches[0].group("source")


@dataclass(frozen=True)
class AnyOf(Verifier):
    verifiers: tuple[Verifier, ...]

    def __post_init__(self) -> None:
        if not self.verifiers:
            raise ValueError("AnyOf requires at least one verifier")

    def score(self, completion: str) -> float:
        return max(verifier.score(completion) for verifier in self.verifiers)

