from __future__ import annotations

from dataclasses import dataclass
from random import Random


DEFAULT_RUNTIME_TRANSFORM_SEED = 1234


@dataclass(frozen=True)
class TransformContext:
    rng: Random
    iteration: int


def new_runtime_transform_rng(seed: int = DEFAULT_RUNTIME_TRANSFORM_SEED) -> Random:
    return Random(seed)
