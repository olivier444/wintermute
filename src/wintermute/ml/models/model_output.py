from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerKvCache:
    key: torch.Tensor
    value: torch.Tensor
    length: int


PastKeyValues = tuple[LayerKvCache, ...]


@dataclass(frozen=True)
class CausalLmOutput:
    """Stable output contract for Wintermute causal language models."""

    logits: torch.Tensor
    past_key_values: PastKeyValues | None = None
