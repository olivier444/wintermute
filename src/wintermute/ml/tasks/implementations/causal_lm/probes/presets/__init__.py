from __future__ import annotations

from wintermute.ml.tasks.implementations.causal_lm.probes import ProbeSuite

from .pretrain import PRETRAIN_PROBE_SUITE
from .sft import SFT_PROBE_SUITE


PROBE_PRESETS = (
    PRETRAIN_PROBE_SUITE,
    SFT_PROBE_SUITE,
)


def list_probe_presets() -> dict[str, ProbeSuite]:
    return {preset.name: preset for preset in PROBE_PRESETS}


def get_probe_preset(key: str) -> ProbeSuite:
    normalized = key.strip().lower()
    for preset in PROBE_PRESETS:
        if preset.name.lower() == normalized:
            return preset
    raise ValueError(f"unknown probe preset: {normalized}")


__all__ = [
    "PROBE_PRESETS",
    "PRETRAIN_PROBE_SUITE",
    "SFT_PROBE_SUITE",
    "get_probe_preset",
    "list_probe_presets",
]
