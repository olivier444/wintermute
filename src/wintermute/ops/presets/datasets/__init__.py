from __future__ import annotations

from .pretrain import PRETRAIN_DATASET_PRESETS
from .sft import SFT_DATASET_PRESETS

DATASET_PRESETS = [
    *PRETRAIN_DATASET_PRESETS,
    *SFT_DATASET_PRESETS,
]
