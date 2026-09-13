from __future__ import annotations

from .eval import PRESETS as EVAL_PRESETS
from .midtrain import PRESETS as MIDTRAIN_PRESETS
from .pretrain import PRESETS as PRETRAIN_PRESETS
from .sft import PRESETS as SFT_PRESETS


TRAINING_VIEW_PRESETS = [
    *PRETRAIN_PRESETS,
    *MIDTRAIN_PRESETS,
    *SFT_PRESETS,
    *EVAL_PRESETS,
]


__all__ = ["TRAINING_VIEW_PRESETS"]
