from __future__ import annotations

from .assistant_textproc import PRESET as ASSISTANT_TEXTPROC_PRESET
from .common import PRESETS as COMMON_PRESETS
from .task_routing import PRESET as TASK_ROUTING_PRESET
from .text_processing_sft import PRESET as TEXT_PROCESSING_SFT_PRESET


PRESETS = [
    *COMMON_PRESETS,
    TEXT_PROCESSING_SFT_PRESET,
    ASSISTANT_TEXTPROC_PRESET,
    TASK_ROUTING_PRESET,
]
