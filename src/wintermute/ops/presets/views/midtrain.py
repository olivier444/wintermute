from __future__ import annotations

from .builders import _composite_view_config, _view_preset, _view_component
from .transforms import text_processing, translation


PRESETS = [
    _view_preset(
        "translation-midtrain",
        _composite_view_config(
            [
                _view_component(
                    "mix40b-txt",
                    0.85,
                    include_prompt_in_loss=True,
                    shard_offset_ratio=0.5,
                ),
                _view_component(
                    "agentlans-en-fr-hq",
                    0.15,
                    record_transform=translation.agentlans_en_fr_hq_transform(),
                ),
            ],
            seed=2053,
        ),
    ),
    _view_preset(
        "summarization-midtrain",
        _composite_view_config(
            [
                _view_component(
                    "mix40b-txt",
                    0.81,
                    include_prompt_in_loss=True,
                    shard_offset_ratio=0.53,
                ),
                _view_component(
                    "agentlans-en-fr-hq",
                    0.05,
                    max_restarts=1,
                    record_transform=translation.agentlans_en_fr_hq_transform(),
                ),
                _view_component(
                    "xsum",
                    0.08,
                    max_restarts=7,
                    record_transform=text_processing.xsum_transform(),
                ),
                _view_component(
                    "orange-sum-abstract",
                    0.01,
                    max_restarts=14,
                    record_transform=text_processing.orange_sum_abstract_transform(),
                ),
            ],
            seed=2053,
        ),
    ),
]
