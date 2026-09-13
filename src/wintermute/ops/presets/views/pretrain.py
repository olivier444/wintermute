from __future__ import annotations

from .builders import _composite_view_config, _view_preset, _snapshot_view_config, _view_component


PRESETS = [
    _view_preset(
        "default4-pretrain",
        _snapshot_view_config("default4-txt", split="train", include_prompt_in_loss=True),
    ),
    _view_preset("abstraction-pretrain", _snapshot_view_config("abstraction-v1", split="train")),
    _view_preset(
        "default4-abstraction-pretrain",
        _composite_view_config(
            [
                _view_component("default4-txt", 0.9),
                _view_component("abstraction-v1", 0.1),
            ],
            seed=7312,
        ),
    ),
    _view_preset(
        "default5-pretrain",
        _snapshot_view_config("default5-txt", split="train", include_prompt_in_loss=True),
    ),
    _view_preset(
        "default6-pretrain-deprecated",
        _snapshot_view_config("default6-txt.deprecated", split="train", include_prompt_in_loss=True),
    ),
    _view_preset(
        "mix40b-pretrain",
        _snapshot_view_config("mix40b-txt", split="train", include_prompt_in_loss=True),
    ),
    _view_preset(
        "mix40b-continued-pretrain",
        _snapshot_view_config(
            "mix40b-txt",
            split="train",
            include_prompt_in_loss=True,
            shard_offset_ratio=0.5,
        ),
    ),
    _view_preset(
        "mix40b-continued-pretrain2",
        _composite_view_config(
            [
                _view_component("mix40b-txt", 0.3, include_prompt_in_loss=True, shard_offset_ratio=0.7),
                _view_component("dclm-baseline-20b", 0.4, include_prompt_in_loss=True),
                _view_component("nem-math4p", 0.1, include_prompt_in_loss=True),                
                _view_component("ufw-l3-qa", 0.1, include_prompt_in_loss=True),    
                _view_component("fw2-fr-txt", 0.1, include_prompt_in_loss=True),                         
            ],
            seed=1312,
        ),
    ),
]
