from __future__ import annotations

from ..builders import _composite_view_config, _view_component, _view_preset


PRESETS = [
    _view_preset(
        "legacy-sft-1",
        _composite_view_config(
            [
                _view_component("default1-sft", 0.9),
                _view_component("default4-txt", 0.08),
                _view_component("abstraction-v1", 0.02),
            ],
            seed=7312,
        ),
    ),
    _view_preset(
        "legacy-sft-2",
        _composite_view_config(
            [
                _view_component("default2-sft", 0.85, max_restarts=1),
                _view_component("default4-txt", 0.13),
                _view_component("abstraction-v1", 0.02),
            ],
            seed=7312,
        ),
    ),
    _view_preset(
        "abstraction-sft",
        _composite_view_config(
            [
                _view_component("abstraction-v1", 0.7),
                _view_component("abstraction1-sft", 0.4, max_restarts=9),
                _view_component("default2-sft", 0.2, max_restarts=1),
                _view_component("default5-txt", 0.4),
            ],
            seed=7312,
        ),
    ),
    _view_preset(
        "assistant-sft",
        _composite_view_config(
            [
                _view_component("abstraction1-sft", 0.05, max_restarts=10),
                _view_component("default2-sft", 0.75, max_restarts=1),
                _view_component("default5-txt", 0.2),
            ],
            seed=7312,
        ),
    ),
    _view_preset(
        "assistant-sft-extended",
        _composite_view_config(
            [
                _view_component("abstraction1-sft", 0.1, max_restarts=20),
                _view_component("default2-sft", 0.7, max_restarts=1),
                _view_component("default5-txt", 0.2),
            ],
            seed=7312,
        ),
    ),
]
