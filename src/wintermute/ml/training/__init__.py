from __future__ import annotations

from typing import TYPE_CHECKING

from .trainer import Trainer

if TYPE_CHECKING:
    from .workspace import TrainingWorkspace

__all__ = ["Trainer", "TrainingWorkspace"]


def __getattr__(name: str) -> object:
    if name == "TrainingWorkspace":
        from .workspace import TrainingWorkspace

        return TrainingWorkspace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
