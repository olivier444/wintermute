from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .workspace import TrainingWorkspace

__all__ = ["TrainingWorkspace"]


def __getattr__(name: str) -> object:
    if name == "TrainingWorkspace":
        from .workspace import TrainingWorkspace

        return TrainingWorkspace
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
