from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import List


class TrainingMonitor(ABC):
    @property
    @abstractmethod
    def previews(self) -> Mapping[str, List[str]]: ...

    @abstractmethod
    def is_due(self) -> bool: ...

    @abstractmethod
    def run_due(self, *, global_step: int) -> None: ...

    @abstractmethod
    def run_initial(self, *, global_step: int) -> None: ...


class NoTrainingMonitor(TrainingMonitor):
    @property
    def previews(self) -> Mapping[str, List[str]]:
        return {}

    def is_due(self) -> bool:
        return False

    def run_due(self, *, global_step: int) -> None:
        del global_step

    def run_initial(self, *, global_step: int) -> None:
        del global_step
