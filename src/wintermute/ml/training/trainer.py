from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn

from wintermute.ml.tasks.factory import TaskWrapper


class Trainer(ABC):
    """Common execution contract for training strategies."""

    task_wrapper: TaskWrapper

    @abstractmethod
    def train(self) -> nn.Module: ...
